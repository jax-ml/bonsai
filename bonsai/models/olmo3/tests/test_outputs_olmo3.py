# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from jax.sharding import AxisType
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


from bonsai.models.olmo3 import modeling, params
import unittest
from huggingface_hub import snapshot_download
from tqdm import tqdm
from jax import P


# used to set highest precision on matrix multiplication for testing
jax.config.update("jax_default_matmul_precision", "highest")


class TestModuleForwardPasses(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_id = "allenai/Olmo-3-1025-7B"
        torch_device = "cpu"

        cls.mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        model_ckpt_path = snapshot_download(model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.padding_side = "left"

        cls.bonsai_config = modeling.ModelConfig.olmo3_7b(False, False, jnp.float32)
        cls.bonsai_model = params.create_olmo3_from_pretrained(model_ckpt_path, cls.bonsai_config, mesh=cls.mesh)
        cls.baseline_model = AutoModelForCausalLM.from_pretrained(model_id).to(device=torch_device, dtype=torch.float32)
        cls.baseline_model.eval()

        # Constants
        torch.manual_seed(0)

    # TODO: Need to test batching
    def _make_torch_inputs(self):
        # message = ["Language modeling is ", "really fun"]
        message = ["Language modeling is "]
        return self.tokenizer(message, return_tensors="pt", return_token_type_ids=False, padding=True)

    def _process_torch_decoder_inputs(self, t_inputs):
        # Adapted from transformers.models.olmo3.modeling_olmo3.py
        input_ids = t_inputs["input_ids"]
        attention_mask = t_inputs["attention_mask"]

        inputs_embeds: torch.Tensor = self.baseline_model.model.embed_tokens(input_ids)

        past_seen_tokens = 0
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.baseline_model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings_mapping = {
            "sliding_attention": self.baseline_model.model.rotary_embs["sliding_attention"](
                hidden_states, position_ids
            ),
            "full_attention": self.baseline_model.model.rotary_embs["full_attention"](hidden_states, position_ids),
        }
        return dict(
            hidden_states=hidden_states,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings_mapping,
        )

    @unittest.skip("Done")
    def test_embedding(self):
        tm = self.baseline_model.model.embed_tokens
        nm = self.bonsai_model.model.embed_tokens

        tx = self._make_torch_inputs()["input_ids"]
        jx = jnp.array(tx.detach().cpu().numpy())

        ty, jy = tm(tx), nm(jx, out_sharding=None)
        np.testing.assert_allclose(jy, ty.detach().cpu().numpy())

    def test_attn_components(self):
        tm = self.baseline_model.model.layers[0].self_attn
        nm = self.bonsai_model.model.layers[0].self_attn

        jx = jax.random.normal(jax.random.key(0), (2, 4, 4096))
        tx = torch.tensor(jx)

        for comp in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            ty = getattr(tm, comp)(tx)
            jy = getattr(nm, comp)(jx, out_sharding=None)
            np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5, err_msg=comp)

        for comp in ["q_norm", "k_norm"]:
            ty, jy = getattr(tm, comp)(tx), getattr(nm, comp)(jx)
            np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), err_msg=comp)

    def test_mlp(self):
        tm = self.baseline_model.model.layers[0].mlp
        nm = self.bonsai_model.model.layers[0].mlp

        jx = jax.random.normal(jax.random.key(0), (2, 4, 4096))
        tx = torch.tensor(jx)

        ty, jy = tm(tx), nm(jx)
        np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    def test_decoder_layer(self):
        tm = self.baseline_model.model.layers
        nm = self.bonsai_model.model.layers

        orig_t_inputs = self._make_torch_inputs()
        jseg = jnp.array(orig_t_inputs["attention_mask"].detach().cpu().numpy())
        orig_t_inputs = self._process_torch_decoder_inputs(orig_t_inputs)

        t_inputs = {k: v.clone() for k, v in orig_t_inputs.items() if hasattr(v, "clone")}
        jx = jnp.array(t_inputs["hidden_states"].detach().cpu().numpy())
        b, t, _ = jx.shape
        cache = modeling.init_cache(self.bonsai_config, b, t, 1, jnp.float32)

        t = jseg.shape[1]
        causal_mask = modeling.make_causal_mask(cache[0], t, out_sharding=None)
        sliding_mask = modeling.make_window_mask(
            cache[0], t, slide_size=self.bonsai_config.sliding_window, out_sharding=None
        )

        exceptions = []
        for i, lt in enumerate(tqdm(self.bonsai_config.layer_types)):
            t_inputs["attention_mask"] = orig_t_inputs["attention_mask"][lt.value]
            t_inputs["position_embeddings"] = orig_t_inputs["position_embeddings"][lt.value]
            jmask = causal_mask if lt == modeling.AttentionMode.FULL else sliding_mask

            ty = tm[i](**t_inputs)
            jy = nm[i](jx, cache[i], jseg, jmask)

            try:
                np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), err_msg=f"decoder {i}", atol=1e-5, rtol=1e-5)
            except Exception as e:
                exceptions.append(e)

        if exceptions:
            raise AssertionError("Found errors in decoder layers:\n" + "\n".join(map(str, exceptions)))

    def test_model(self):
        tm = self.baseline_model
        nm = self.bonsai_model

        t_inputs = self._make_torch_inputs()
        jx = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        jseg = jnp.array(t_inputs["attention_mask"].detach().cpu().numpy())
        b, t = jx.shape
        cache = modeling.init_cache(self.bonsai_config, b, t, 1, jnp.float32)

        with torch.no_grad():
            ty = tm(**t_inputs).logits
        jy = nm(jx, jseg, cache)

        np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


@unittest.skip("Done")
class TestSharding(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_id = "allenai/Olmo-3-1025-7B"

        cls.mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        model_ckpt_path = snapshot_download(model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.padding_side = "left"

        cls.bonsai_config = modeling.ModelConfig.olmo3_7b(True, True, jnp.float32)
        cls.bonsai_model = params.create_olmo3_from_pretrained(model_ckpt_path, cls.bonsai_config, mesh=cls.mesh)

        # Constants
        torch.manual_seed(0)

    def _make_torch_input(self):
        message = ["Language modeling is "]
        return self.tokenizer(message, return_tensors="pt", return_token_type_ids=False, padding=True)

    def test_full(self):
        nm = self.bonsai_model
        fsdp = modeling.ShardMode.FSDP.value

        t_inputs = self._make_torch_input()
        jx = jnp.array(t_inputs["input_ids"].detach().cpu().numpy(), out_sharding=P(fsdp))
        jseg = jnp.array(t_inputs["attention_mask"].detach().cpu().numpy(), out_sharding=P(fsdp))
        b, t = jx.shape
        cache = modeling.init_cache(self.bonsai_config, b, t, 1, jnp.float32)

        nm(jx, jseg, cache)


if __name__ == "__main__":
    absltest.main()
