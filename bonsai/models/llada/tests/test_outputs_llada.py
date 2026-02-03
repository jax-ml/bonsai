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
from transformers import AutoTokenizer, AutoModel, AutoConfig

from bonsai.models.llada import modeling, params
from flax import nnx
from typing import Any
import tempfile
from safetensors.torch import save_file
import os

# used to set highest precision on matrix multiplication for testing
jax.config.update("jax_default_matmul_precision", "highest")


def create_llada_for_testing(use_shd: bool) -> tuple[Any, torch.nn.Module, modeling.ModelConfig, modeling.LLaDAModel]:
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    baseline_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    factor = 32
    baseline_config.n_layers = 1
    baseline_config.d_model //= factor
    baseline_config.n_heads //= factor
    baseline_config.n_kv_heads //= factor
    baseline_config.vocab_size //= factor
    baseline_config.embedding_size //= factor
    baseline_config.mlp_hidden_size //= factor

    baseline_model = AutoModel.from_config(baseline_config, trust_remote_code=True, dtype=torch.float32)
    bonsai_config = modeling.ModelConfig(
        dtype=jnp.float32,
        d_model=4096 // factor,
        n_heads=32 // factor,
        n_kv_heads=32 // factor,
        n_layers=1,
        embedding_dropout=0.0,
        max_sequence_length=4096,
        rope_theta=500000.0,
        include_qkv_bias=False,
        include_bias=False,
        vocab_size=126464 // factor,
        embedding_size=126464 // factor,
        mlp_hidden_size=12288 // factor,
        shd_cfg=modeling.ShardingConfig.default(use_shd, use_shd),
        return_hidden_states=False,
    )
    bonsai_model = modeling.LLaDAModel(bonsai_config, rngs=nnx.Rngs(0))

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "ref.safetensors")
        save_file(baseline_model.state_dict(), filename)
        bonsai_model = params.create_llada_from_pretrained(temp_dir, bonsai_config)

    return baseline_config, baseline_model, bonsai_config, bonsai_model


class TestModuleForwardPasses(absltest.TestCase):
    """Numerical tests for LLaDA
    Full model takes up 40+ Gb. Using smaller model for numerical testing.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)
        cls.baseline_config, cls.baseline_model, cls.bonsai_config, cls.bonsai_model = create_llada_for_testing(False)

        torch.manual_seed(42)
        cls.batch_size, cls.seq_len = 2, 100

    def make_torch_inputs(self):
        shape = (self.batch_size, self.seq_len)
        return dict(
            input_ids=torch.randint(0, self.bonsai_config.vocab_size, shape, dtype=torch.int32),
            attention_mask=torch.ones(shape, dtype=torch.int32),
        )

    def test_block_ops(self):
        tm = self.baseline_model.model.transformer["blocks"][0]
        nm = self.bonsai_model.blocks[0]

        shape = (self.batch_size, self.seq_len, self.bonsai_config.d_model)
        jx = jax.random.normal(jax.random.key(0), shape, jnp.float32)
        tx = torch.tensor(jx)

        for attr in ["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"]:
            ty, ny = getattr(tm, attr)(tx), getattr(nm, attr)(jx, out_sharding=None)
            np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), err_msg=attr)

        # norms
        for attr in ["attn_norm", "ff_norm"]:
            ty, ny = getattr(tm, attr)(tx), getattr(nm, attr)(jx)
            np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=5e-7, atol=5e-7, err_msg=attr)

        # ff_out
        shape = (self.batch_size, self.seq_len, self.bonsai_config.mlp_hidden_size)
        jx = jax.random.normal(jax.random.key(0), shape, jnp.float32)
        tx = torch.tensor(jx)

        ty, ny = tm.ff_out(tx), nm.ff_out(jx, out_sharding=None)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), err_msg="ff_out")

    def test_wte(self):
        tm = self.baseline_model.model.transformer["wte"]
        nm = self.bonsai_model.wte

        tx = self.make_torch_inputs()["input_ids"]
        jx = jnp.array(tx.detach().cpu().numpy())

        ty, ny = tm(tx), nm(jx, out_sharding=None)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy())

    def test_rmsnorm(self):
        tm = self.baseline_model.model.transformer["ln_f"]
        nm = self.bonsai_model.ln_f

        shape = (self.batch_size, self.seq_len, self.bonsai_config.d_model)
        jx = jax.random.normal(jax.random.key(0), shape, jnp.float32)
        tx = torch.tensor(jx)

        ty, ny = tm(tx), nm(jx)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=5e-7, atol=5e-7)

    def test_ff_out(self):
        tm = self.baseline_model.model.transformer["ff_out"]
        nm = self.bonsai_model.ff_out

        shape = (self.batch_size, self.seq_len, self.bonsai_config.d_model)
        jx = jax.random.normal(jax.random.key(0), shape, jnp.float32)
        tx = torch.tensor(jx)

        ty, ny = tm(tx), nm(jx, out_sharding=None)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=2e-6, atol=2e-6)

    def test_sin_cos(self):
        tm = self.baseline_model.model.transformer.blocks[0].rotary_emb
        t_sin, t_cos = tm.get_rotary_embedding(self.seq_len, "cpu")

        head_dim = self.bonsai_config.d_model // self.bonsai_config.n_heads
        rope_theta = self.bonsai_config.rope_theta
        segment_ids = jnp.ones((self.batch_size, self.seq_len), jnp.int32)

        left_pads = modeling.count_left_pads(segment_ids)
        start_ind = left_pads.reshape((-1, 1))
        position_ids = modeling.compute_positions_from_segment_ids(segment_ids) + start_ind
        n_sin, n_cos = modeling._generate_pos_embeddings(position_ids, head_dim, rope_theta)

        for ny, ty, name in [(n_sin, t_sin, "sin"), (n_cos, t_cos, "cos")]:
            ty_part = ty[0, 0, :, :64].detach().cpu().numpy()
            np.testing.assert_allclose(ny[0, ...], ty_part, rtol=5e-6, atol=5e-6, err_msg=name)

    def test_block(self):
        tm = self.baseline_model.model.transformer.blocks[0]
        nm = self.bonsai_model.blocks[0]

        shape = (self.batch_size, self.seq_len, self.bonsai_config.d_model)
        head_dim = self.bonsai_config.d_model // self.bonsai_config.n_heads
        rope_theta = self.bonsai_config.rope_theta
        jx = jax.random.normal(jax.random.key(0), shape, jnp.float32)
        segment_ids = jnp.ones((self.batch_size, self.seq_len), jnp.int32)
        tx = torch.tensor(jx)

        left_pads = modeling.count_left_pads(segment_ids)
        start_ind = left_pads.reshape((-1, 1))
        position_ids = modeling.compute_positions_from_segment_ids(segment_ids) + start_ind
        sin, cos = modeling._generate_pos_embeddings(position_ids, head_dim, rope_theta)

        ty = tm(tx, attention_bias=None, layer_past=None, use_cache=False)[0]
        ny = nm(jx, sin, cos, None, jax.random.key(0))
        np.testing.assert_allclose(ny[0], ty.detach().cpu().numpy(), rtol=5e-7, atol=5e-7)

    def test_full(self):
        tm = self.baseline_model
        nm = self.bonsai_model

        t_inputs = self.make_torch_inputs()
        n_inputs = {k: jnp.array(v.detach().cpu().numpy()) for k, v in t_inputs.items()}
        n_inputs["attention_mask"] = n_inputs["attention_mask"] > -1

        ty, ny = tm(**t_inputs, use_cache=False).logits, nm(**n_inputs, key=jax.random.key(0))
        np.testing.assert_allclose(ny[0], ty.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)


class TestGenerationSmall(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_name = "GSAI-ML/LLaDA-8B-Instruct"

        cls.mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)
        cls.processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float32)
        if cls.processor.padding_side != "left":
            cls.processor.padding_side = "left"
        assert cls.processor.pad_token_id != 126336

        cls.baseline_config, cls.baseline_model, cls.bonsai_config, cls.bonsai_model = create_llada_for_testing(True)
        torch.manual_seed(42)

    def test_generate(self):
        prompts = [
            "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
        ]
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        prompts = [
            self.processor.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
            for message in messages
        ]

        encoded_outputs = self.processor(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = jnp.array(encoded_outputs["input_ids"].detach().cpu().numpy())
        attention_mask = encoded_outputs["attention_mask"].detach().cpu().numpy() > 0.5

        out = modeling.generate(
            self.bonsai_model,
            input_ids,
            attention_mask,
            steps=128,
            gen_length=128,
            block_length=128,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=126336,
            logits_eos_inf=False,
            confidence_eos_eot_inf=False,
            key=jax.random.key(0),
        )


if __name__ == "__main__":
    absltest.main()
