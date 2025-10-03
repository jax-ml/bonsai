import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl import flags
from absl.testing import absltest, parameterized
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3PreTrainedModel

from bonsai.models.qwen3 import modeling, params
from bonsai.models.qwen3.tests.run_model import tokenize

RUN_ALL = True


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        model_name: str = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        ## models
        self.torch_model = Qwen3ForCausalLM.from_pretrained(model_name, dtype="auto")
        self.bonsai_config = modeling.ModelCfg.qwen3_0_6b()
        model_ckpt_path = snapshot_download("Qwen/Qwen3-0.6B")
        self.nnx_model = params.create_model_from_safe_tensors(model_ckpt_path, self.bonsai_config)

        self.batch_size = 32
        self.num_input_tokens = 5
        self.token_dim = 151936
        self.cache_size, self.gen_steps = 128, 10

    def _check_batched(self, left_pads, torch_logits, nnx_logits):
        """Checks logits from batched inputs"""
        max_len = torch_logits.shape[-2]
        for lp, tl, nl in zip(left_pads, torch_logits, nnx_logits):
            torch.testing.assert_close(torch.tensor(nl)[lp:max_len, :], tl[lp:, :], rtol=1e-3, atol=1e-3)

    def _setup_torch_attn(
        self, input_embeddings, attention_mask=None, position_ids=None, higher_precision: bool = False
    ):
        # This function replicates the forward method from Qwen3Model in transformers/models/qwen3/modeling_qwen3.py

        past_key_values = DynamicCache(config=self.torch_model.config)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + input_embeddings.shape[1], device=self.torch_model.device
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.torch_model.config,
                "input_embeds": input_embeddings,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.torch_model.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.torch_model.model.rotary_emb(input_embeddings, position_ids)
        out = dict(
            hidden_states=input_embeddings,
            attention_mask=causal_mask_mapping[self.torch_model.model.layers[0].attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if higher_precision:
            for k in out:
                if hasattr(out[k], "to"):
                    out[k] = out[k].to(torch.float32)
        return out

    def _nnx_forward_logits(self, cache, tokens, dtype=None):
        segment_ids = 1 * (tokens != self.tokenizer.pad_token_id)
        x = self.nnx_model.embedder.encode(tokens)
        if dtype is not None:
            x = x.astype(dtype)
        right_pads = modeling.num_right_pad(segment_ids[0])
        for i, layer in enumerate(self.nnx_model.layers):
            x = layer(x, cache[i], segment_ids, right_pads)
            if dtype is not None:
                x = x.astype(dtype)
        nnx_logits = self.nnx_model.lm_head(self.nnx_model.final_norm(x))
        return nnx_logits

    def _process_hf_tokens(self, query: list[str]):
        """Converts queries into tokens for huggingface that is consistent with Bonsai."""
        messages = [{"role": "user", "content": s} for s in query]
        text = [
            self.tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for m in messages
        ]
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True, padding_side="left").to(
            self.torch_model.device
        )
        tmp = model_inputs["attention_mask"]
        num_zeros = tmp.shape[1] - tmp.sum(dim=-1)
        model_inputs["left_pads"] = num_zeros
        pos_ids = torch.arange(tmp.shape[1]) - num_zeros.reshape(-1, 1)
        pos_ids[pos_ids < 0] = 2**30
        model_inputs["position_ids"] = pos_ids

        return model_inputs

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_embedder(self):
        nm = self.nnx_model.embedder
        tm = self.torch_model.model.embed_tokens
        tm.eval()

        tx = torch.randint(0, self.token_dim, size=(self.batch_size, self.num_input_tokens))
        jx = tx.cpu().detach().numpy()

        jy, ty = nm.encode(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_decoder_layer(self):
        nm = self.nnx_model.layers[0]
        tm = self.torch_model.model.layers[0]
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024))
        tx = torch.tensor(jx)
        tm = tm.to(torch.float32)

        nnx_cache = modeling.init_cache(
            num_layers=self.bonsai_config.num_layers,
            batch_size=self.batch_size,
            cache_size=self.cache_size,
            num_kv_heads=self.bonsai_config.num_kv_heads,
            head_dim=self.bonsai_config.head_dim,
            dtype=jnp.float32,
        )

        jy = nm(jx, nnx_cache[0], jnp.ones((32, 5)), 0)
        torch_inputs = self._setup_torch_attn(tx, higher_precision=True)
        ty = tm(**torch_inputs)

        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_all_decoder_layers(self):
        nnx_cache = modeling.init_cache(
            num_layers=self.bonsai_config.num_layers,
            batch_size=self.batch_size,
            cache_size=self.cache_size,
            num_kv_heads=self.bonsai_config.num_kv_heads,
            head_dim=self.bonsai_config.head_dim,
            dtype=jnp.float32,
        )

        for i, (nm, tm) in enumerate(zip(self.nnx_model.layers, self.torch_model.model.layers)):
            tm.eval()

            jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024))
            tx = torch.tensor(jx)
            tm = tm.to(torch.float32)

            jy = nm(jx, nnx_cache[i], jnp.ones((32, 5)), 0)
            torch_inputs = self._setup_torch_attn(tx, higher_precision=True)
            ty = tm(**torch_inputs)

            torch.testing.assert_close(torch.tensor(jy), ty, atol=1e-3, rtol=1e-3)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_rmsnorm(self):
        nm = self.nnx_model.layers[0].input_layernorm
        tm = self.torch_model.model.layers[0].input_layernorm
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024), dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_self_attn(self):
        nm = self.nnx_model.layers[0].attn
        tm = self.torch_model.model.layers[0].self_attn
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024))
        tm = tm.to(torch.float32)

        tx = torch.tensor(jx)

        nnx_cache = modeling.init_cache(
            num_layers=self.bonsai_config.num_layers,
            batch_size=self.batch_size,
            cache_size=self.cache_size,
            num_kv_heads=self.bonsai_config.num_kv_heads,
            head_dim=self.bonsai_config.head_dim,
            dtype=jnp.float32,
        )

        jy = nm(jx, nnx_cache[0], jnp.ones((32, 5), dtype=jnp.float32), 0)
        torch_inputs = self._setup_torch_attn(tx, higher_precision=True)
        ty = tm(**torch_inputs)[0]

        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_qnorm(self):
        nm = self.nnx_model.layers[0].attn.q_norm
        tm = self.torch_model.model.layers[0].self_attn.q_norm
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 16, 128), dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)

        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_knorm(self):
        nm = self.nnx_model.layers[0].attn.q_norm
        tm = self.torch_model.model.layers[0].self_attn.q_norm
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 8, 128), dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)

        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_q_proj(self):
        nm = self.nnx_model.layers[0].attn.q_proj
        tm = self.torch_model.model.layers[0].self_attn.q_proj
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024), dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy = nm(jx)
        ty = tm(tx).reshape(32, 5, 16, 128)

        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_k_proj(self):
        nm = self.nnx_model.layers[0].attn.k_proj
        tm = self.torch_model.model.layers[0].self_attn.k_proj
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024), dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy = nm(jx)
        ty = tm(tx).reshape(32, 5, 8, 128)

        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_o_proj(self):
        nm = self.nnx_model.layers[0].attn.o_proj
        tm = self.torch_model.model.layers[0].self_attn.o_proj
        tm.eval()

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 16, 128), dtype=jnp.bfloat16)
        tx = torch.tensor(jx).reshape(32, 5, -1)

        jy = nm(jx)
        ty = tm(tx)

        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_mlp(self):
        nm = self.nnx_model.layers[0].mlp
        tm = self.torch_model.model.layers[0].mlp
        tm.eval()
        tm = tm.to(torch.float32)

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024))
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty, rtol=2e-5, atol=1e-5)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_lm_head(self):
        nm = self.nnx_model.lm_head
        tm = self.torch_model.lm_head
        tm.eval()
        tm = tm.to(torch.float32)

        jx = jax.random.normal(jax.random.key(0), shape=(32, 5, 1024))
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    @unittest.skipIf(not RUN_ALL, "skipping submodule tests")
    def test_sin_cos(self):
        hidden_states = torch.ones((2, 10, 128))
        jp = jnp.stack([jnp.arange(10), jnp.arange(10)])
        js, jc = modeling._generate_pos_embeddings(jp, 128)
        tc, ts = self.torch_model.model.rotary_emb(hidden_states, torch.tensor(jp))
        tc, ts = tc[:, :, :64], ts[:, :, :64]
        torch.testing.assert_close(torch.tensor(js), ts)
        torch.testing.assert_close(torch.tensor(jc), tc)

    def test_full(self):
        query = ["Why is the sky blue instead of any other color like purple?"]
        tokens, max_len = tokenize(self.tokenizer, query)

        self.torch_model.eval()
        self.torch_model = self.torch_model.to(torch.float32)

        nnx_cache = modeling.init_cache(
            num_layers=self.bonsai_config.num_layers,
            batch_size=len(query),
            cache_size=self.cache_size,
            num_kv_heads=self.bonsai_config.num_kv_heads,
            head_dim=self.bonsai_config.head_dim,
            dtype=jnp.float32,
        )

        nnx_logits = self._nnx_forward_logits(nnx_cache, tokens, jnp.float32)
        torch_inputs = self._process_hf_tokens(query)
        torch_logits = self.torch_model(**torch_inputs).logits
        torch.testing.assert_close(torch.tensor(nnx_logits)[:, :max_len, :], torch_logits, rtol=1e-3, atol=1e-3)

    def test_full_batched(self):
        query = ["Why is the sky blue instead of any other color like purple?", "Who am I?"]
        tokens, max_len = tokenize(self.tokenizer, query)

        self.torch_model.eval()
        self.torch_model = self.torch_model.to(torch.float32)

        nnx_cache = modeling.init_cache(
            num_layers=self.bonsai_config.num_layers,
            batch_size=len(query),
            cache_size=self.cache_size,
            num_kv_heads=self.bonsai_config.num_kv_heads,
            head_dim=self.bonsai_config.head_dim,
            dtype=jnp.float32,
        )

        nnx_logits = self._nnx_forward_logits(nnx_cache, tokens, jnp.float32)
        torch_inputs = self._process_hf_tokens(query)
        torch_logits = self.torch_model(**torch_inputs).logits

        self._check_batched(torch_inputs["left_pads"], torch_logits, nnx_logits)


if __name__ == "__main__":
    absltest.main()
