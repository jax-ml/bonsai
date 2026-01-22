import dataclasses
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import AxisType
from transformers import AutoModelForCausalLM, AutoTokenizer

from bonsai.models.llama32 import modeling, params

# used to set highest precision on matrix multiplication for testing
jax.config.update("jax_default_matmul_precision", "highest")


def check_hf_token():
    try:
        access_token = os.environ["HF_TOKEN"]
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=access_token)
    except Exception as e:
        print("Failed to access HF_TOKEN or download tokenizer:")
        print(e)
        return True
    return False


@unittest.skipIf(check_hf_token(), "Skipping Llama32 output tests due to HF_TOKEN failure.")
class TestOutputsLlama32(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        cls.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        access_token = os.environ["HF_TOKEN"]

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, token=access_token)
        if cls.tokenizer.pad_token_id is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.tokenizer.padding_side = "left"
        cls.pad_id = cls.tokenizer.pad_token_id

        cls.torch_model = (
            AutoModelForCausalLM.from_pretrained(cls.model_name, token=access_token, dtype=torch.float32)
            .to(device=cls.torch_device, dtype=torch.float32)
            .eval()
        )

        fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value
        cls.mesh = jax.make_mesh((1, 1), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        cls.llama_config = dataclasses.replace(
            modeling.ModelConfig.llama3_2_1b(use_fsdp=False, use_tp=False),
            dtype=jnp.float32,
        )
        model_ckpt_path = snapshot_download(cls.model_name, token=access_token)
        graph_def, state = nnx.split(
            params.create_model_from_safe_tensors(model_ckpt_path, cls.llama_config, mesh=cls.mesh)
        )
        state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, state)
        cls.llama_model = nnx.merge(graph_def, state)

        cls.batch_size = 4
        cls.num_input_tokens = 6
        cls.relaxed_tol = 1e-3

    def _make_torch_input(self):
        messages = [{"role": "user", "content": "Summarize what a tokenizer does in one paragraph."}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.tokenizer([prompt], padding=False, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(device=self.torch_device) for k, v in batch.items()}

    def _make_hidden_states(self, shape):
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        return jx, tx

    def test_embedder(self):
        nm = self.llama_model.embedder
        tm = self.torch_model.model.embed_tokens

        tx = torch.randint(0, self.torch_model.config.vocab_size, size=(self.batch_size, self.num_input_tokens))
        jx = jnp.array(tx.cpu().detach().numpy())

        jy = nm(jx)
        ty = tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_rms_norm(self):
        nm = self.llama_model.layers[0].input_layernorm
        tm = self.torch_model.model.layers[0].input_layernorm

        shape = (self.batch_size, self.num_input_tokens, self.llama_config.hidden_size)
        jx, tx = self._make_hidden_states(shape)

        jy = nm(jx)
        ty = tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_q_proj(self):
        nm = self.llama_model.layers[0].self_attn.q_proj
        tm = self.torch_model.model.layers[0].self_attn.q_proj.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.llama_config.hidden_size)
        jx, tx = self._make_hidden_states(shape)

        jy = nm(jx).reshape(
            self.batch_size, self.num_input_tokens, self.llama_config.num_attention_heads, self.llama_config.head_dim
        )
        ty = tm(tx).reshape(
            self.batch_size, self.num_input_tokens, self.llama_config.num_attention_heads, self.llama_config.head_dim
        )
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_k_proj(self):
        nm = self.llama_model.layers[0].self_attn.k_proj
        tm = self.torch_model.model.layers[0].self_attn.k_proj.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.llama_config.hidden_size)
        jx, tx = self._make_hidden_states(shape)

        jy = nm(jx).reshape(
            self.batch_size, self.num_input_tokens, self.llama_config.num_key_value_heads, self.llama_config.head_dim
        )
        ty = tm(tx).reshape(
            self.batch_size, self.num_input_tokens, self.llama_config.num_key_value_heads, self.llama_config.head_dim
        )
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_v_proj(self):
        nm = self.llama_model.layers[0].self_attn.v_proj
        tm = self.torch_model.model.layers[0].self_attn.v_proj.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.llama_config.hidden_size)
        jx, tx = self._make_hidden_states(shape)

        jy = nm(jx).reshape(
            self.batch_size, self.num_input_tokens, self.llama_config.num_key_value_heads, self.llama_config.head_dim
        )
        ty = tm(tx).reshape(
            self.batch_size, self.num_input_tokens, self.llama_config.num_key_value_heads, self.llama_config.head_dim
        )
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_o_proj(self):
        nm = self.llama_model.layers[0].self_attn.o_proj
        tm = self.torch_model.model.layers[0].self_attn.o_proj.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.llama_config.hidden_size)
        jx, tx = self._make_hidden_states(shape)

        jy = nm(jx)
        ty = tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_mlp(self):
        nm = self.llama_model.layers[0].mlp
        tm = self.torch_model.model.layers[0].mlp.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.llama_config.hidden_size)
        jx, tx = self._make_hidden_states(shape)

        jy = nm(jx)
        ty = tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_lm_head(self):
        nm = self.llama_model.embedder
        tm = self.torch_model.lm_head.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.llama_config.hidden_size)
        jx, tx = self._make_hidden_states(shape)

        jy = nm.decode(jx)
        ty = tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_full_logits(self):
        t_inputs = self._make_torch_input()
        n_tokens = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        attention_mask = jnp.array(t_inputs["attention_mask"].detach().cpu().numpy())
        segment_ids = attention_mask.astype(jnp.int32)

        with torch.no_grad():
            t_logits = self.torch_model(**t_inputs).logits

        n_logits = self.llama_model(n_tokens, segment_ids, cache=None, attn_mask=None)
        np.testing.assert_allclose(n_logits, t_logits.detach().cpu().numpy(), rtol=5e-2, atol=5e-2)

    def test_forward_logits(self):
        t_inputs = self._make_torch_input()
        n_tokens = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        attention_mask = jnp.array(t_inputs["attention_mask"].detach().cpu().numpy())
        batch_size, token_len = n_tokens.shape
        cache = self.llama_model.init_cache(self.llama_config, batch_size, token_len, generate_steps=1)

        with torch.no_grad():
            t_logits = self.torch_model(**t_inputs).logits

        n_logits, _ = modeling.forward(self.llama_model, cache, n_tokens, self.pad_id, attention_mask=attention_mask)
        np.testing.assert_allclose(n_logits, t_logits[:, -1].detach().cpu().numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    absltest.main()
