# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, T5Gemma2Model

from bonsai.models.t5gemma2 import modeling, params


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        model_name: str = "google/t5gemma-2-270m-270m"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_ckpt_path = snapshot_download(model_name)

        # HuggingFace reference model
        self.torch_model = T5Gemma2Model.from_pretrained(model_name, torch_dtype=torch.float32).eval()

        # Bonsai model (no vision to match text-only comparison)
        self.bonsai_config = modeling.T5Gemma2Config.t5gemma2_270m_270m(with_vision=False)
        graph_def, state = nnx.split(params.create_model_from_safe_tensors(model_ckpt_path, self.bonsai_config))
        state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, state)
        self.nnx_model = nnx.merge(graph_def, state)

        self.batch_size = 2
        self.num_input_tokens = 8
        self.relaxed_tol = 1e-3

        self.enc_cfg = self.bonsai_config.encoder.text_config

    def _to_torch(self, jax_arr):
        return torch.tensor(np.array(jax_arr, dtype=np.float32))

    def _assert_close(self, jax_out, torch_out):
        torch.testing.assert_close(
            self._to_torch(jax_out),
            torch_out.float(),
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_embedder(self):
        nm = self.nnx_model.embedder
        tm = self.torch_model.encoder.embed_tokens

        tx = torch.randint(0, self.enc_cfg.vocab_size, size=(self.batch_size, self.num_input_tokens))
        jx = jnp.array(tx.cpu().numpy())

        self._assert_close(nm(jx), tm(tx))

    def test_rms_norm(self):
        nm = self.nnx_model.encoder.blocks[0].pre_attention_norm
        tm = self.torch_model.encoder.layers[0].pre_self_attn_layernorm

        shape = (self.batch_size, self.num_input_tokens, self.enc_cfg.embed_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = self._to_torch(jx)

        self._assert_close(nm(jx), tm(tx))

    def test_feed_forward(self):
        nm = self.nnx_model.encoder.blocks[0].mlp
        tm = self.torch_model.encoder.layers[0].mlp

        shape = (self.batch_size, self.num_input_tokens, self.enc_cfg.embed_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = self._to_torch(jx)

        self._assert_close(nm(jx), tm(tx))

    def test_full_encoder(self):
        text = "Translate English to French: Hello, how are you?"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [modeling.BOS_TOKEN] + tokens

        jx = jnp.array([tokens], dtype=jnp.int32)
        tx = torch.tensor([tokens], dtype=torch.long)

        jy = self.nnx_model.encoder(jx)
        ty = self.torch_model.encoder(tx).last_hidden_state
        self._assert_close(jy, ty)

    def test_full_encoder_batched(self):
        texts = [
            "Translate English to French: Hello, how are you?",
            "Translate English to German: The weather is nice today.",
        ]
        tokenized = [[modeling.BOS_TOKEN] + self.tokenizer.encode(t, add_special_tokens=False) for t in texts]
        # Pad to same length
        max_len = max(len(t) for t in tokenized)
        padded = [t + [self.enc_cfg.pad_token_id] * (max_len - len(t)) for t in tokenized]

        jx = jnp.array(padded, dtype=jnp.int32)
        tx = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor([[1] * len(t) + [0] * (max_len - len(t)) for t in tokenized])

        jy = self.nnx_model.encoder(jx)
        ty = self.torch_model.encoder(tx, attention_mask=attention_mask).last_hidden_state
        self._assert_close(jy, ty)

    def test_full_model(self):
        text = "Translate English to French: Hello, how are you?"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [modeling.BOS_TOKEN] + tokens

        encoder_ids = jnp.array([tokens], dtype=jnp.int32)
        decoder_ids = jnp.array([[modeling.BOS_TOKEN]], dtype=jnp.int32)

        torch_encoder_ids = torch.tensor([tokens], dtype=torch.long)
        torch_decoder_ids = torch.tensor([[modeling.BOS_TOKEN]], dtype=torch.long)

        # Bonsai model returns (logits, encoder_outputs)
        jax_logits, _ = self.nnx_model(encoder_ids, decoder_ids, decode=False)

        # HF T5Gemma2Model returns decoder hidden states; compute logits via tied embeddings
        torch_out = self.torch_model(input_ids=torch_encoder_ids, decoder_input_ids=torch_decoder_ids)
        torch_hidden = torch_out.last_hidden_state
        embed_table = self.torch_model.encoder.embed_tokens.weight
        torch_logits = torch.einsum("btd,vd->btv", torch_hidden, embed_table)

        self._assert_close(jax_logits, torch_logits)


if __name__ == "__main__":
    absltest.main()
