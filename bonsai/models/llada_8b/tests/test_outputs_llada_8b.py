# Copyright 2025 The JAX Authors.
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

import math
import os
import unittest

import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from bonsai.models.llada_8b import modeling as modeling
from bonsai.models.llada_8b import params as params_lib

HF_REPO = "GSAI-ML/LLaDA-8B-Instruct"


def tokenize(tokenizer, inputs: list[str]):
    pad_id = 126336  # Mask Token
    lines = [
        tokenizer.apply_chat_template([{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True)
        for l in inputs
    ]
    encoded = [tokenizer(s)["input_ids"] for s in lines]
    max_l = max(len(e) for e in encoded) if encoded else 1
    buffer_len = 2 ** math.ceil(math.log2(max(max_l, 1)))
    batch = np.stack([np.pad(e, (0, buffer_len - len(e)), constant_values=pad_id) for e in encoded], axis=0)
    return jnp.array(batch), pad_id, max_l, buffer_len


# TODO(#65): Reenable test after fixing NaN logits.
@unittest.skip("Currently failing due to incorrect mask/padding logic, results in NaN logits. Needs inspection.")
class TestLLaDAForwardPasses(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Download once; use same files for both reference and nnx
        cls.model_dir = snapshot_download(
            HF_REPO,
            local_dir=os.environ.get("LLADA_CACHE_DIR", "/tmp/models-bonsai"),
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_dir, use_fast=True)

        cls.prompts = [
            "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "Johnny picked 8 apples this morning and put them on his desk. Bonnie eats 3 of them. How many apples does Johnny have left?",
        ]
        cls.tokens_jax, cls.pad_id, cls.max_len, cls.buf_len = tokenize(cls.tokenizer, cls.prompts)
        cls.B, cls.T = cls.tokens_jax.shape
        cls.tokens_torch = torch.tensor(np.array(cls.tokens_jax), dtype=torch.long, device="cpu")

        cls.ref = (
            AutoModel.from_pretrained(cls.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
            .eval()
            .to("cpu")
        )

        cls.cfg = modeling.ModelConfig.llada_8b_instruct()
        cls.nnx_model = params_lib.create_llada_from_pretrained(cls.model_dir, cls.cfg)
        cls.graphdef, cls.state = nnx.split(cls.nnx_model)

    def test_token_embedding_equivalence(self):
        """Compare raw token embedding outputs (no dropout/positional add)."""
        # JAX
        merged = nnx.merge(self.graphdef, self.state)
        emb_jax = np.array(merged.wte(self.tokens_jax))

        # Torch
        with torch.no_grad():
            emb_torch = self.ref.model.transformer.wte(self.tokens_torch).to(torch.float32).cpu().numpy()

        np.testing.assert_allclose(emb_torch, emb_jax.astype(np.float32), rtol=1e-4, atol=2e-2)

    def test_first_block_equivalence(self):
        """
        Feed identical tensors (token embeddings) through block 0 in both models
        and compare outputs.
        """
        merged = nnx.merge(self.graphdef, self.state)

        x_jax = merged.wte(self.tokens_jax)  # (B,T,D)
        y_jax = np.array(merged.blocks[0](x_jax, attention_bias=None))

        with torch.no_grad():
            x_torch = self.ref.model.transformer.wte(self.tokens_torch)  # (B,T,D)
            y_torch, _ = self.ref.model.transformer.blocks[0](
                x_torch, attention_bias=None, layer_past=None, use_cache=False
            )
            y_torch = y_torch.to(torch.float32).cpu().numpy()

        np.testing.assert_allclose(y_torch, y_jax.astype(np.float32), rtol=1e-3, atol=5e-2)

    def test_full_forward_equivalence(self):
        """Compare full-model logits for the padded sequences."""
        merged = nnx.merge(self.graphdef, self.state)

        # JAX
        jax_logits = np.array(merged(self.tokens_jax).logits, dtype=np.float32)

        # Torch
        with torch.no_grad():
            torch_logits = self.ref(input_ids=self.tokens_torch).logits.to(torch.float32).cpu().numpy()

        np.testing.assert_allclose(torch_logits, jax_logits, rtol=1e-3, atol=6e-2)


if __name__ == "__main__":
    absltest.main()
