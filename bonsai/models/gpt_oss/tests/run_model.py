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

import tempfile
import time

import jax
import jax.numpy as jnp
import torch
from transformers import GptOssConfig, GptOssForCausalLM

from bonsai.models.gpt_oss import modeling, params


def run_model():
    # No small public gpt-oss checkpoint exists, so smoke-test on a tiny
    # randomly-initialized model round-tripped through safetensors.
    vocab_size, hidden, inter = 128, 64, 96
    hf_config = GptOssConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        attention_bias=True,
        sliding_window=4096,
        layer_types=["full_attention"] * 2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
        pad_token_id=0,
    )
    bonsai_config = modeling.GptOssConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        attention_bias=True,
        pad_token_id=0,
    )

    torch.manual_seed(0)
    hf_model = GptOssForCausalLM(hf_config)
    with torch.no_grad():
        for _, p in hf_model.named_parameters():
            p.normal_(mean=0.0, std=0.02)

    with tempfile.TemporaryDirectory() as d:
        hf_model.save_pretrained(d, safe_serialization=True)
        model = params.create_gpt_oss_from_pretrained(d, bonsai_config)

    @jax.jit
    def forward(m, ids):
        return m(ids)

    batch_size, seq_len = 4, 16
    dummy = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    _ = forward(model, dummy).block_until_ready()  # warmup / compile

    t0 = time.perf_counter()
    for _ in range(10):
        logits = forward(model, dummy).block_until_ready()
    print(f"Step time: {(time.perf_counter() - t0) / 10:.4f} s")

    print("Predicted next-token ids:", jnp.argmax(logits[:, -1, :], axis=-1))


if __name__ == "__main__":
    run_model()


__all__ = ["run_model"]
