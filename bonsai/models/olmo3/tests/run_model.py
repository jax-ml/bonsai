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

"""Run a small inference example for Olmo3."""

import jax
import jax.numpy as jnp
import torch
import tqdm
from huggingface_hub import snapshot_download
from jax.sharding import AxisType
from transformers import AutoTokenizer

from bonsai.models.olmo3 import modeling, params
from bonsai.utils import Sampler


def run_model():
    model_name: str = "allenai/Olmo-3-1025-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value

    mesh = jax.make_mesh(((1, 1)), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)

    model_ckpt_path = snapshot_download(model_name)

    bonsai_config = modeling.ModelConfig.olmo3_7b(False, False, jnp.float32)
    bonsai_model = params.create_olmo3_from_pretrained(model_ckpt_path, bonsai_config, mesh=mesh)

    # Make inputs
    input_text = ["Language modeling is "]
    torch_input = tokenizer(input_text, return_tensors="pt", padding=True)
    tx, tseg = torch_input["input_ids"], torch_input["attention_mask"]
    jx = jnp.array(tx.detach().cpu().numpy())
    jseg = jnp.array(tseg.detach().cpu().numpy())

    gen_steps = 32
    batch_size, num_tokens = jx.shape
    cache = modeling.init_cache(bonsai_config, batch_size, num_tokens, gen_steps, jnp.float32)

    source_key = jax.random.key(0)
    sampler = jax.jit(Sampler(temperature=1.0, top_p=0.8, top_k=10))

    all_tokens = [jx]
    pbar = tqdm.trange(gen_steps, desc="Generating output")

    # Prefill
    jseg = jnp.ones((batch_size, num_tokens))
    out, cache = modeling.forward(bonsai_model, cache, jx, jseg)

    source_key, key = jax.random.split(source_key)
    jx = sampler(out, key=key)
    pbar.update(1)
    all_tokens.append(jx)

    # Decode
    num_tokens = 1
    jseg = jnp.ones((batch_size, num_tokens))

    for _ in pbar:
        out, cache = modeling.forward(bonsai_model, cache, jx, jseg)
        source_key, key = jax.random.split(source_key)
        jx = sampler(out, key=key)
        all_tokens.append(jx)

    full_tokens = torch.tensor(jnp.concat(all_tokens, axis=1))
    out_tokens = tokenizer.decode(full_tokens[0], skip_special_tokens=True)
    print(out_tokens)


if __name__ == "__main__":
    run_model()
