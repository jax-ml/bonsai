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

"""Demo script to run UMT5 model inference."""

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.models.umt5.modeling import UMT5Model, forward
from bonsai.models.umt5.params import create_model, load_model_config


def main():
    """Run UMT5 model inference demo."""
    print("=" * 80)
    print("UMT5 Model Demo - JAX Implementation")
    print("=" * 80)

    # Model configuration
    model_name = "google/umt5-base"
    model_ckpt_path = snapshot_download(model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model config and create model
    model_conf = load_model_config(model_ckpt_path)

    jax_model = create_model(
        UMT5Model,
        file_dir=model_ckpt_path,
        cfg=model_conf,
    )
    graphdef, state = nnx.split(jax_model)

    # Prepare input
    prompts = [
        "A beautiful sunset over the ocean with waves crashing on the shore",
        "translate to French: I love cat",
    ]

    # Tokenize input
    inputs = tokenizer(prompts, padding=True, return_tensors="np")
    input_ids = jnp.array(inputs.input_ids)
    attention_mask = jnp.array(inputs.attention_mask)

    # forward
    bs = len(prompts)
    decoder_input_ids = jnp.full((bs, 1), model_conf.decoder_start_token_id, dtype=jnp.int32)
    decoder_output = forward(
        graphdef,
        state,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    print(f"Decoder output shape: {decoder_output.shape}")


if __name__ == "__main__":
    main()
