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

"""Test accuracy of jax impelement vs torch transformer impelement."""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers import UMT5EncoderModel as TorchUMT5EncoderModel
from transformers import UMT5Model as TorchUMT5Model

from bonsai.models.umt5.modeling import UMT5EncoderModel, UMT5Model
from bonsai.models.umt5.params import create_model, load_model_config


def _to_np(arr):
    """
    If dtype is bf16, we need to convert it to float32, because bf16 is not supported in numpy.
    """
    if isinstance(arr, torch.Tensor):
        if arr.dtype == torch.bfloat16:
            arr = arr.float()
        arr = arr.detach().cpu().numpy()
    elif isinstance(arr, jax.Array):
        if arr.dtype == jnp.bfloat16:
            arr = arr.astype(jnp.float32)
        arr = np.array(arr)

    return arr


def compare_outputs(jax_output: jax.Array, torch_output, name: str, rtol: float = 1e-3, atol: float = 1e-5):
    """Compare JAX and PyTorch outputs and report differences.
    Args:
        jax_output: Output from JAX model
        torch_output: Output from PyTorch model (torch.Tensor)
        name: Name of the output being compared
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    jax_np = _to_np(jax_output)
    torch_np = _to_np(torch_output)

    # Check shapes match
    try:
        np.testing.assert_allclose(jax_np, torch_np, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(f"\nFAIL: {name} comparison failed")
        print(e)
        return False


class UMT5Test(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.model_name = "google/umt5-base"
        self.tokenizer_name = "google/umt5-base"

        self.model_ckpt_path = snapshot_download(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model_config = load_model_config(self.model_ckpt_path)

    def test_t5_encoder_accuracy(self):
        jax_t5 = create_model(
            UMT5EncoderModel,
            file_dir=self.model_ckpt_path,
            cfg=self.model_config,
        )

        hf_t5 = TorchUMT5EncoderModel.from_pretrained(self.model_ckpt_path)

        prompts = [
            "A beautiful sunset over the ocean with waves crashing on the shore",
            "translate to French: I love cat",
        ]
        torch_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        jax_inputs = self.tokenizer(prompts, padding=True, return_tensors="np")

        # test encoder accuracy
        pytorch_output = hf_t5.encoder(input_ids=torch_inputs.input_ids, attention_mask=torch_inputs.attention_mask)
        jax_output = jax_t5.encoder(
            input_ids=jnp.array(jax_inputs.input_ids), attention_mask=jnp.array(jax_inputs.attention_mask)
        )

        torch_embeddings = pytorch_output.last_hidden_state

        seq_lens = torch_inputs.attention_mask.gt(0).sum(dim=1).long()
        for i in range(jax_output.shape[0]):
            self.assertTrue(
                compare_outputs(
                    jax_output[i, : seq_lens[i], :],
                    torch_embeddings[i, : seq_lens[i], :],
                    f"UMT5 Encoder For Prompt: {prompts[i]}",
                )
            )

    def test_t5_decoder_accuracy(self):
        jax_t5 = create_model(
            UMT5Model,
            file_dir=self.model_ckpt_path,
            cfg=self.model_config,
        )

        hf_t5 = TorchUMT5Model.from_pretrained(self.model_ckpt_path)

        prompts = [
            "A beautiful sunset over the ocean with waves crashing on the shore",
            "translate to French: I love cat",
        ]
        torch_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")

        # test encoder accuracy
        pytorch_output = hf_t5.encoder(input_ids=torch_inputs.input_ids, attention_mask=torch_inputs.attention_mask)
        encoder_hidden_states = pytorch_output.last_hidden_state

        # test decoder accuracy
        # use torch encoder output as docoder input
        bs = encoder_hidden_states.shape[0]
        decoder_input_ids = [0, 1, 2, 3]
        torch_input_ids = torch.tensor(decoder_input_ids, dtype=torch.int32).unsqueeze(0).repeat(bs, 1)
        pytorch_decoder_output = hf_t5.decoder(
            input_ids=torch_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=torch_inputs.attention_mask,
        )
        decoder_hidden_states = pytorch_decoder_output.last_hidden_state

        jax_decoder_output = jax_t5.decoder(
            input_ids=jnp.array(torch_input_ids.numpy()),
            encoder_hidden_states=jnp.array(encoder_hidden_states.detach().numpy()),
            encoder_attention_mask=jnp.array(torch_inputs.attention_mask.detach().numpy()),
        )
        self.assertTrue(compare_outputs(jax_decoder_output, decoder_hidden_states, "UMT5 Decoder"))


if __name__ == "__main__":
    absltest.main()
