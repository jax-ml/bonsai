import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from jax import P, make_mesh, set_mesh
from jax.sharding import AxisType
from jax.typing import DTypeLike
from tqdm import trange
from transformers import AutoProcessor

from bonsai.models.gemma3 import modeling, params

# artificial cpu devices
jax.config.update("jax_num_cpu_devices", 2)


class TestSharding(absltest.TestCase):
    # using this for faster testing. This way we can avoid reloading the model.
    # Make sure not to modify the Gemma3 model in inconsistent ways between tests.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_name: str = "google/gemma-3-4b-it"
        # self.model_name: str = "google/gemma-3-270m" # This is text only
        access_token = os.environ["HF_TOKEN"]
        cls.processor = AutoProcessor.from_pretrained(cls.model_name, token=access_token, use_fast=False)
        cls.torch_device = "cpu"

        fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value

        cls.mesh = jax.make_mesh(((2, 1)), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        cls.bonsai_config = modeling.ModelConfig.gemma3_4b()
        model_ckpt_path = snapshot_download(cls.model_name)
        cls.bonsai_model = params.create_gemma3_from_pretrained(model_ckpt_path, cls.bonsai_config)

        cls.batch_size = 1
        cls.cache_size, cls.gen_steps = 512, 10

    def _make_torch_input(self):
        # returns model inputs:
        # KEY               SHAPE                           DTYPE
        # input_ids         torch.Size([1, 281])            int64
        # attention_mask    torch.Size([1, 281])            int64
        # token_type_ids    torch.Size([1, 281])            int64
        # pixel_values      torch.Size([1, 3, 896, 896])    bfloat16 -> float32
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    },
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ]

        out = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        out["pixel_values"] = out["pixel_values"].to(dtype=torch.float32)

        return {k: v.to(device=self.torch_device) for k, v in out.items()}

    def test_full(self):
        nm = self.bonsai_model

        t_inputs = self._make_torch_input()

        n_img = jnp.array(np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)))
        n_text = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())

        # Test simple batching
        n_img = jnp.concat([n_img, n_img])
        n_text = jnp.concat([n_text, n_text])
        n_tti = jnp.concat([n_tti, n_tti])

        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        nm(n_text, n_img, cache, segment_ids, n_tti)


if __name__ == "__main__":
    absltest.main()
