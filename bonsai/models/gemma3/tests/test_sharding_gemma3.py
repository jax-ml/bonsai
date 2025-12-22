import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from jax import P
from jax.sharding import AxisType
from transformers import AutoProcessor

from bonsai.models.gemma3 import modeling
from bonsai.models.gemma3.tests.test_outputs_gemma3 import check_hf_token

SKIP_ALL: bool = False


class TestSharding(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if SKIP_ALL:
            return
        cls.model_name: str = "google/gemma-3-4b-it"
        access_token = os.environ["HF_TOKEN"]
        cls.processor = AutoProcessor.from_pretrained(cls.model_name, token=access_token, use_fast=False)
        cls.torch_device = "cpu"

        fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value

        cls.mesh = jax.make_mesh(((1, 1)), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        cls.bonsai_config = modeling.ModelConfig.gemma3_4b_it(True, True)
        cls.bonsai_model = modeling.Gemma3Model(cls.bonsai_config, rngs=nnx.Rngs(0))

    def _make_torch_input(self):
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

    @unittest.skipIf(SKIP_ALL, "Done")
    def test_full(self):
        nm = self.bonsai_model
        fsdp = modeling.ShardMode.FSDP.value

        t_inputs = self._make_torch_input()

        n_img = jnp.array(
            np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)), out_sharding=P(fsdp)
        )
        n_text = jnp.array(t_inputs["input_ids"].detach().cpu().numpy(), out_sharding=P(fsdp))
        n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy(), out_sharding=P(fsdp))

        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens), out_sharding=P(fsdp))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        nm(n_text, n_img, cache, segment_ids, n_tti)

    @unittest.skip("Only for viewing purposes")
    def test_view_model(self):
        state = nnx.state(self.bonsai_model)
        out = jax.tree_util.tree_map(lambda x: jax.typeof(x), state)

        # print(out)
        # print(out.vision_tower)
        # print(out.language_model)
        # print(out.embed_tokens)
        print(out.multi_modal_projector)


if __name__ == "__main__":
    err = check_hf_token()
    if err:
        SKIP_ALL = True
        print("Failed to access HF_TOKEN or download Processor:")
        print(err)
    absltest.main()
