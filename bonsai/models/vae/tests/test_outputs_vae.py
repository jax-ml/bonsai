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

import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest, parameterized
from diffusers.models import AutoencoderKL
from bonsai.models.vae import modeling


class TestModuleForwardPasses(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.torch_device = "cpu"
        cls.img_size = 256

        model_name = "stabilityai/sd-vae-ft-mse"
        cls.jax_model = modeling.VAE.from_pretrained(model_name)
        cls.dif_model = AutoencoderKL.from_pretrained(model_name)

    def test_encoder(self):
        batch = 1
        tx = torch.rand((batch, 3, self.img_size, self.img_size), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))

        tm = self.dif_model.encoder.to(self.torch_device).eval()
        jm = self.jax_model.encoder

        with torch.no_grad():
            ty = tm(tx)
        jy = jm(jx)

        np.testing.assert_allclose(jy, ty.permute(0, 2, 3, 1).cpu().detach().numpy(), atol=5e-3)

    def test_quant_conv(self):
        batch = 1
        tx = torch.rand((batch, 8, 32, 32), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))

        tm = self.dif_model.quant_conv.to(self.torch_device).eval()
        jm = self.jax_model.quant_conv

        with torch.no_grad():
            ty = tm(tx)
        jy = jm(jx)

        np.testing.assert_allclose(jy, ty.permute(0, 2, 3, 1).cpu().detach().numpy(), atol=5e-3)

    def test_post_quant_conv(self):
        batch = 1
        tx = torch.rand((batch, 8, 32, 32), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))

        t_mean, _ = torch.chunk(tx, chunks=2, dim=1)
        j_mean, _ = jnp.split(jx, 2, axis=-1)

        tm = self.dif_model.post_quant_conv.to(self.torch_device).eval()
        jm = self.jax_model.post_quant_conv

        with torch.no_grad():
            ty = tm(t_mean)
        jy = jm(j_mean)

        np.testing.assert_allclose(jy, ty.permute(0, 2, 3, 1).cpu().detach().numpy(), atol=8e-3)

    def test_decoder(self):
        batch = 1
        tx = torch.rand((batch, 4, 32, 32), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))

        tm = self.dif_model.decoder.to(self.torch_device).eval()
        jm = self.jax_model.decoder

        with torch.no_grad():
            ty = tm(tx)
        jy = jm(jx)

        np.testing.assert_allclose(jy, ty.permute(0, 2, 3, 1).cpu().detach().numpy(), atol=5e-3)

    def test_full(self):
        batch = 1
        tx = torch.rand((batch, 3, self.img_size, self.img_size), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))

        tm = self.dif_model.to(self.torch_device).eval()
        jm = self.jax_model

        with torch.no_grad():
            ty = tm(tx).sample
        jy = jm(jx)

        np.testing.assert_allclose(jy, ty.permute(0, 2, 3, 1).cpu().detach().numpy(), atol=5e-3)

    def test_full_batched(self):
        batch = 32
        tx = torch.rand((batch, 3, self.img_size, self.img_size), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))

        tm = self.dif_model.to(self.torch_device).eval()
        jm = self.jax_model

        with torch.no_grad():
            ty = tm(tx).sample
        jy = jm(jx)

        np.testing.assert_allclose(jy, ty.permute(0, 2, 3, 1).cpu().detach().numpy(), atol=5e-3)


if __name__ == "__main__":
    absltest.main()
