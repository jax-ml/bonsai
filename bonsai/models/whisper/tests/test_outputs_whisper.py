# Testing

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from jax.typing import DTypeLike
from transformers import AutoModel, AutoProcessor, AutoTokenizer, WhisperModel

from bonsai.models.whisper import modeling, params

# Test in float64. This should reveal implementation errors.
HIGH_PRECISION: bool = True
if HIGH_PRECISION:
    jax.config.update("jax_enable_x64", True)

# Runs a subset of the tests. Can change these with @unittest.skipIf(...)
FAST_TEST: bool = False

# Lets you see the reference model running.
# Helpful if you put print statements, etc. in the reference implementation.
OBSERVE_REFERENCE: bool = False


# NOTE: Tolerance values set based on float32 values.
# In float64 the rtol and atol can be smaller.
class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.jdtype = jnp.float64 if HIGH_PRECISION else jnp.float32
        self.tdtype = torch.float64 if HIGH_PRECISION else torch.float32

        torch.manual_seed(0)
        self.model_name: str = "openai/whisper-tiny"
        self.tdevice = "cpu"

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.torch_model = AutoModel.from_pretrained(self.model_name).to(device=self.tdevice, dtype=self.tdtype)
        self.torch_config = self.torch_model.config
        self.torch_model.eval()

        self.bonsai_config = modeling.ModelCfg.whisper_tiny()
        model_ckpt_path = snapshot_download(self.model_name)
        self.bonsai_model = params.create_whisper_from_pretrained(
            model_ckpt_path, self.bonsai_config, dtype=self.jdtype
        )

        self.batch_size = 2
        self.num_input_tokens = 281
        self.cache_size, self.gen_steps = 512, 10

    @unittest.skipIf(not OBSERVE_REFERENCE, "Not observing reference right now")
    def test_observe_reference(self):
        # NOTE: This is used to see a forward pass through the reference model
        # TODO: Delete when done testing
        t_inputs = dict(
            input_features=torch.randn((1, 80, 3000), dtype=self.tdtype, device=self.tdevice),
            decoder_input_ids=torch.ones((1, 281), dtype=torch.int64, device=self.tdevice)
            * self.torch_config.decoder_start_token_id,
        )
        with torch.no_grad():
            self.torch_model(**t_inputs)

    @unittest.skipIf(FAST_TEST, "Done")
    def test_conv1(self):
        tm = self.torch_model.encoder.conv1
        nm = self.bonsai_model.encoder.conv1

        tx = torch.randn((1, 80, 3000), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().swapaxes(1, 2).astype(self.jdtype)

        ty = tm(tx).swapaxes(1, 2)
        ny = nm(nx)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    @unittest.skipIf(FAST_TEST, "Done")
    def test_conv2(self):
        tm = self.torch_model.encoder.conv2
        nm = self.bonsai_model.encoder.conv2

        tx = torch.randn((1, 384, 3000), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().swapaxes(1, 2).astype(self.jdtype)

        ty = tm(tx).swapaxes(1, 2)
        ny = nm(nx)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    @unittest.skipIf(FAST_TEST, "Done")
    def test_embed_positions(self):
        tm = self.torch_model.encoder.embed_positions
        nm = self.bonsai_model.encoder.embed_positions

        tx = torch.arange(1500, device=self.tdevice)
        nx = tx.detach().cpu().numpy()

        ty = tm(tx)
        ny = nm(nx)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    @unittest.skipIf(FAST_TEST, "Done")
    def test_attn_projs(self):
        tm = self.torch_model.encoder.layers[0].self_attn
        nm = self.bonsai_model.encoder.layers[0].self_attn

        b, s, d = 1, 281, 384

        tx = torch.randn((b, s, d), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().astype(self.jdtype)

        ty = tm.q_proj(tx)
        ny = nm.query(nx).reshape(b, s, -1)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5, err_msg="q")

        ty = tm.k_proj(tx)
        ny = nm.key(nx).reshape(b, s, -1)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5, err_msg="k")

        ty = tm.v_proj(tx)
        ny = nm.value(nx).reshape(b, s, -1)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5, err_msg="v")

    @unittest.skipIf(FAST_TEST, "Done")
    def test_attn_out_proj(self):
        tm = self.torch_model.encoder.layers[0].self_attn.out_proj
        nm = self.bonsai_model.encoder.layers[0].self_attn.out

        b, s, d = 1, 281, 384
        nh, hd = 6, 64

        tx = torch.randn((b, s, d), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().reshape(b, s, nh, hd).astype(self.jdtype)

        ty = tm(tx)
        ny = nm(nx)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    @unittest.skipIf(FAST_TEST, "Done")
    def test_encoder_attention(self):
        tm = self.torch_model.encoder.layers[0].self_attn
        nm = self.bonsai_model.encoder.layers[0].self_attn

        tx = torch.randn((1, 281, 384), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().astype(self.jdtype)

        ty = tm(tx)[0]
        ny = nm(nx, decode=False)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    # @unittest.skipIf(FAST_TEST, "Done. This is 1.8e-5")
    def test_encoder_layer(self):
        tm = self.torch_model.encoder.layers[0]
        nm = self.bonsai_model.encoder.layers[0]

        tx = torch.randn((1, 1500, 384), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().astype(self.jdtype)

        ty = tm(
            tx,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False,
        )[0]
        ny = nm(nx, None)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=3e-5, atol=3e-5)

    @unittest.skipIf(FAST_TEST, "Done")
    def test_encoder(self):
        tm = self.torch_model.encoder
        nm = self.bonsai_model.encoder

        tx = torch.randn((1, 80, 3000), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().swapaxes(1, 2).astype(self.jdtype)

        ty = tm(
            tx, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None
        ).last_hidden_state
        ny = nm(nx, None)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=5e-2, atol=5e-2)

    # TODO: This test uses decode=False just for checking if things are the same.
    # TODO: Test with decode=True
    @unittest.skipIf(FAST_TEST, "Done")
    def test_decoder_self_attn(self):
        tm = self.torch_model.decoder.layers[0].self_attn
        nm = self.bonsai_model.decoder.layers[0].self_attn

        b, s, d = 2, 1500, 384

        tx = torch.randn((b, s, d), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().astype(self.jdtype)

        ty = tm(tx)[0]
        n_mask = nnx.make_causal_mask(nx[:, :, 0])
        ny = nm(nx, decode=False, mask=n_mask)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    # TODO: Test with decode=True
    @unittest.skipIf(FAST_TEST, "Done")
    def test_decoder_cross_attn(self):
        tm = self.torch_model.decoder.layers[0].encoder_attn
        nm = self.bonsai_model.decoder.layers[0].encoder_attn

        b, s, d = 2, 1500, 384

        tx = torch.randn((b, s, d), device=self.tdevice, dtype=self.tdtype)
        nx = tx.detach().cpu().numpy().astype(self.jdtype)

        encoder_tx = torch.randn((b, s, d), device=self.tdevice, dtype=self.tdtype)
        encoder_nx = encoder_tx.detach().cpu().numpy().astype(self.jdtype)

        n_mask = nnx.make_attention_mask(nx[:, :, 0], encoder_nx[:, :, 0])
        ty = tm(tx, encoder_tx)[0]
        ny = nm(nx, encoder_nx, encoder_nx, mask=n_mask, decode=False)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    @unittest.skipIf(FAST_TEST, "Done")
    def test_decoder_layer(self):
        # TODO: Need to set this test up properly. Need attention masks, etc.
        for i, (tm, nm) in enumerate(zip(self.torch_model.decoder.layers, self.bonsai_model.decoder.layers)):
            tx = torch.randn((1, 1500, 384), device=self.tdevice, dtype=self.tdtype)
            nx = tx.detach().cpu().numpy().astype(self.jdtype)

            ty = tm(tx, encoder_hidden_states=tx)[0]
            # nm.init_cache((1, 1500, 384))

            self_mask = nnx.make_causal_mask(nx[:, :, 0])
            cross_mask = nnx.make_attention_mask(nx[:, :, 0], nx[:, :, 0])
            # ^ second arg of cross_mask should come from the encoder, but this is fine for now
            ny = nm(nx, self_mask, nx, cross_mask, decode=False)
            np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-2, atol=1e-2, err_msg=f"layer {i}")

    @unittest.skipIf(FAST_TEST, "Done")
    def test_decoder_embeds(self):
        tm = self.torch_model.decoder
        nm = self.bonsai_model.decoder

        t_input_ids = torch.randint(0, 100, (1, 200), device=self.tdevice)
        n_input_ids = t_input_ids.detach().cpu().numpy()

        tx = tm.embed_tokens(t_input_ids)
        nx = nm.embed_tokens(n_input_ids)

        np.testing.assert_allclose(nx, tx.detach().cpu().numpy(), err_msg="embed_tokens")

        tpos = tm.embed_positions(tx, position_ids=torch.arange(200, device=self.tdevice))
        npos = nm.embed_positions(jnp.arange(200))

        np.testing.assert_allclose(npos, tpos.detach().cpu().numpy(), err_msg="pos_tokens")

    # @unittest.skipIf(FAST_TEST, "TODO")
    def test_decoder(self):
        tm = self.torch_model.decoder
        nm = self.bonsai_model.decoder

        t_input_ids = torch.randint(0, 100, (1, 200), device=self.tdevice)
        n_input_ids = t_input_ids.detach().cpu().numpy()

        t_encoder = torch.randn((1, 200, 384), device=self.tdevice)
        n_encoder = t_encoder.detach().cpu().numpy()
        self_mask = nnx.make_causal_mask(n_input_ids)
        cross_mask = nnx.make_attention_mask(n_input_ids, n_input_ids)

        ty = tm(
            input_ids=t_input_ids,
            encoder_hidden_states=t_encoder,
            position_ids=torch.arange(200, device=self.tdevice),
        ).last_hidden_state
        ny = nm(n_input_ids, self_mask, n_encoder, cross_mask, decode=False)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-2, atol=1e-2)

    @unittest.skipIf(FAST_TEST, "TODO")
    def test_full(self):
        pass

    @unittest.skipIf(FAST_TEST, "TODO")
    def test_full_batched(self):
        pass

    @unittest.skipIf(FAST_TEST, "TODO")
    def test_full_chunked(self):
        pass


if __name__ == "__main__":
    absltest.main()
