import os
import json
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from huggingface_hub import snapshot_download

from bonsai.models.mimo_audio.mimo_audio_tokenizer_params import load_tokenizer_weights_from_safetensors
from bonsai.models.mimo_audio.mimo_audio_tokenizer_configuration import MiMoAudioTokenizerConfig as JaxTokenizerConfig

# Since the transformer library does not yet support mimo-audio-tokenizer,
# during testing, the official implementation code of mimo-audio-tokenizer (https://github.com/XiaomiMiMo/MiMo-Audio) needs to be copied to the corresponding location.
from bonsai.models.mimo_audio.pytorch.src.mimo_audio_tokenizer.modeling_audio_tokenizer import (
    MiMoAudioTokenizer as TorchTokenizer,
)
from bonsai.models.mimo_audio.pytorch.src.mimo_audio_tokenizer.configuration_audio_tokenizer import (
    MiMoAudioTokenizerConfig as TorchTokenizerConfig,
)


class TestMiMoAudioTokenizerOutputs(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        jax.config.update("jax_default_matmul_precision", "float32")
        jax.config.update("jax_platforms", "cpu")

        model_name = "XiaomiMiMo/MiMo-Audio-Tokenizer"
        model_ckpt_path = snapshot_download(model_name)

        safetensors_path = os.path.join(model_ckpt_path, "model.safetensors")
        config_path = os.path.join(model_ckpt_path, "config.json")

        torch_config = TorchTokenizerConfig.from_pretrained(model_ckpt_path)
        cls.torch_model = TorchTokenizer.from_pretrained(
            model_ckpt_path, config=torch_config, torch_dtype=torch.float32
        )

        cls.torch_model = cls.torch_model.cpu()
        cls.torch_model.eval()

        def move_to_cpu(module):
            for child in module.children():
                move_to_cpu(child)
            for name, param in module._parameters.items():
                if param is not None:
                    module._parameters[name] = param.cpu()
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.cpu()

        move_to_cpu(cls.torch_model)

        with open(config_path) as f:
            config_dict = json.load(f)
        jax_config = JaxTokenizerConfig(**config_dict, use_sharding=False)
        cls.nnx_model = load_tokenizer_weights_from_safetensors(
            config=jax_config, safetensors_path=safetensors_path, dtype=jnp.float32, mesh=None, rngs=nnx.Rngs(params=0)
        )

        cls.batch_size = 1
        cls.n_mels = jax_config.n_mels
        cls.mel_frames = 50
        cls.d_model = jax_config.d_model
        cls.tol = 1e-3

    def _compare(self, jy, ty):
        if ty.dim() == 2 and jy.ndim == 3:
            ty = ty.unsqueeze(0)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.tol,
            atol=self.tol,
            check_dtype=False,
        )

    def test_encoder_conv1(self):
        mels = torch.randn(self.batch_size, self.n_mels, self.mel_frames, dtype=torch.float32)
        jx = jnp.array(mels.permute(0, 2, 1).numpy())

        jy = jax.nn.gelu(self.nnx_model.encoder.conv1(jx))
        ty = torch.nn.functional.gelu(self.torch_model.encoder.conv1(mels))

        self._compare(jy, ty.permute(0, 2, 1))

    def test_encoder_conv2(self):
        x = torch.randn(self.batch_size, self.d_model, self.mel_frames, dtype=torch.float32)
        jx = jnp.array(x.permute(0, 2, 1).numpy())

        jy = jax.nn.gelu(self.nnx_model.encoder.conv2(jx))
        ty = torch.nn.functional.gelu(self.torch_model.encoder.conv2(x))

        self._compare(jy, ty.permute(0, 2, 1))

    def test_quantizer_encode(self):
        seq_len = 12
        x = torch.randn(seq_len, self.d_model, dtype=torch.float32)
        jx = jnp.array(x.numpy())

        jcodes, jquantized = self.nnx_model.encoder.quantizer.encode(jx, mask=None, n_q=None)

        tcodes = self.torch_model.encoder.quantizer.encode(x.unsqueeze(0))

        np.testing.assert_array_equal(np.array(jcodes), tcodes.squeeze(1).numpy())

    def test_quantizer_decode(self):
        num_q = self.nnx_model.encoder.quantizer.n_q
        seq_len = 12

        codes_list = []
        for i in range(num_q):
            codebook_size = self.nnx_model.encoder.quantizer.codebooks[i].value.shape[0]
            codes_list.append(torch.randint(0, codebook_size, (seq_len,), dtype=torch.long))
        codes = torch.stack(codes_list, dim=0)
        jcodes = jnp.array(codes.numpy())

        jdecoded = self.nnx_model.encoder.decode_vq(jcodes)
        tdecoded = self.torch_model.encoder.decode_vq(codes)

        self._compare(jdecoded, tdecoded)

    def test_decoder_dconv1(self):
        seq_len = 12
        x = torch.randn(self.batch_size, seq_len, self.d_model, dtype=torch.float32)
        jx = jnp.array(x.numpy())

        input_length = torch.tensor([seq_len], dtype=torch.long)
        jinput_length = jnp.array([seq_len], dtype=jnp.int32)

        if self.nnx_model.decoder.dconv1 is not None:
            jy, jout_len = self.nnx_model.decoder.dconv1(jx, jinput_length)
            ty, tout_len = self.torch_model.decoder.dconv1(x, input_length, output_dim=3)

            self._compare(jy, ty)
            np.testing.assert_array_equal(np.array(jout_len), tout_len.numpy())

    def test_decoder_dconv2(self):
        seq_len = 24
        x = torch.randn(self.batch_size, seq_len, self.d_model, dtype=torch.float32)
        jx = jnp.array(x.numpy())

        input_length = torch.tensor([seq_len], dtype=torch.long)
        jinput_length = jnp.array([seq_len], dtype=jnp.int32)

        jy, jout_len = self.nnx_model.decoder.dconv2(jx, jinput_length)

        tx = torch.masked_select(x, torch.ones_like(x, dtype=bool)).view(-1, self.d_model)
        ty, tout_len = self.torch_model.decoder.dconv2(tx, input_length, output_dim=3)

        self._compare(jy, ty)
        np.testing.assert_array_equal(np.array(jout_len), tout_len.numpy())

    def test_vocoder_embeddings(self):
        seq_len = 48
        x = torch.randn(self.batch_size, seq_len, self.n_mels, dtype=torch.float32)
        jx = jnp.array(x.numpy())

        jy = self.nnx_model.decoder.vocoder.embeddings(jx)
        ty = self.torch_model.decoder.vocoder.embeddings(x)

        self._compare(jy, ty)

    def test_vocoder_istft_head(self):
        vocoder_dim = self.nnx_model.decoder.vocoder.config.vocoder_dim
        seq_len = 48
        x = torch.randn(self.batch_size, seq_len, vocoder_dim, dtype=torch.float32)
        jx = jnp.array(x.numpy())

        jy = self.nnx_model.decoder.vocoder.head(jx)
        ty = self.torch_model.decoder.vocoder.head(x)

        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=1e-2,
            atol=1e-2,
            check_dtype=False,
        )


if __name__ == "__main__":
    absltest.main()
