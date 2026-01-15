import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import constants, snapshot_download
from safetensors.torch import save_file
from transformers import (
    AutoVideoProcessor,
    VJEPA2Config,
    VJEPA2ForVideoClassification,
    VJEPA2Model,
)
from transformers import VJEPA2ForVideoClassification as VJEPA2ForVideoClassificationTorch

from bonsai.models.vjepa2 import modeling as model_lib
from bonsai.models.vjepa2 import params


class TestForwardPass(absltest.TestCase):
    """Test forward pass using small manually-initialized models."""

    def setUp(self):
        super().setUp()
        self.save_dir = constants.default_cache_path
        os.makedirs(self.save_dir, exist_ok=True)

        self.hfconfig = VJEPA2Config(
            crop_size=64,
            frames_per_clip=4,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            mlp_ratio=4.0,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            patch_size=16,
            tubelet_size=2,
            in_chans=3,
            num_pooler_layers=1,
        )
        self.baseline_model = VJEPA2Model(config=self.hfconfig)
        self.model_ckpt_path = os.path.join(self.save_dir, "model.safetensors")
        save_file(self.baseline_model.state_dict(), self.model_ckpt_path)

        self.config = model_lib.VJEPA2FlaxConfig.standard_test()
        self.bonsai_model = params.create_model_from_safe_tensors(self.save_dir, self.config, classifier=False)

        os.remove(self.model_ckpt_path)
        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 1
        self.num_frames = self.config.frames_per_clip
        self.height = self.config.crop_size
        self.width = self.config.crop_size
        self.channels = self.config.in_chans

    def test_input_embeddings(self):
        torch_emb = self.baseline_model.encoder.embeddings
        nnx_emb = self.bonsai_model.encoder.embeddings

        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        with torch.inference_mode():
            ty = torch_emb(tx)
        jy = nnx_emb(jx_flax)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=2e-3)

    def test_first_layer(self):
        torch_emb = self.baseline_model.encoder.embeddings
        nnx_emb = self.bonsai_model.encoder.embeddings
        torch_layer = self.baseline_model.encoder.layer[0]
        nnx_layer = self.bonsai_model.encoder.layer[0]

        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)
        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        jhs = nnx_emb(jx_flax)

        with torch.inference_mode():
            ths = torch_emb(tx)
            ty = torch_layer(ths, position_mask=None, output_attentions=False)[0]

        jy = nnx_layer(jhs, None)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=1e-3)

    def test_rope_attention(self):
        """Test RoPE attention output matches between PyTorch and Flax."""
        torch_emb = self.baseline_model.encoder.embeddings
        nnx_emb = self.bonsai_model.encoder.embeddings
        torch_attn = self.baseline_model.encoder.layer[0].attention
        nnx_attn = self.bonsai_model.encoder.layer[0].attention

        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)
        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        with torch.inference_mode():
            ths = torch_emb(tx)
            ths_normed = self.baseline_model.encoder.layer[0].norm1(ths)
            ty = torch_attn(ths_normed, position_mask=None, output_attentions=False)[0]

        jhs = nnx_emb(jx_flax)
        jhs_normed = self.bonsai_model.encoder.layer[0].norm1(jhs)
        jy = nnx_attn(jhs_normed, None)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=2e-5)

    def test_last_hidden_state(self):
        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        with torch.inference_mode():
            ty = self.baseline_model(tx, skip_predictor=True).last_hidden_state
        jy = self.bonsai_model(jx_flax, skip_predictor=True)["last_hidden_state"]

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=2e-3)


class TestVideoClassification(absltest.TestCase):
    """Test video classification model with small manually-initialized models."""

    def setUp(self):
        super().setUp()
        self.save_dir = constants.default_cache_path
        os.makedirs(self.save_dir, exist_ok=True)

        self.hfconfig = VJEPA2Config(
            crop_size=64,
            frames_per_clip=4,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            mlp_ratio=4.0,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            patch_size=16,
            tubelet_size=2,
            in_chans=3,
            num_pooler_layers=1,
            num_labels=10,
        )
        self.baseline_model = VJEPA2ForVideoClassificationTorch(config=self.hfconfig)
        self.model_ckpt_path = os.path.join(self.save_dir, "classifier_model.safetensors")
        save_file(self.baseline_model.state_dict(), self.model_ckpt_path)

        self.config = model_lib.VJEPA2FlaxConfig(
            crop_size=64,
            frames_per_clip=4,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            mlp_ratio=4.0,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            num_pooler_layers=1,
            num_labels=10,
        )
        self.bonsai_model = params.create_model_from_safe_tensors(self.save_dir, self.config, classifier=True)

        os.remove(self.model_ckpt_path)
        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 1
        self.num_frames = self.config.frames_per_clip
        self.height = self.config.crop_size
        self.width = self.config.crop_size
        self.channels = self.config.in_chans

    def test_classification_logits(self):
        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        with torch.inference_mode():
            ty = self.baseline_model(tx).logits
        jy = self.bonsai_model(jx_flax)["logits"]

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        self.assertEqual(ty_bonsai.shape, ty.shape)
        self.assertEqual(ty_bonsai.shape[-1], self.config.num_labels)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=1e-2)


class TestPretrainedFoundationModel(absltest.TestCase):
    """Test pretrained foundation model (VJEPA2Model)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hf_repo = "facebook/vjepa2-vitl-fpc64-256"
        cls.model_dir = snapshot_download(cls.hf_repo)

        cls.torch_model = VJEPA2Model.from_pretrained(cls.hf_repo)
        cls.torch_model.eval()

        cls.flax_config = model_lib.VJEPA2FlaxConfig.vitl_fpc64_256()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_dir, cls.flax_config, classifier=False)
        cls.flax_model.eval()

    def _prepare_inputs(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        video = np.random.randn(1, 16, 3, 256, 256).astype(np.float32)

        pixel_values_videos = torch.tensor(video)
        video_jax = jnp.asarray(video.transpose(0, 1, 3, 4, 2))

        return pixel_values_videos, video_jax

    def test_rope_attention_pretrained(self):
        """Test RoPE attention with pretrained weights."""
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_emb = self.torch_model.encoder.embeddings(pixel_values_videos)
            torch_normed = self.torch_model.encoder.layer[0].norm1(torch_emb)
            torch_attn = self.torch_model.encoder.layer[0].attention(torch_normed, None)[0]

        flax_emb = self.flax_model.encoder.embeddings(video_jax)
        flax_normed = self.flax_model.encoder.layer[0].norm1(flax_emb)
        flax_attn = self.flax_model.encoder.layer[0].attention(flax_normed, None)

        torch.testing.assert_close(
            torch.tensor(np.asarray(jax.device_get(flax_attn))),
            torch.tensor(torch_attn.numpy()),
            rtol=1e-5,
            atol=2e-2,
        )

    def test_encoder_only(self):
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_out = self.torch_model(pixel_values_videos, skip_predictor=True)
        torch_hidden = torch_out.last_hidden_state.numpy()

        flax_out = self.flax_model(video_jax, skip_predictor=True)
        flax_hidden = np.asarray(jax.device_get(flax_out["last_hidden_state"]))

        self.assertEqual(torch_hidden.shape, flax_hidden.shape)

        torch.testing.assert_close(
            torch.tensor(flax_hidden),
            torch.tensor(torch_hidden),
            rtol=1e-5,
            atol=6e-1,
        )

    def test_full_model_output(self):
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_out = self.torch_model(pixel_values_videos, skip_predictor=False)
        torch_hidden = torch_out.last_hidden_state.numpy()
        torch_predictor = torch_out.predictor_output.last_hidden_state.numpy()

        flax_out = self.flax_model(video_jax, skip_predictor=False)
        flax_hidden = np.asarray(jax.device_get(flax_out["last_hidden_state"]))
        flax_predictor = np.asarray(jax.device_get(flax_out["predictor_last_hidden_state"]))

        self.assertEqual(torch_hidden.shape, flax_hidden.shape)

        torch.testing.assert_close(
            torch.tensor(flax_hidden),
            torch.tensor(torch_hidden),
            rtol=1e-5,
            atol=6e-1,
        )

        self.assertEqual(torch_predictor.shape, flax_predictor.shape)

        torch.testing.assert_close(
            torch.tensor(flax_predictor),
            torch.tensor(torch_predictor),
            rtol=1e-1,
            atol=5e-1,
        )


class TestPretrainedClassificationModel(absltest.TestCase):
    """Test pretrained classification model."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
        cls.model_dir = snapshot_download(cls.hf_repo)

        cls.torch_model = VJEPA2ForVideoClassification.from_pretrained(cls.hf_repo)
        cls.torch_model.eval()

        cls.flax_config = model_lib.VJEPA2FlaxConfig.vitl_fpc16_256()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_dir, cls.flax_config, classifier=True)
        cls.flax_model.eval()

        cls.processor = AutoVideoProcessor.from_pretrained(cls.hf_repo)

    def _prepare_inputs(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        num_frames = self.flax_config.frames_per_clip
        video = np.random.randn(num_frames, 3, 256, 256).astype(np.float32)

        inputs = self.processor(video, return_tensors="pt")
        pixel_values_videos = inputs.pixel_values_videos

        video_jax = jnp.asarray(pixel_values_videos.numpy())
        video_jax = video_jax.transpose(0, 1, 3, 4, 2)

        return pixel_values_videos, video_jax

    def test_classification_logits(self):
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_out = self.torch_model(pixel_values_videos)
        torch_logits = torch_out.logits.numpy()

        flax_out = self.flax_model(video_jax)
        flax_logits = np.asarray(jax.device_get(flax_out["logits"]))

        self.assertEqual(torch_logits.shape, flax_logits.shape)

        torch.testing.assert_close(
            torch.tensor(flax_logits),
            torch.tensor(torch_logits),
            rtol=1e-5,
            atol=6e-2,
        )

    def test_top_k_predictions(self):
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_logits = self.torch_model(pixel_values_videos).logits
        torch_top5 = torch.topk(torch_logits, 5).indices[0].numpy()

        flax_logits = self.flax_model(video_jax)["logits"]
        flax_logits_np = np.asarray(jax.device_get(flax_logits))
        flax_top5 = np.argsort(flax_logits_np[0])[-5:][::-1]

        np.testing.assert_array_equal(flax_top5, torch_top5)


if __name__ == "__main__":
    absltest.main()
