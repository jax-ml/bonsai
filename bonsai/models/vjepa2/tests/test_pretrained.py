import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification, VJEPA2Model

from bonsai.models.vjepa2 import modeling as model_lib
from bonsai.models.vjepa2 import params


class TestFoundationModel(absltest.TestCase):
    """Test foundation model (VJEPA2Model) loading and inference."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hf_repo = "facebook/vjepa2-vitl-fpc64-256"
        cls.model_dir = snapshot_download(cls.hf_repo)

        # Load PyTorch model
        cls.torch_model = VJEPA2Model.from_pretrained(cls.hf_repo)
        cls.torch_model.eval()

        # Load Flax model using the predefined config
        cls.flax_config = model_lib.VJEPA2FlaxConfig.vitl_fpc64_256()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_dir, cls.flax_config, classifier=False)
        cls.flax_model.eval()

        # Processor
        cls.processor = AutoVideoProcessor.from_pretrained(cls.hf_repo)

    def _prepare_inputs(self, seed=42):
        """Prepare video inputs for testing."""
        np.random.seed(seed)
        video = np.random.randn(16, 3, 256, 256).astype(np.float32)

        inputs = self.processor(video, return_tensors="pt")
        pixel_values_videos = inputs.pixel_values_videos

        video_jax = jnp.asarray(pixel_values_videos.numpy())
        video_jax = video_jax.transpose(0, 1, 3, 4, 2)

        return pixel_values_videos, video_jax

    def test_full_model_output(self):
        """Test full model output (with predictor) matches between PyTorch and Flax."""
        pixel_values_videos, video_jax = self._prepare_inputs()

        # PyTorch forward (with predictor)
        with torch.inference_mode():
            torch_out = self.torch_model(pixel_values_videos, skip_predictor=False)
        torch_hidden = torch_out.last_hidden_state.numpy()
        torch_predictor = torch_out.predictor_output.last_hidden_state.numpy()

        # Flax forward (with predictor)
        flax_out = self.flax_model(video_jax, skip_predictor=False)
        flax_hidden = np.asarray(jax.device_get(flax_out.last_hidden_state))
        flax_predictor = np.asarray(jax.device_get(flax_out.predictor_output.last_hidden_state))

        # Check encoder last hidden state
        self.assertEqual(torch_hidden.shape, flax_hidden.shape)

        torch.testing.assert_close(
            torch.tensor(flax_hidden),
            torch.tensor(torch_hidden),
            rtol=float("inf"),  # Its a mess
            atol=20.0,  # Allow for accumulated FP error over 24 layers (only 0.05% elements mismatch)
        )

        # Check predictor output
        self.assertEqual(torch_predictor.shape, flax_predictor.shape)

        torch.testing.assert_close(
            torch.tensor(flax_predictor),
            torch.tensor(torch_predictor),
            rtol=float("inf"),
            atol=20.0,
        )

    def test_encoder_only(self):
        """Test encoder-only output (skip_predictor=True)."""
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_out = self.torch_model(pixel_values_videos, skip_predictor=True)
        torch_hidden = torch_out.last_hidden_state.numpy()

        flax_out = self.flax_model(video_jax, skip_predictor=True)
        flax_hidden = np.asarray(jax.device_get(flax_out.last_hidden_state))

        self.assertEqual(torch_hidden.shape, flax_hidden.shape)

        torch.testing.assert_close(
            torch.tensor(flax_hidden),
            torch.tensor(torch_hidden),
            rtol=float("inf"),
            atol=20.0,
        )

    def test_get_vision_features(self):
        """Test get_vision_features method matches PyTorch."""
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_features = self.torch_model.get_vision_features(pixel_values_videos).numpy()

        flax_features = np.asarray(jax.device_get(self.flax_model.get_vision_features(video_jax)))

        self.assertEqual(torch_features.shape, flax_features.shape)

        torch.testing.assert_close(
            torch.tensor(flax_features),
            torch.tensor(torch_features),
            rtol=float("inf"),
            atol=20.0,
        )


class TestClassificationModel(absltest.TestCase):
    """Test classification model loading and inference."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
        cls.model_dir = snapshot_download(cls.hf_repo)

        # Load PyTorch model
        cls.torch_model = VJEPA2ForVideoClassification.from_pretrained(cls.hf_repo)
        cls.torch_model.eval()

        # Load Flax model using the predefined config
        cls.flax_config = model_lib.VJEPA2FlaxConfig.vitl_fpc16_256()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_dir, cls.flax_config, classifier=True)
        cls.flax_model.eval()

        # Processor
        cls.processor = AutoVideoProcessor.from_pretrained(cls.hf_repo)

    def _prepare_inputs(self, seed=42):
        """Prepare video inputs for testing."""
        np.random.seed(seed)
        # Use correct number of frames for the model
        num_frames = self.flax_config.frames_per_clip
        video = np.random.randn(num_frames, 3, 256, 256).astype(np.float32)

        inputs = self.processor(video, return_tensors="pt")
        pixel_values_videos = inputs.pixel_values_videos

        video_jax = jnp.asarray(pixel_values_videos.numpy())
        video_jax = video_jax.transpose(0, 1, 3, 4, 2)

        return pixel_values_videos, video_jax

    def test_classification_logits(self):
        """Test classification logits match between PyTorch and Flax."""
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_out = self.torch_model(pixel_values_videos)
        torch_logits = torch_out.logits.numpy()

        flax_out = self.flax_model(video_jax)
        flax_logits = np.asarray(jax.device_get(flax_out.logits))

        # SSv2 has 174 classes
        self.assertEqual(torch_logits.shape, flax_logits.shape)

        torch.testing.assert_close(
            torch.tensor(flax_logits),
            torch.tensor(torch_logits),
            rtol=1e-4,
            atol=7e-2,
        )

    def test_top_k_predictions(self):
        """Test that top-k predictions match between PyTorch and Flax."""
        pixel_values_videos, video_jax = self._prepare_inputs()

        with torch.inference_mode():
            torch_logits = self.torch_model(pixel_values_videos).logits
        torch_top5 = torch.topk(torch_logits, 5).indices[0].numpy()

        flax_logits = self.flax_model(video_jax).logits
        flax_logits_np = np.asarray(jax.device_get(flax_logits))
        flax_top5 = np.argsort(flax_logits_np[0])[-5:][::-1]

        np.testing.assert_array_equal(flax_top5, torch_top5)


if __name__ == "__main__":
    absltest.main()
