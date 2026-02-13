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
from transformers import VJEPA2ForVideoClassification as VJEPA2TorchClassif
from bonsai.models.vjepa2 import modeling as model_lib
from bonsai.models.vjepa2 import params


class ParityTestBase(absltest.TestCase):
    def _get_inputs(self, seed=0):
        key = jax.random.PRNGKey(seed)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)
        jx_cl = jnp.transpose(jx, (0, 1, 3, 4, 2))  # NFCHW -> NFHWC
        return tx, jx_cl

    def assert_parity(self, ty, jy, rtol=1e-5, atol=2e-3):
        jy_torch = torch.tensor(np.asarray(jax.device_get(jy)), dtype=torch.float32)
        self.assertEqual(jy_torch.shape, ty.shape)
        torch.testing.assert_close(jy_torch, ty, rtol=rtol, atol=atol)


class TestForwardPass(ParityTestBase):
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
        self.config = model_lib.ModelConfig.standard_test()
        self.bonsai_model = params.create_model_from_safe_tensors(self.save_dir, self.config, classifier=False)
        if os.path.exists(self.model_ckpt_path):
            os.remove(self.model_ckpt_path)
        self.bonsai_model.eval()
        self.baseline_model.eval()
        self.batch_size, self.num_frames = 1, self.config.frames_per_clip
        self.height, self.width, self.channels = self.config.crop_size, self.config.crop_size, self.config.in_chans

    def test_input_embeddings(self):
        tx, jx = self._get_inputs()
        with torch.inference_mode():
            ty = self.baseline_model.encoder.embeddings(tx)
        jy = self.bonsai_model.encoder.embeddings(jx)
        self.assert_parity(ty, jy, rtol=1e-5, atol=2e-3)

    def test_first_layer(self):
        tx, jx = self._get_inputs()
        with torch.inference_mode():
            torch_hidden = self.baseline_model.encoder.embeddings(tx)
            ty = self.baseline_model.encoder.layer[0](torch_hidden, position_mask=None, output_attentions=False)[0]
        jax_hidden = self.bonsai_model.encoder.embeddings(jx)
        jy = self.bonsai_model.encoder.layer[0](jax_hidden, None)
        self.assert_parity(ty, jy, rtol=1e-5, atol=1e-3)

    def test_rope_attention(self):
        tx, jx = self._get_inputs()
        with torch.inference_mode():
            torch_hidden = self.baseline_model.encoder.layer[0].norm1(self.baseline_model.encoder.embeddings(tx))
            ty = self.baseline_model.encoder.layer[0].attention(
                torch_hidden, position_mask=None, output_attentions=False
            )[0]
        jax_hidden = self.bonsai_model.encoder.layer[0].norm1(self.bonsai_model.encoder.embeddings(jx))
        jy = self.bonsai_model.encoder.layer[0].attention(jax_hidden, None)
        self.assert_parity(ty, jy, rtol=1e-5, atol=2e-5)

    def test_last_hidden_state(self):
        tx, jx = self._get_inputs()
        with torch.inference_mode():
            ty = self.baseline_model(tx, skip_predictor=True).last_hidden_state
        jy = self.bonsai_model(jx, skip_predictor=True)["last_hidden_state"]
        self.assert_parity(ty, jy, rtol=1e-5, atol=2e-3)


class TestVideoClassification(ParityTestBase):
    """Test video classification model with small manually-initialized models."""

    def setUp(self):
        super().setUp()
        self.save_dir = constants.default_cache_path
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
        self.baseline_model = VJEPA2TorchClassif(config=self.hfconfig)
        self.model_ckpt_path = os.path.join(self.save_dir, "classifier_model.safetensors")
        save_file(self.baseline_model.state_dict(), self.model_ckpt_path)
        self.config = model_lib.ModelConfig(
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
        if os.path.exists(self.model_ckpt_path):
            os.remove(self.model_ckpt_path)
        self.bonsai_model.eval()
        self.baseline_model.eval()
        self.batch_size, self.num_frames = 1, self.config.frames_per_clip
        self.height, self.width, self.channels = self.config.crop_size, self.config.crop_size, self.config.in_chans

    def test_classification_logits(self):
        tx, jx = self._get_inputs()
        with torch.inference_mode():
            ty = self.baseline_model(tx).logits
        jy = self.bonsai_model(jx)["logits"]
        self.assert_parity(ty, jy, rtol=1e-5, atol=1e-2)


class TestPretrainedFoundationModel(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hf_repo = "facebook/vjepa2-vitl-fpc64-256"
        cls.model_dir = snapshot_download(cls.hf_repo)
        cls.torch_model = VJEPA2Model.from_pretrained(cls.hf_repo).eval()
        cls.jax_config = model_lib.ModelConfig.vitl_fpc64_256()
        cls.jax_model = params.create_model_from_safe_tensors(cls.model_dir, cls.jax_config, classifier=False)
        cls.jax_model.eval()

    def _prepare_inputs(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        video = np.random.randn(1, 16, 3, 256, 256).astype(np.float32)
        return torch.tensor(video), jnp.asarray(video.transpose(0, 1, 3, 4, 2))

    def test_rope_attention_pretrained(self):
        tx, jx = self._prepare_inputs()
        with torch.inference_mode():
            torch_hidden = self.torch_model.encoder.layer[0].norm1(self.torch_model.encoder.embeddings(tx))
            ty = self.torch_model.encoder.layer[0].attention(torch_hidden, None, output_attentions=False)[0]
        jax_hidden = self.jax_model.encoder.layer[0].norm1(self.jax_model.encoder.embeddings(jx))
        jy = self.jax_model.encoder.layer[0].attention(jax_hidden, None)
        torch.testing.assert_close(torch.tensor(np.asarray(jax.device_get(jy))), ty, rtol=1e-5, atol=2e-2)

    def test_encoder_only(self):
        tx, jx = self._prepare_inputs()
        with torch.inference_mode():
            ty = self.torch_model(tx, skip_predictor=True).last_hidden_state
        jy = self.jax_model(jx, skip_predictor=True)["last_hidden_state"]
        torch.testing.assert_close(torch.tensor(np.asarray(jax.device_get(jy))), ty, rtol=1e-5, atol=6e-1)

    def test_full_model_output(self):
        tx, jx = self._prepare_inputs()
        with torch.inference_mode():
            t_out = self.torch_model(tx, skip_predictor=False)
        j_out = self.jax_model(jx, skip_predictor=False)

        j_hid = torch.tensor(np.asarray(jax.device_get(j_out["last_hidden_state"])))
        j_pred = torch.tensor(np.asarray(jax.device_get(j_out["predictor_last_hidden_state"])))

        torch.testing.assert_close(j_hid, t_out.last_hidden_state, rtol=1e-5, atol=6e-1)
        torch.testing.assert_close(j_pred, t_out.predictor_output.last_hidden_state, rtol=1e-1, atol=5e-1)


class TestPretrainedClassificationModel(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
        cls.model_dir = snapshot_download(cls.hf_repo)
        cls.torch_model = VJEPA2ForVideoClassification.from_pretrained(cls.hf_repo).eval()
        cls.jax_config = model_lib.ModelConfig.vitl_fpc16_256()
        cls.jax_model = params.create_model_from_safe_tensors(cls.model_dir, cls.jax_config, classifier=True)
        cls.jax_model.eval()
        cls.processor = AutoVideoProcessor.from_pretrained(cls.hf_repo)

    def _prepare_inputs(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        video = np.random.randn(self.jax_config.frames_per_clip, 3, 256, 256).astype(np.float32)
        tx = self.processor(video, return_tensors="pt").pixel_values_videos
        jx = jnp.asarray(tx.numpy()).transpose(0, 1, 3, 4, 2)
        return tx, jx

    def test_classification_logits(self):
        tx, jx = self._prepare_inputs()
        with torch.inference_mode():
            ty = self.torch_model(tx).logits
        jy = self.jax_model(jx)["logits"]
        torch.testing.assert_close(torch.tensor(np.asarray(jax.device_get(jy))), ty, rtol=1e-5, atol=6e-2)

    def test_top_k_predictions(self):
        tx, jx = self._prepare_inputs()
        with torch.inference_mode():
            torch_logits = self.torch_model(tx).logits
        torch_top5 = torch.topk(torch_logits, 5).indices[0].numpy()

        jax_logits = self.jax_model(jx)["logits"]
        jax_logits_np = np.asarray(jax.device_get(jax_logits))
        jax_top5 = np.argsort(jax_logits_np[0])[-5:][::-1]

        np.testing.assert_array_equal(jax_top5, torch_top5)


if __name__ == "__main__":
    absltest.main()
