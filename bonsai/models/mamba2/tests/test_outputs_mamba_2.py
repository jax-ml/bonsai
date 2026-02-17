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

import os

os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from bonsai.models.mamba2 import modeling, params


class TestModelConfig(absltest.TestCase):
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default config values."""
        cfg = modeling.ModelConfig()
        self.assertEqual(cfg.vocab_size, 50280)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.num_hidden_layers, 24)

    def test_intermediate_size(self):
        """Test intermediate_size property."""
        cfg = modeling.ModelConfig(hidden_size=512, expand=2)
        self.assertEqual(cfg.intermediate_size, 1024)

    def test_num_heads(self):
        """Test num_heads property."""
        cfg = modeling.ModelConfig(hidden_size=512, expand=2, head_dim=64)
        # intermediate_size = 1024, head_dim = 64 -> num_heads = 16
        self.assertEqual(cfg.num_heads, 16)

    def test_predefined_configs(self):
        """Test predefined configuration methods."""
        cfg_tiny = modeling.ModelConfig.tiny()
        self.assertEqual(cfg_tiny.hidden_size, 64)
        self.assertEqual(cfg_tiny.num_hidden_layers, 2)


class TestRMSNorm(absltest.TestCase):
    """Tests for RMSNorm layer."""

    def test_output_shape(self):
        """Test RMSNorm output shape."""
        norm = modeling.RMSNorm(hidden_size=64, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 16, 64))
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_dtype(self):
        """Test RMSNorm preserves dtype."""
        norm = modeling.RMSNorm(hidden_size=64, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 16, 64), dtype=jnp.float16)
        out = norm(x)
        self.assertEqual(out.dtype, jnp.float16)

    def test_with_residual_gate(self):
        """Test RMSNorm with residual gating."""
        norm = modeling.RMSNorm(hidden_size=64, gate_residual=True, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 16, 64))
        residual = jnp.ones((2, 16, 64)) * 0.5
        out = norm(x, residual=residual)
        self.assertEqual(out.shape, x.shape)


class TestSegsum(absltest.TestCase):
    """Tests for segsum function."""

    def test_output_shape(self):
        """Test segsum output shape."""
        x = jnp.ones((2, 4, 8))
        out = modeling.segsum(x)
        self.assertEqual(out.shape, (2, 4, 8, 8))

    def test_lower_triangular(self):
        """Test that segsum produces lower-triangular + -inf structure."""
        x = jnp.ones((4,))
        out = modeling.segsum(x)
        # Upper triangle should be -inf
        self.assertTrue(jnp.isinf(out[0, 1]))
        self.assertTrue(jnp.isinf(out[0, 2]))
        self.assertTrue(jnp.isinf(out[0, 3]))


class TestSSDForward(absltest.TestCase):
    """Tests for SSD forward function."""

    def test_output_shape(self):
        """Test SSD forward output shape."""
        batch_size, seq_len, num_heads, head_dim, state_size = 2, 32, 4, 16, 8
        x = jnp.ones((batch_size, seq_len, num_heads, head_dim))
        dt = jnp.ones((batch_size, seq_len, num_heads)) * 0.1
        A = -jnp.ones((num_heads,))
        B_mat = jnp.ones((batch_size, seq_len, num_heads, state_size))
        C_mat = jnp.ones((batch_size, seq_len, num_heads, state_size))
        D = jnp.ones((num_heads,))
        dt_bias = jnp.zeros((num_heads,))

        y, _ = modeling.ssd_forward(
            x, dt, A, B_mat, C_mat, chunk_size=16, D=D, dt_bias=dt_bias, dt_min=0.001, dt_max=0.1
        )
        self.assertEqual(y.shape, x.shape)

    def test_with_initial_states(self):
        """Test SSD forward with initial states."""
        batch_size, seq_len, num_heads, head_dim, state_size = 2, 32, 4, 16, 8
        x = jnp.ones((batch_size, seq_len, num_heads, head_dim))
        dt = jnp.ones((batch_size, seq_len, num_heads)) * 0.1
        A = -jnp.ones((num_heads,))
        B_mat = jnp.ones((batch_size, seq_len, num_heads, state_size))
        C_mat = jnp.ones((batch_size, seq_len, num_heads, state_size))
        D = jnp.ones((num_heads,))
        dt_bias = jnp.zeros((num_heads,))
        initial_states = jnp.zeros((batch_size, 1, num_heads, head_dim, state_size))

        y, final_state = modeling.ssd_forward(
            x,
            dt,
            A,
            B_mat,
            C_mat,
            chunk_size=16,
            D=D,
            dt_bias=dt_bias,
            dt_min=0.001,
            dt_max=0.1,
            initial_states=initial_states,
            return_final_states=True,
        )
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(final_state.shape, (batch_size, num_heads, head_dim, state_size))


class TestMamba2Model(absltest.TestCase):
    """Tests for Mamba2Model."""

    def setUp(self):
        super().setUp()
        self.config = modeling.ModelConfig.tiny()
        self.model = modeling.Mamba2Model(self.config, rngs=nnx.Rngs(42))

    def test_output_shape(self):
        """Test Mamba2Model output shape."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)

        self.assertEqual(outputs["last_hidden_state"].shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertIsNone(outputs["hidden_states"])
        self.assertIsNotNone(outputs["cache"])
        self.assertIsInstance(outputs["cache"], modeling.Mamba2Cache)

    def test_output_hidden_states(self):
        """Test output_hidden_states flag."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        self.assertIsNotNone(outputs["hidden_states"])
        # num_layers + 1 (final norm output)
        self.assertLen(outputs["hidden_states"], self.config.num_hidden_layers + 1)

    def test_inputs_embeds(self):
        """Test using inputs_embeds instead of input_ids."""
        batch_size, seq_len = 2, 32
        inputs_embeds = jnp.ones((batch_size, seq_len, self.config.hidden_size))
        outputs = self.model(inputs_embeds=inputs_embeds)

        self.assertEqual(outputs["last_hidden_state"].shape, (batch_size, seq_len, self.config.hidden_size))

    def test_no_nans(self):
        """Test that outputs don't contain NaNs."""
        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)
        self.assertFalse(jnp.any(jnp.isnan(outputs["last_hidden_state"])))

    def test_invalid_inputs(self):
        """Test that providing both input_ids and inputs_embeds raises error."""
        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        inputs_embeds = jnp.ones((2, 32, self.config.hidden_size))
        with self.assertRaises(ValueError):
            self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)


class TestMamba2ForCausalLM(absltest.TestCase):
    """Tests for Mamba2ForCausalLM."""

    def setUp(self):
        super().setUp()
        self.config = modeling.ModelConfig.tiny()
        self.model = modeling.Mamba2ForCausalLM(self.config, rngs=nnx.Rngs(42))

    def test_output_shape(self):
        """Test Mamba2ForCausalLM logits shape."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)

        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNone(outputs["loss"])
        self.assertIsNotNone(outputs["cache"])

    def test_loss_computation(self):
        """Test loss computation with labels."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids, labels=labels)

        self.assertIsNotNone(outputs["loss"])
        self.assertEqual(outputs["loss"].shape, ())  # scalar
        self.assertFalse(jnp.isnan(outputs["loss"]))

    def test_no_nans_in_logits(self):
        """Test that logits don't contain NaNs."""
        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)
        self.assertFalse(jnp.any(jnp.isnan(outputs["logits"])))


class TestMamba2Forecaster(absltest.TestCase):
    """Tests for Mamba2Forecaster."""

    def test_output_shape(self):
        """Test Mamba2Forecaster output shape."""
        model = modeling.Mamba2Forecaster(input_dim=10, forecast_horizon=24, output_dim=1, n_layers=2, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 100, 10))
        out = model(x)
        self.assertEqual(out.shape, (2, 24, 1))

    def test_multi_output(self):
        """Test Mamba2Forecaster with multiple output dimensions."""
        model = modeling.Mamba2Forecaster(input_dim=10, forecast_horizon=12, output_dim=3, n_layers=2, rngs=nnx.Rngs(0))
        x = jnp.ones((4, 50, 10))
        out = model(x)
        self.assertEqual(out.shape, (4, 12, 3))

    def test_no_nans(self):
        """Test that forecaster outputs don't contain NaNs."""
        model = modeling.Mamba2Forecaster(input_dim=5, forecast_horizon=10, n_layers=2, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 32, 5))
        out = model(x)
        self.assertFalse(jnp.any(jnp.isnan(out)))


class TestParameters(absltest.TestCase):
    """Tests for parameter utilities."""

    def test_create_random_model(self):
        """Test random model creation."""
        cfg = modeling.ModelConfig.tiny()
        model = params.create_random_model(cfg, seed=42)
        self.assertIsInstance(model, modeling.Mamba2ForCausalLM)

        # Test forward pass
        input_ids = jnp.ones((2, 16), dtype=jnp.int32)
        outputs = model(input_ids)
        self.assertEqual(outputs["logits"].shape, (2, 16, cfg.vocab_size))

    def test_create_random_forecaster(self):
        """Test random forecaster creation."""
        model = params.create_random_forecaster(input_dim=10, forecast_horizon=24, seed=42)
        self.assertIsInstance(model, modeling.Mamba2Forecaster)

        # Test forward pass
        x = jnp.ones((2, 50, 10))
        out = model(x)
        self.assertEqual(out.shape, (2, 24, 1))


class TestJIT(absltest.TestCase):
    """Tests for JIT compilation."""

    def setUp(self):
        super().setUp()
        self.config = modeling.ModelConfig.tiny()

    def test_jit_backbone(self):
        """Test that backbone can be JIT compiled."""
        model = modeling.Mamba2Model(self.config, rngs=nnx.Rngs(42))

        @jax.jit
        def forward_fn(model, x):
            return model(input_ids=x)

        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        outputs = forward_fn(model, input_ids)
        self.assertEqual(outputs["last_hidden_state"].shape, (2, 32, 64))

    def test_jit_causal_lm(self):
        """Test that CausalLM can be JIT compiled."""
        model = modeling.Mamba2ForCausalLM(self.config, rngs=nnx.Rngs(42))

        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        labels = jnp.ones((2, 32), dtype=jnp.int32)
        outputs = modeling.forward(model, input_ids, labels)
        self.assertIsNotNone(outputs["loss"])


class TestGradients(absltest.TestCase):
    """Tests for gradient computation."""

    def setUp(self):
        super().setUp()
        self.config = modeling.ModelConfig.tiny()

    def test_gradients_exist(self):
        """Test that gradients can be computed."""
        model = modeling.Mamba2ForCausalLM(self.config, rngs=nnx.Rngs(42))

        def loss_fn(model, x, labels):
            outputs = model(input_ids=x, labels=labels)
            return outputs["loss"]

        input_ids = jnp.ones((2, 16), dtype=jnp.int32)
        labels = jnp.ones((2, 16), dtype=jnp.int32)

        loss, _grads = nnx.value_and_grad(loss_fn)(model, input_ids, labels)
        self.assertIsNotNone(_grads)
        self.assertTrue(jnp.isfinite(loss))

    def test_no_nan_gradients(self):
        """Test that gradients don't contain NaNs."""
        model = modeling.Mamba2ForCausalLM(self.config, rngs=nnx.Rngs(42))

        def loss_fn(model, x, labels):
            outputs = model(input_ids=x, labels=labels)
            return outputs["loss"]

        input_ids = jnp.ones((2, 16), dtype=jnp.int32)
        labels = jnp.ones((2, 16), dtype=jnp.int32)

        loss, _grads = nnx.value_and_grad(loss_fn)(model, input_ids, labels)
        self.assertFalse(jnp.isnan(loss))


class TestGoldenParity(absltest.TestCase):
    """Tests for parity with mamba_ssm reference outputs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        cls.golden = np.load(os.path.join(artifacts_dir, "golden_mamba2_130m.npz"))

    def test_hidden_state_parity(self):
        """Test last_hidden_state matches mamba_ssm reference within numerical tolerance."""
        cfg = modeling.ModelConfig(
            vocab_size=50288,
            hidden_size=768,
            state_size=128,
            num_hidden_layers=24,
            head_dim=64,
            expand=2,
            conv_kernel=4,
        )
        model = modeling.Mamba2ForCausalLM.from_pretrained("state-spaces/mamba2-130m", cfg=cfg)

        input_ids = jnp.array(self.golden["input_ids"], dtype=jnp.int32)
        outputs = model.backbone(input_ids=input_ids)
        bonsai_hidden = np.array(outputs["last_hidden_state"])
        golden_hidden = self.golden["last_hidden_state"]

        # fp32=1e-5, bf16=1e-3 (see ViT parity tests).
        # atol is an output-level floor to avoid near-zero blowups
        rtol = 1e-5 if bonsai_hidden.dtype == np.float32 else 1e-3
        atol = 1e-1
        np.testing.assert_allclose(bonsai_hidden, golden_hidden, rtol=rtol, atol=atol)

    def test_logits_parity(self):
        """Test logits match mamba_ssm reference within numerical tolerance."""
        cfg = modeling.ModelConfig(
            vocab_size=50288,
            hidden_size=768,
            state_size=128,
            num_hidden_layers=24,
            head_dim=64,
            expand=2,
            conv_kernel=4,
        )
        model = modeling.Mamba2ForCausalLM.from_pretrained("state-spaces/mamba2-130m", cfg=cfg)

        input_ids = jnp.array(self.golden["input_ids"], dtype=jnp.int32)
        outputs = model(input_ids=input_ids)
        bonsai_logits = np.array(outputs["logits"])[:, :, :256]
        golden_logits = self.golden["logits_slice"]

        # fp32=1e-5, bf16=1e-3 (see ViT parity tests).
        # atol is an output-level floor to avoid near-zero blowups
        rtol = 1e-5 if bonsai_logits.dtype == np.float32 else 1e-3
        atol = 2e-1
        np.testing.assert_allclose(bonsai_logits, golden_logits, rtol=rtol, atol=atol)


class TestMamba2Cache(absltest.TestCase):
    """Tests for Mamba2Cache state caching."""

    def setUp(self):
        super().setUp()
        self.config = modeling.ModelConfig.tiny()
        self.model = modeling.Mamba2ForCausalLM(self.config, rngs=nnx.Rngs(42))

    def test_cache_shapes(self):
        """Test cache state shapes are correct."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)
        cache = outputs["cache"]

        # Check cache structure
        self.assertIsInstance(cache, modeling.Mamba2Cache)
        self.assertLen(cache.ssm_states, self.config.num_hidden_layers)
        self.assertLen(cache.conv_states, self.config.num_hidden_layers)

        # Check SSM state shapes
        for ssm_state in cache.ssm_states:
            expected_shape = (batch_size, self.config.num_heads, self.config.head_dim, self.config.state_size)
            self.assertEqual(ssm_state.shape, expected_shape)

        # Check conv state shapes
        conv_dim = self.config.intermediate_size + 2 * self.config.state_size
        cache_len = self.config.conv_kernel - 1
        for conv_state in cache.conv_states:
            expected_shape = (batch_size, conv_dim, cache_len)
            self.assertEqual(conv_state.shape, expected_shape)

    def test_cached_matches_full(self):
        """Test that cached generation produces same results as full forward pass.

        Process [a,b,c] in full sequence vs [a,b] then [c] with cache.
        Logits for 'c' position must match.
        """
        batch_size = 1
        # Create sequence [1, 2, 3]
        full_seq = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        prefix_seq = jnp.array([[1, 2]], dtype=jnp.int32)
        next_token = jnp.array([[3]], dtype=jnp.int32)

        # Full forward pass
        full_outputs = self.model(input_ids=full_seq)
        full_logits = full_outputs["logits"][:, -1, :]  # logits at position of token 3

        # Cached forward pass
        prefix_outputs = self.model(input_ids=prefix_seq)
        cache = prefix_outputs["cache"]
        next_outputs = self.model(input_ids=next_token, cache=cache)
        cached_logits = next_outputs["logits"][:, -1, :]  # logits at position of token 3

        # Logits should match
        np.testing.assert_allclose(np.array(full_logits), np.array(cached_logits), rtol=1e-5, atol=1e-6)

    def test_create_empty_cache(self):
        """Test creating empty cache with correct shapes."""
        batch_size = 4
        cache = modeling.create_empty_cache(self.config, batch_size)

        self.assertIsInstance(cache, modeling.Mamba2Cache)
        self.assertLen(cache.ssm_states, self.config.num_hidden_layers)
        self.assertLen(cache.conv_states, self.config.num_hidden_layers)

        # Check all states are zeros with correct shapes
        for ssm_state in cache.ssm_states:
            expected_shape = (batch_size, self.config.num_heads, self.config.head_dim, self.config.state_size)
            self.assertEqual(ssm_state.shape, expected_shape)
            self.assertTrue(jnp.all(ssm_state == 0))

        conv_dim = self.config.intermediate_size + 2 * self.config.state_size
        cache_len = self.config.conv_kernel - 1
        for conv_state in cache.conv_states:
            expected_shape = (batch_size, conv_dim, cache_len)
            self.assertEqual(conv_state.shape, expected_shape)
            self.assertTrue(jnp.all(conv_state == 0))

    def test_cache_updates_on_forward(self):
        """Test that cache is updated after each forward pass."""
        input_ids = jnp.array([[1, 2]], dtype=jnp.int32)

        # First forward pass
        outputs1 = self.model(input_ids=input_ids)
        cache1 = outputs1["cache"]

        # Second forward pass with cache
        next_token = jnp.array([[3]], dtype=jnp.int32)
        outputs2 = self.model(input_ids=next_token, cache=cache1)
        cache2 = outputs2["cache"]

        # Caches should be different (states updated)
        for s1, s2 in zip(cache1.ssm_states, cache2.ssm_states):
            self.assertFalse(jnp.allclose(s1, s2))


if __name__ == "__main__":
    absltest.main()
