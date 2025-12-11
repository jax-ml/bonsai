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

"""Tests for Mamba2 model outputs.

Run with: python -m absl.testing.absltest bonsai/models/mamba2/tests/test_outputs.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from bonsai.models.mamba2 import modeling, params


class TestMamba2Config(absltest.TestCase):
    """Tests for Mamba2Config."""

    def test_default_config(self):
        """Test default config values."""
        cfg = modeling.Mamba2Config()
        self.assertEqual(cfg.vocab_size, 50280)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.num_hidden_layers, 24)

    def test_intermediate_size(self):
        """Test intermediate_size property."""
        cfg = modeling.Mamba2Config(hidden_size=512, expand=2)
        self.assertEqual(cfg.intermediate_size, 1024)

    def test_num_heads(self):
        """Test num_heads property."""
        cfg = modeling.Mamba2Config(hidden_size=512, expand=2, head_dim=64)
        # intermediate_size = 1024, head_dim = 64 -> num_heads = 16
        self.assertEqual(cfg.num_heads, 16)

    def test_predefined_configs(self):
        """Test predefined configuration methods."""
        cfg_tiny = modeling.Mamba2Config.tiny()
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
        self.cfg = modeling.Mamba2Config.tiny()
        self.model = modeling.Mamba2Model(self.cfg, rngs=nnx.Rngs(42))

    def test_output_shape(self):
        """Test Mamba2Model output shape."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)

        self.assertEqual(outputs["last_hidden_state"].shape, (batch_size, seq_len, self.cfg.hidden_size))
        self.assertIsNone(outputs["hidden_states"])
        self.assertIsNone(outputs["last_ssm_states"])

    def test_output_hidden_states(self):
        """Test output_hidden_states flag."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        self.assertIsNotNone(outputs["hidden_states"])
        # num_layers + 1 (final norm output)
        self.assertLen(outputs["hidden_states"], self.cfg.num_hidden_layers + 1)

    def test_inputs_embeds(self):
        """Test using inputs_embeds instead of input_ids."""
        batch_size, seq_len = 2, 32
        inputs_embeds = jnp.ones((batch_size, seq_len, self.cfg.hidden_size))
        outputs = self.model(inputs_embeds=inputs_embeds)

        self.assertEqual(outputs["last_hidden_state"].shape, (batch_size, seq_len, self.cfg.hidden_size))

    def test_no_nans(self):
        """Test that outputs don't contain NaNs."""
        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)
        self.assertFalse(jnp.any(jnp.isnan(outputs["last_hidden_state"])))

    def test_invalid_inputs(self):
        """Test that providing both input_ids and inputs_embeds raises error."""
        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        inputs_embeds = jnp.ones((2, 32, self.cfg.hidden_size))
        with self.assertRaises(ValueError):
            self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)


class TestMamba2ForCausalLM(absltest.TestCase):
    """Tests for Mamba2ForCausalLM."""

    def setUp(self):
        super().setUp()
        self.cfg = modeling.Mamba2Config.tiny()
        self.model = modeling.Mamba2ForCausalLM(self.cfg, rngs=nnx.Rngs(42))

    def test_output_shape(self):
        """Test Mamba2ForCausalLM logits shape."""
        batch_size, seq_len = 2, 32
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        outputs = self.model(input_ids=input_ids)

        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, self.cfg.vocab_size))
        self.assertIsNone(outputs["loss"])

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

    def test_count_parameters(self):
        """Test parameter counting."""
        cfg = modeling.Mamba2Config.tiny()
        model = modeling.Mamba2ForCausalLM(cfg, rngs=nnx.Rngs(0))
        num_params = params.count_parameters(model)
        self.assertGreater(num_params, 0)
        self.assertIsInstance(num_params, int)

    def test_create_random_model(self):
        """Test random model creation."""
        cfg = modeling.Mamba2Config.tiny()
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
        self.cfg = modeling.Mamba2Config.tiny()

    def test_jit_backbone(self):
        """Test that backbone can be JIT compiled."""
        model = modeling.Mamba2Model(self.cfg, rngs=nnx.Rngs(42))

        @jax.jit
        def forward_fn(model, x):
            return model(input_ids=x)

        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        outputs = forward_fn(model, input_ids)
        self.assertEqual(outputs["last_hidden_state"].shape, (2, 32, 64))

    def test_jit_causal_lm(self):
        """Test that CausalLM can be JIT compiled."""
        model = modeling.Mamba2ForCausalLM(self.cfg, rngs=nnx.Rngs(42))

        input_ids = jnp.ones((2, 32), dtype=jnp.int32)
        labels = jnp.ones((2, 32), dtype=jnp.int32)
        outputs = modeling.forward(model, input_ids, labels)
        self.assertIsNotNone(outputs["loss"])


class TestGradients(absltest.TestCase):
    """Tests for gradient computation."""

    def setUp(self):
        super().setUp()
        self.cfg = modeling.Mamba2Config.tiny()

    def test_gradients_exist(self):
        """Test that gradients can be computed."""
        model = modeling.Mamba2ForCausalLM(self.cfg, rngs=nnx.Rngs(42))

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
        model = modeling.Mamba2ForCausalLM(self.cfg, rngs=nnx.Rngs(42))

        def loss_fn(model, x, labels):
            outputs = model(input_ids=x, labels=labels)
            return outputs["loss"]

        input_ids = jnp.ones((2, 16), dtype=jnp.int32)
        labels = jnp.ones((2, 16), dtype=jnp.int32)

        loss, _grads = nnx.value_and_grad(loss_fn)(model, input_ids, labels)
        self.assertFalse(jnp.isnan(loss))


if __name__ == "__main__":
    absltest.main()
