"""Numerical-parity tests for the Bonsai (JAX/NNX) gpt_oss port.

Strategy
--------
There is no small public ``gpt-oss`` checkpoint (the released models are
20B/120B and MXFP4-quantized), so instead of downloading a checkpoint we:

  1. build a *tiny* random HuggingFace ``GptOssForCausalLM``,
  2. ``save_pretrained`` it to a temp dir as safetensors,
  3. load those exact weights into the Bonsai model through the real
     ``params.create_gpt_oss_from_pretrained`` path.

Both models therefore hold identical weights, so every submodule must
produce numerically identical outputs (up to float tolerance).

Two known gaps between the Bonsai port and HF gpt_oss are neutralized so
the suite isolates real regressions rather than re-reporting them:

  * **Attention sinks** -- HF appends a per-head learned "sink" logit to
    the attention softmax (``eager_attention_forward``). The Bonsai
    ``GptOssAttention`` declares ``self.sinks`` but never applies it. We
    set the HF sinks to a large negative value so ``exp(sink) -> 0`` and
    HF reduces to the plain softmax the Bonsai port implements. To test
    the sink path, implement it in ``modeling.GptOssAttention`` and drop
    the override in ``_init_hf_weights``.
  * **Sliding-window attention** -- HF alternates sliding/full attention
    layers; the Bonsai port is always full-causal. We pin all HF layers
    to ``"full_attention"`` in the config so the two agree.
"""

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from bonsai.models.gpt_oss import modeling, params

# Large-negative sink so HF's softmax sink column vanishes and HF matches
# the Bonsai port's plain-softmax attention. See module docstring.
_DEAD_SINK = -1e30


def _make_configs():
    """A small config shared by the HF reference and the Bonsai model."""
    vocab_size = 64
    hidden_size = 32
    intermediate_size = 48
    num_hidden_layers = 2
    num_local_experts = 4
    num_experts_per_tok = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 8
    rms_norm_eps = 1e-6
    rope_theta = 10000.0
    max_position_embeddings = 64

    hf_config = GptOssConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        attention_bias=True,
        # Window > seq_len: the sliding mask equals full-causal for our short
        # sequences. HF builds it unconditionally, so it must be a valid int.
        sliding_window=4096,
        # Pin every layer to full attention so the Bonsai full-causal port
        # matches (Bonsai does not implement sliding-window attention).
        layer_types=["full_attention"] * num_hidden_layers,
        # Plain RoPE so HF matches modeling.create_rope_embeddings.
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
        tie_word_embeddings=False,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    hf_config._attn_implementation = "eager"  # _supports_sdpa is False anyway

    bonsai_config = modeling.GptOssConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        attention_bias=True,
        sliding_window=None,
        rms_norm_eps=rms_norm_eps,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    return hf_config, bonsai_config


def _init_hf_weights(model: GptOssForCausalLM):
    """Deterministically fill every parameter (HF uses torch.empty for some).

    Norm weights are centered at 1.0 (realistic + numerically stable);
    everything else is small Gaussian. Attention sinks are forced to a
    large negative value -- see module docstring.
    """
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("sinks"):
                p.fill_(_DEAD_SINK)
            elif "norm" in name and p.dim() == 1:
                p.normal_(mean=1.0, std=0.02)
            else:
                p.normal_(mean=0.0, std=0.02)
    model.eval()


class TestModuleForwardPasses(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        jax.config.update("jax_default_matmul_precision", "float32")

        cls.hf_config, cls.bonsai_config = _make_configs()

        cls.torch_model = GptOssForCausalLM(cls.hf_config)
        _init_hf_weights(cls.torch_model)

        # Round-trip through safetensors so the real param-loading code runs.
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.torch_model.save_pretrained(cls._tmpdir.name, safe_serialization=True)

        graph_def, state = nnx.split(
            params.create_gpt_oss_from_pretrained(cls._tmpdir.name, cls.bonsai_config)
        )
        state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, state)
        cls.nnx_model = nnx.merge(graph_def, state)

        cls.batch_size = 2
        cls.seq_len = 6
        cls.hidden = cls.bonsai_config.hidden_size
        cls.head_dim = cls.bonsai_config.head_dim
        cls.tol = 1e-4
        cls.full_tol = 2e-3

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()
        super().tearDownClass()

    # ---- helpers ---------------------------------------------------------

    def _assert_close(self, jy, ty, tol):
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty.to(torch.float32),
            rtol=tol,
            atol=tol,
            check_dtype=False,
        )

    def _rand_hidden(self, key=0):
        shape = (self.batch_size, self.seq_len, self.hidden)
        jx = jax.random.normal(jax.random.key(key), shape, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        return jx, tx

    def _bonsai_rope_and_mask(self):
        sins, coss = modeling.create_rope_embeddings(
            self.seq_len, self.head_dim, self.bonsai_config.rope_theta
        )
        mask = modeling.make_causal_mask(self.seq_len)
        mask = jnp.where(mask, 0, -1e9)
        return sins, coss, mask

    def _torch_attn_inputs(self, hidden_t):
        """Reproduce the position-embeddings + causal mask GptOssModel builds."""
        past = DynamicCache(config=self.hf_config)
        cache_position = torch.arange(self.seq_len)
        position_ids = cache_position.unsqueeze(0)
        mask = create_causal_mask(
            config=self.hf_config,
            input_embeds=hidden_t,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past,
            position_ids=position_ids,
        )
        cos, sin = self.torch_model.model.rotary_emb(hidden_t, position_ids)
        return {
            "attention_mask": mask,
            "position_ids": position_ids,
            "position_embeddings": (cos, sin),
            "past_key_values": None,
            "use_cache": False,
        }

    # ---- leaf modules ----------------------------------------------------

    def test_embed_tokens(self):
        ids = np.random.randint(0, self.bonsai_config.vocab_size, size=(self.batch_size, self.seq_len))
        jy = self.nnx_model.model.embed_tokens(jnp.asarray(ids))
        ty = self.torch_model.model.embed_tokens(torch.tensor(ids))
        self._assert_close(jy, ty, self.tol)

    def test_rms_norm(self):
        jx, tx = self._rand_hidden()
        jy = self.nnx_model.model.layers[0].input_layernorm(jx)
        ty = self.torch_model.model.layers[0].input_layernorm(tx)
        self._assert_close(jy, ty, self.tol)

    def test_final_norm(self):
        jx, tx = self._rand_hidden()
        jy = self.nnx_model.model.norm(jx)
        ty = self.torch_model.model.norm(tx)
        self._assert_close(jy, ty, self.tol)

    def test_q_proj(self):
        jx, tx = self._rand_hidden()
        jy = self.nnx_model.model.layers[0].self_attn.q_proj(jx)
        ty = self.torch_model.model.layers[0].self_attn.q_proj(tx)
        self._assert_close(jy, ty, self.tol)

    def test_k_proj(self):
        jx, tx = self._rand_hidden()
        jy = self.nnx_model.model.layers[0].self_attn.k_proj(jx)
        ty = self.torch_model.model.layers[0].self_attn.k_proj(tx)
        self._assert_close(jy, ty, self.tol)

    def test_v_proj(self):
        jx, tx = self._rand_hidden()
        jy = self.nnx_model.model.layers[0].self_attn.v_proj(jx)
        ty = self.torch_model.model.layers[0].self_attn.v_proj(tx)
        self._assert_close(jy, ty, self.tol)

    def test_o_proj(self):
        n_heads = self.bonsai_config.num_attention_heads
        shape = (self.batch_size, self.seq_len, n_heads * self.head_dim)
        jx = jax.random.normal(jax.random.key(0), shape, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        jy = self.nnx_model.model.layers[0].self_attn.o_proj(jx)
        ty = self.torch_model.model.layers[0].self_attn.o_proj(tx)
        self._assert_close(jy, ty, self.tol)

    def test_lm_head(self):
        jx, tx = self._rand_hidden()
        jy = self.nnx_model.lm_head(jx)
        ty = self.torch_model.lm_head(tx)
        self._assert_close(jy, ty, self.tol)

    def test_rope_sin_cos(self):
        sins, coss = modeling.create_rope_embeddings(
            self.seq_len, self.head_dim, self.bonsai_config.rope_theta
        )
        hidden_t = torch.ones((1, self.seq_len, self.hidden))
        position_ids = torch.arange(self.seq_len).unsqueeze(0)
        cos_t, sin_t = self.torch_model.model.rotary_emb(hidden_t, position_ids)
        # HF returns (B, S, head_dim/2); Bonsai returns (S, head_dim/2).
        self._assert_close(sins, sin_t[0], self.tol)
        self._assert_close(coss, cos_t[0], self.tol)

    # ---- composite modules ----------------------------------------------

    def test_mlp(self):
        # Covers GptOssTopKRouter + GptOssExperts (different internal score
        # representations, but the routed output must agree).
        jx, tx = self._rand_hidden()
        jy, _ = self.nnx_model.model.layers[0].mlp(jx)
        ty, _ = self.torch_model.model.layers[0].mlp(tx)
        self._assert_close(jy, ty, self.tol)

    def test_attention(self):
        jx, tx = self._rand_hidden()
        sins, coss, mask = self._bonsai_rope_and_mask()
        jy = self.nnx_model.model.layers[0].self_attn(jx, sins, coss, mask)

        ti = self._torch_attn_inputs(tx)
        ty, _ = self.torch_model.model.layers[0].self_attn(
            hidden_states=tx,
            position_embeddings=ti["position_embeddings"],
            attention_mask=ti["attention_mask"],
            past_key_values=None,
        )
        self._assert_close(jy, ty, self.tol)

    def test_decoder_layer(self):
        jx, tx = self._rand_hidden()
        sins, coss, mask = self._bonsai_rope_and_mask()
        jy = self.nnx_model.model.layers[0](jx, sins, coss, mask)

        ti = self._torch_attn_inputs(tx)
        ty = self.torch_model.model.layers[0](hidden_states=tx, **ti)
        self._assert_close(jy, ty, self.tol)

    def test_all_decoder_layers(self):
        for idx in range(self.bonsai_config.num_hidden_layers):
            jx, tx = self._rand_hidden(key=idx)
            sins, coss, mask = self._bonsai_rope_and_mask()
            jy = self.nnx_model.model.layers[idx](jx, sins, coss, mask)
            ti = self._torch_attn_inputs(tx)
            ty = self.torch_model.model.layers[idx](hidden_states=tx, **ti)
            self._assert_close(jy, ty, self.tol)

    # ---- full model ------------------------------------------------------

    def test_full(self):
        ids = np.random.randint(0, self.bonsai_config.vocab_size, size=(1, self.seq_len))
        jy = self.nnx_model(jnp.asarray(ids))
        with torch.no_grad():
            ty = self.torch_model(input_ids=torch.tensor(ids)).logits
        self._assert_close(jy, ty, self.full_tol)

    def test_full_batched(self):
        ids = np.random.randint(
            0, self.bonsai_config.vocab_size, size=(self.batch_size, self.seq_len)
        )
        jy = self.nnx_model(jnp.asarray(ids))
        with torch.no_grad():
            ty = self.torch_model(input_ids=torch.tensor(ids)).logits
        self._assert_close(jy, ty, self.full_tol)


if __name__ == "__main__":
    absltest.main()
