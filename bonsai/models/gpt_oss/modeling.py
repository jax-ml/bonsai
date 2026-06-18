import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional
from dataclasses import dataclass


@dataclass
class GptOssConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    router_aux_loss_coef: float = 0.001
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    attention_bias: bool = False
    sliding_window: Optional[int] = None
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    pad_token_id: int = 0
    attention_dropout: float = 0.0


def create_rope_embeddings(max_seq_len, head_dim, theta=10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2)[: (head_dim // 2)].astype(jnp.float32) / head_dim))
    t = jnp.arange(max_seq_len)
    freqs = jnp.outer(t, freqs)
    return jnp.sin(freqs), jnp.cos(freqs)


def apply_rotary_pos_emb(x, sins, coss):
    d = x.shape[-1] // 2
    x_r, x_i = x[..., :d], x[..., d:]

    sins = sins[None, : x.shape[1], None, :]
    coss = coss[None, : x.shape[1], None, :]

    out_r = x_r * coss - x_i * sins
    out_i = x_r * sins + x_i * coss
    return jnp.concatenate([out_r, out_i], axis=-1)


def make_causal_mask(seq_len, dtype=jnp.float32):
    idx = jnp.arange(seq_len)
    mask = idx[:, None] >= idx[None, :]
    return mask[None, None, :, :]


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, rngs: nnx.Rngs = None):
        self.weight = nnx.Param(jnp.ones(hidden_size))
        self.variance_epsilon = eps

    def __call__(self, hidden_state):
        hidden_state = hidden_state.astype(jnp.float32)
        variance = jnp.mean(hidden_state**2, axis=-1, keepdims=True)
        mean = jnp.sqrt(variance + self.variance_epsilon)
        return (hidden_state / mean) * self.weight.value


class GptOssExperts(nnx.Module):
    def __init__(self, config: GptOssConfig):
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_up_proj = nnx.Param(jnp.zeros((self.num_experts, self.hidden_size, 2 * self.expert_dim)))
        self.gate_up_proj_bias = nnx.Param(jnp.zeros((self.num_experts, 2 * self.expert_dim)))
        self.down_proj = nnx.Param(jnp.zeros((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nnx.Param(jnp.zeros((self.num_experts, self.hidden_size)))

    def __call__(self, hidden_states, router_indices=None, routing_weights=None):
        B, S, H = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, H)
        num_experts = routing_weights.shape[1]

        hidden_rep = jnp.broadcast_to(hidden_flat[None, :, :], (self.num_experts, B * S, H))
        gate_up = jnp.einsum("enh,ehd->end", hidden_rep, self.gate_up_proj.value)
        gate_up = gate_up + self.gate_up_proj_bias.value[:, None, :]

        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
        gate = jnp.minimum(gate, self.limit)
        up = jnp.clip(up, -self.limit, self.limit)
        glu = gate * jax.nn.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu

        next_states = jnp.einsum("end,edh->enh", gated_output, self.down_proj.value)
        next_states = next_states + self.down_proj_bias.value[:, None, :]

        next_states = next_states.reshape(self.num_experts, B, S, H)
        weights = routing_weights.T.reshape(self.num_experts, B, S)
        final_output = jnp.sum(next_states * weights[..., None], axis=0)
        return final_output


class GptOssTopKRouter(nnx.Module):
    def __init__(self, config: GptOssConfig, *, rngs: nnx.Rngs):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.linear = nnx.Linear(in_features=self.hidden_dim, out_features=self.num_experts, use_bias=True, rngs=rngs)

    def __call__(self, hidden_states):
        B, S, H = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, H)
        router_logits = self.linear(hidden_flat)  # [B*S, E]
        router_top_value, router_indices = jax.lax.top_k(router_logits, k=self.top_k)
        router_top_value = nnx.softmax(router_top_value, axis=-1)

        router_scores = jnp.zeros_like(router_logits)
        token_idx = jnp.arange(B * S)[:, None]
        router_scores = router_scores.at[token_idx, router_indices].set(router_top_value)
        return router_scores, router_indices


class GptOssMLP(nnx.Module):
    def __init__(self, config: GptOssConfig, *, rngs: nnx.Rngs):
        self.router = GptOssTopKRouter(config, rngs=rngs)
        self.experts = GptOssExperts(config)

    def __call__(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores


class GptOssAttention(nnx.Module):
    def __init__(self, config: GptOssConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if config.head_dim is not None else self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nnx.Linear(
            self.hidden_size, self.num_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim, self.hidden_size, use_bias=config.attention_bias, rngs=rngs
        )

        self.sinks = nnx.Param(jnp.zeros((self.num_heads,)))

    def __call__(self, hidden_states, sins, coss, mask=None):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(B, S, self.num_kv_heads, self.head_dim)

        q = apply_rotary_pos_emb(q, sins, coss)
        k = apply_rotary_pos_emb(k, sins, coss)

        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=2)
            v = jnp.repeat(v, self.num_kv_groups, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores = scores + mask

        probs = nnx.softmax(scores, axis=-1)

        output = jnp.matmul(probs, v)
        output = output.transpose(0, 2, 1, 3).reshape(B, S, self.num_heads * self.head_dim)

        return self.o_proj(output)


class GptOssDecoderLayer(nnx.Module):
    def __init__(self, config: GptOssConfig, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GptOssMLP(config, rngs=rngs)

    def __call__(self, hidden_states, sins, coss, mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states, sins, coss, mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GptOssModel(nnx.Module):
    def __init__(self, config: GptOssConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)

        self.layers = nnx.List([GptOssDecoderLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        B, S, _ = hidden_states.shape
        head_dim = (
            self.config.head_dim
            if self.config.head_dim is not None
            else self.config.hidden_size // self.config.num_attention_heads
        )
        sins, coss = create_rope_embeddings(S, head_dim, self.config.rope_theta)

        mask = make_causal_mask(S)
        mask = jnp.where(mask, 0, -1e9)

        for layer in self.layers:
            hidden_states = layer(hidden_states, sins, coss, mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GptOssForCausalLM(nnx.Module):
    def __init__(self, config: GptOssConfig, *, rngs: nnx.Rngs):
        self.model = GptOssModel(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
