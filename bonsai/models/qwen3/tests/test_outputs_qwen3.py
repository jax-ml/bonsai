import jax
import jax.numpy as jnp
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from jax.typing import DTypeLike
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3 import Qwen3ForCausalLM

from bonsai.models.qwen3 import modeling, params
from bonsai.models.qwen3.tests.run_model import tokenize


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        model_name: str = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        ## models
        self.torch_model = Qwen3ForCausalLM.from_pretrained(model_name, dtype="auto").eval()
        self.bonsai_config = modeling.ModelCfg.qwen3_0_6b()
        model_ckpt_path = snapshot_download("Qwen/Qwen3-0.6B")
        self.mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"))
        with self.mesh:
            self.nnx_model = params.create_model_from_safe_tensors(model_ckpt_path, self.bonsai_config, self.mesh)

        self.batch_size = 32
        self.num_input_tokens = 5
        self.cache_size, self.gen_steps = 128, 10
        self.relaxed_tol = 1e-3

    def _check_batched_logits(self, left_pads: int, torch_logits: torch.Tensor, nnx_logits: jax.Array):
        """Checks logits from batched inputs"""
        max_len = torch_logits.shape[-2]
        for lp, tl, nl in zip(left_pads, torch_logits, nnx_logits):
            torch.testing.assert_close(
                torch.tensor(nl)[lp:max_len, :], tl[lp:, :], rtol=self.relaxed_tol, atol=self.relaxed_tol
            )

    def _setup_torch_attn(self, input_embeddings: torch.Tensor, attention_mask: None = None):
        """This function replicates the forward method from Qwen3Model in transformers/models/qwen3/modeling_qwen3.py"""
        past_key_values = DynamicCache(config=self.torch_model.config)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + input_embeddings.shape[1], device=self.torch_model.device
        )
        position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.torch_model.config,
                "input_embeds": input_embeddings,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.torch_model.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.torch_model.model.rotary_emb(input_embeddings, position_ids)
        out = dict(
            hidden_states=input_embeddings.to(torch.float32),
            attention_mask=causal_mask_mapping[self.torch_model.model.layers[0].attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return out

    def _nnx_forward_logits(self, cache: modeling.Cache, tokens: jax.Array, dtype: DTypeLike = jnp.float32):
        """Forward pass for the nnx model"""
        segment_ids = 1 * (tokens != self.tokenizer.pad_token_id)
        x = self.nnx_model.embedder.encode(tokens).astype(dtype)
        right_pads = modeling.num_right_pad(segment_ids[0])
        for i, layer in enumerate(self.nnx_model.layers):
            x = layer(x, cache[i], segment_ids, right_pads).astype(dtype)
        nnx_logits = self.nnx_model.lm_head(self.nnx_model.final_norm(x))
        return nnx_logits

    def _process_hf_tokens(self, query: list[str]):
        """Converts queries into tokens for huggingface that is consistent with Bonsai."""
        messages = [{"role": "user", "content": s} for s in query]
        text = [
            self.tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for m in messages
        ]
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True, padding_side="left").to(
            self.torch_model.device
        )
        tmp = model_inputs["attention_mask"]
        num_zeros = tmp.shape[1] - tmp.sum(dim=-1)
        model_inputs["left_pads"] = num_zeros
        pos_ids = torch.arange(tmp.shape[1]) - num_zeros.reshape(-1, 1)
        pos_ids[pos_ids < 0] = 2**30
        model_inputs["position_ids"] = pos_ids
        return model_inputs

    def _init_nnx_cache(self, batch_size: int):
        with self.mesh:
            return modeling.init_cache(
                num_layers=self.bonsai_config.num_layers,
                batch_size=batch_size,
                cache_size=self.cache_size,
                num_kv_heads=self.bonsai_config.num_kv_heads,
                head_dim=self.bonsai_config.head_dim,
                dtype=jnp.float32,
                sharding=self.bonsai_config.shd_cfg.act_btnh,
            )

    def test_embedder(self):
        nm = self.nnx_model.embedder
        tm = self.torch_model.model.embed_tokens

        tx = torch.randint(0, self.torch_model.config.vocab_size, size=(self.batch_size, self.num_input_tokens))
        jx = tx.cpu().detach().numpy()

        jy, ty = nm.encode(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_decoder_layer(self):
        nm = self.nnx_model.layers[0]
        tm = self.torch_model.model.layers[0].to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(jx)
        nnx_cache = self._init_nnx_cache(self.batch_size)
        torch_inputs = self._setup_torch_attn(tx)

        jy, ty = nm(jx, nnx_cache[0], jnp.ones((self.batch_size, self.num_input_tokens)), 0), tm(**torch_inputs)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_all_decoder_layers(self):
        nnx_cache = self._init_nnx_cache(self.batch_size)
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)

        for nm, tm, nc in zip(self.nnx_model.layers, self.torch_model.model.layers, nnx_cache):
            jx = jax.random.normal(jax.random.key(0), shape=shape)
            tx = torch.tensor(jx)

            jy = nm(jx, nc, jnp.ones((self.batch_size, self.num_input_tokens)), 0)
            torch_inputs = self._setup_torch_attn(tx)
            ty = tm.to(torch.float32)(**torch_inputs)
            torch.testing.assert_close(torch.tensor(jy), ty, atol=self.relaxed_tol, rtol=self.relaxed_tol)

    def test_rms_norm(self):
        nm = self.nnx_model.layers[0].input_layernorm
        tm = self.torch_model.model.layers[0].input_layernorm

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape, dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_self_attn(self):
        nm = self.nnx_model.layers[0].attn
        tm = self.torch_model.model.layers[0].self_attn.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(jx)
        torch_inputs = self._setup_torch_attn(tx)
        nnx_cache = self._init_nnx_cache(self.batch_size)

        jy = nm(jx, nnx_cache[0], jnp.ones((self.batch_size, self.num_input_tokens), dtype=jnp.float32), 0)
        ty = tm(**torch_inputs)[0]
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_q_norm(self):
        nm = self.nnx_model.layers[0].attn.q_norm
        tm = self.torch_model.model.layers[0].self_attn.q_norm

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.num_heads, self.bonsai_config.head_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape, dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_k_norm(self):
        nm = self.nnx_model.layers[0].attn.q_norm
        tm = self.torch_model.model.layers[0].self_attn.q_norm

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.num_kv_heads, self.bonsai_config.head_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape, dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_q_proj(self):
        nm = self.nnx_model.layers[0].attn.q_proj
        tm = self.torch_model.model.layers[0].self_attn.q_proj

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape, dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.num_heads, self.bonsai_config.head_dim)
        jy, ty = nm(jx), tm(tx).reshape(shape)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_k_proj(self):
        nm = self.nnx_model.layers[0].attn.k_proj
        tm = self.torch_model.model.layers[0].self_attn.k_proj

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape, dtype=jnp.bfloat16)
        tx = torch.tensor(jx)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.num_kv_heads, self.bonsai_config.head_dim)
        jy, ty = nm(jx), tm(tx).reshape(shape)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_o_proj(self):
        nm = self.nnx_model.layers[0].attn.o_proj
        tm = self.torch_model.model.layers[0].self_attn.o_proj

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.num_heads, self.bonsai_config.head_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape, dtype=jnp.bfloat16)
        tx = torch.tensor(jx).reshape(self.batch_size, self.num_input_tokens, -1)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_mlp(self):
        nm = self.nnx_model.layers[0].mlp
        tm = self.torch_model.model.layers[0].mlp.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty, rtol=self.relaxed_tol, atol=self.relaxed_tol)

    def test_lm_head(self):
        nm = self.nnx_model.lm_head
        tm = self.torch_model.lm_head.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(jx)

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(torch.tensor(jy), ty)

    def test_sin_cos(self):
        batch_size, seq_len, dim = 2, 10, 128
        hidden_states = torch.ones((batch_size, seq_len, dim))
        jp = jnp.stack([jnp.arange(seq_len), jnp.arange(seq_len)])
        js, jc = modeling._generate_pos_embeddings(jp, dim)
        tc, ts = self.torch_model.model.rotary_emb(hidden_states, torch.tensor(jp))
        tc, ts = tc[:, :, : dim // 2], ts[:, :, : dim // 2]
        torch.testing.assert_close(torch.tensor(js), ts)
        torch.testing.assert_close(torch.tensor(jc), tc)

    def test_full(self):
        query = ["Why is the sky blue instead of any other color like purple?"]
        tokens, max_len = tokenize(self.tokenizer, query)
        self.torch_model = self.torch_model.to(torch.float32)
        nnx_cache = self._init_nnx_cache(len(query))

        nnx_logits = self._nnx_forward_logits(nnx_cache, tokens, jnp.float32)
        torch_inputs = self._process_hf_tokens(query)
        torch_logits = self.torch_model(**torch_inputs).logits
        torch.testing.assert_close(
            torch.tensor(nnx_logits)[:, :max_len, :], torch_logits, rtol=self.relaxed_tol, atol=self.relaxed_tol
        )

    def test_full_batched(self):
        query = ["Why is the sky blue instead of any other color like purple?", "Who am I?"]
        tokens, _ = tokenize(self.tokenizer, query)
        self.torch_model = self.torch_model.to(torch.float32)
        nnx_cache = self._init_nnx_cache(len(query))

        nnx_logits = self._nnx_forward_logits(nnx_cache, tokens, jnp.float32)
        torch_inputs = self._process_hf_tokens(query)
        torch_logits = self.torch_model(**torch_inputs).logits

        self._check_batched_logits(torch_inputs["left_pads"], torch_logits, nnx_logits)


if __name__ == "__main__":
    absltest.main()
