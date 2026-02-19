import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import AxisType
from jax.typing import DTypeLike
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2 import Qwen2ForCausalLM

from bonsai.models.qwen2 import modeling, params
from bonsai.models.qwen2.tests.run_model import tokenize


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        model_name: str = "Qwen/Qwen2-0.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.torch_model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
        self.bonsai_config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
        model_ckpt_path = snapshot_download("Qwen/Qwen2-0.5B")
        self.mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(self.mesh)

        graph_def, state = nnx.split(
            params.create_model_from_safe_tensors(model_ckpt_path, self.bonsai_config, self.mesh)
        )
        state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, state)
        self.nnx_model = nnx.merge(graph_def, state)

        self.batch_size = 32
        self.num_input_tokens = 5
        self.cache_size, self.gen_steps = 128, 10
        self.relaxed_tol = 1e-3

    def _check_batched_logits(self, left_pads: int, torch_logits: torch.Tensor, nnx_logits: jax.Array):
        max_len = torch_logits.shape[-2]
        for lp, tl, nl in zip(left_pads, torch_logits, nnx_logits):
            torch.testing.assert_close(
                torch.tensor(np.array(nl, dtype=np.float32))[lp:max_len, :],
                tl[lp:, :],
                rtol=self.relaxed_tol,
                atol=self.relaxed_tol,
                check_dtype=False,
            )

    def _setup_torch_attn(self, input_embeddings: torch.Tensor, attention_mask: None = None):
        past_key_values = DynamicCache()
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + input_embeddings.shape[1], device=self.torch_model.device
        )
        position_ids = cache_position.unsqueeze(0)

        batch_size, seq_length = input_embeddings.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_embeddings.device)

        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_embeddings.device), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
        causal_mask = torch.where(causal_mask, torch.finfo(input_embeddings.dtype).min, 0.0)

        position_embeddings = self.torch_model.model.rotary_emb(input_embeddings, position_ids)
        out = dict(
            hidden_states=input_embeddings.to(torch.float32),
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return out

    def _nnx_forward_logits(self, cache: modeling.Cache, tokens: jax.Array, dtype: DTypeLike = jnp.float32):
        segment_ids = 1 * (tokens != self.tokenizer.pad_token_id)
        x = self.nnx_model.embedder.embedding.value.at[(tokens,)].get().astype(jnp.float32)
        for i, layer in enumerate(self.nnx_model.layers):
            x = layer(x, cache[i], segment_ids).astype(dtype)
        nnx_logits = self.nnx_model.lm_head(self.nnx_model.final_norm(x))
        return nnx_logits

    def _process_hf_tokens(self, query: list[str]):
        messages = [{"role": "user", "content": s} for s in query]
        text = [self.tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
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
        return self.nnx_model.init_cache(
            cfg=self.bonsai_config, batch_size=batch_size, token_len=10, generate_steps=32, dtype=jnp.float32
        )

    def test_embedder(self):
        nm = self.nnx_model.embedder
        tm = self.torch_model.model.embed_tokens

        tx = torch.randint(0, self.torch_model.config.vocab_size, size=(self.batch_size, self.num_input_tokens))
        jx = jnp.array(tx.cpu().detach().numpy())

        jy, ty = nm.embedding.value.at[(jx,)].get(), tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_decoder_layer(self):
        nm = self.nnx_model.layers[0]
        tm = self.torch_model.model.layers[0].to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        nnx_cache = self._init_nnx_cache(self.batch_size)
        torch_inputs = self._setup_torch_attn(tx)

        jy, ty = nm(jx, nnx_cache[0], jnp.ones((self.batch_size, self.num_input_tokens))), tm(**torch_inputs)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_all_decoder_layers(self):
        nnx_cache = self._init_nnx_cache(self.batch_size)
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)

        for nm, tm, nc in zip(self.nnx_model.layers, self.torch_model.model.layers, nnx_cache):
            jx = jax.random.normal(jax.random.key(0), shape=shape)
            tx = torch.tensor(np.array(jx, dtype=np.float32))

            jy = nm(jx, nc, jnp.ones((self.batch_size, self.num_input_tokens)))
            torch_inputs = self._setup_torch_attn(tx)
            ty = tm.to(torch.float32)(**torch_inputs)
            torch.testing.assert_close(
                torch.tensor(np.array(jy, dtype=np.float32)),
                ty,
                atol=self.relaxed_tol,
                rtol=self.relaxed_tol,
                check_dtype=False,
            )

    def test_rms_norm(self):
        nm = self.nnx_model.layers[0].input_layernorm
        tm = self.torch_model.model.layers[0].input_layernorm

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_self_attn(self):
        nm = self.nnx_model.layers[0].attn
        tm = self.torch_model.model.layers[0].self_attn.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        torch_inputs = self._setup_torch_attn(tx)
        nnx_cache = self._init_nnx_cache(self.batch_size)

        jy = nm(jx, nnx_cache[0], jnp.ones((self.batch_size, self.num_input_tokens), dtype=jnp.float32))
        ty = tm(**torch_inputs)[0]
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_q_proj(self):
        nm = self.nnx_model.layers[0].attn.q_proj
        tm = self.torch_model.model.layers[0].self_attn.q_proj.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        jy = nm(jx)
        ty = tm(tx)

        jy_flat = jy.reshape(self.batch_size, self.num_input_tokens, -1)
        torch.testing.assert_close(
            torch.tensor(np.array(jy_flat, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_k_proj(self):
        nm = self.nnx_model.layers[0].attn.k_proj
        tm = self.torch_model.model.layers[0].self_attn.k_proj.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        jy = nm(jx)
        ty = tm(tx)

        jy_flat = jy.reshape(self.batch_size, self.num_input_tokens, -1)
        torch.testing.assert_close(
            torch.tensor(np.array(jy_flat, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_o_proj(self):
        nm = self.nnx_model.layers[0].attn.o_proj
        tm = self.torch_model.model.layers[0].self_attn.o_proj.to(torch.float32)

        input_dim = self.bonsai_config.num_heads * self.bonsai_config.head_dim
        shape = (self.batch_size, self.num_input_tokens, input_dim)

        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        jy = nm(jx)
        ty = tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_mlp(self):
        nm = self.nnx_model.layers[0].mlp
        tm = self.torch_model.model.layers[0].mlp.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_lm_head(self):
        nm = self.nnx_model.lm_head
        tm = self.torch_model.lm_head.to(torch.float32)

        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.emb_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        jy, ty = nm(jx), tm(tx)
        torch.testing.assert_close(
            torch.tensor(np.array(jy, dtype=np.float32)),
            ty,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_sin_cos(self):
        batch_size, seq_len = 2, 10
        dim = self.bonsai_config.head_dim
        hidden_states = torch.ones((batch_size, seq_len, dim))
        jp = jnp.stack([jnp.arange(seq_len), jnp.arange(seq_len)])
        js, jc = modeling._generate_pos_embeddings(jp, dim, self.bonsai_config.rope_theta)

        position_ids = torch.tensor(np.array(jp, dtype=np.int64))
        torch_rope_output = self.torch_model.model.rotary_emb(hidden_states, position_ids)

        if isinstance(torch_rope_output, tuple) and len(torch_rope_output) == 2:
            tc, ts = torch_rope_output
        else:
            tc = torch_rope_output
            ts = torch_rope_output

        half_dim = dim // 2
        if tc.shape[-1] != half_dim:
            tc = tc[:, :, :half_dim]
        if ts.shape[-1] != half_dim:
            ts = ts[:, :, :half_dim]

        torch.testing.assert_close(
            torch.tensor(np.array(js, dtype=np.float32)),
            ts,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch.tensor(np.array(jc, dtype=np.float32)),
            tc,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_full(self):
        query = ["Why is the sky blue instead of any other color like purple?"]
        tokens = tokenize(self.tokenizer, query)
        _, token_len = tokens.shape
        self.torch_model = self.torch_model.to(torch.float32)
        nnx_cache = self._init_nnx_cache(len(query))

        nnx_logits = self._nnx_forward_logits(nnx_cache, tokens, jnp.float32)
        torch_inputs = self._process_hf_tokens(query)
        torch_logits = self.torch_model(**torch_inputs).logits
        torch.testing.assert_close(
            torch.tensor(np.array(nnx_logits, dtype=np.float32))[:, :token_len, :],
            torch_logits,
            rtol=self.relaxed_tol,
            atol=self.relaxed_tol,
            check_dtype=False,
        )

    def test_full_batched(self):
        query = ["Why is the sky blue instead of any other color like purple?", "Who am I?"]
        tokens = tokenize(self.tokenizer, query)
        self.torch_model = self.torch_model.to(torch.float32)
        nnx_cache = self._init_nnx_cache(len(query))

        nnx_logits = self._nnx_forward_logits(nnx_cache, tokens, jnp.float32)
        torch_inputs = self._process_hf_tokens(query)
        torch_logits = self.torch_model(**torch_inputs).logits

        self._check_batched_logits(torch_inputs["left_pads"], torch_logits, nnx_logits)


if __name__ == "__main__":
    absltest.main()
