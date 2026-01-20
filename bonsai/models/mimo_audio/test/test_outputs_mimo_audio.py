import json
import os
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from bonsai.models.mimo_audio import params
from bonsai.models.mimo_audio.mimo_audio_configuration import MiMoAudioConfig, MiMoAudioArguments

# Since the transformer library does not yet support mimo-audio-tokenizer,
# during testing, the official implementation code of mimo-audio-tokenizer (https://github.com/XiaomiMiMo/MiMo-Audio) needs to be copied to the corresponding location.
from bonsai.models.mimo_audio.pytorch.src.mimo_audio.modeling_mimo_audio import (
    MiMoAudioForCausalLM as TorchMiMoAudio,
    MiMoAudioConfig as TorchMiMoAudioConfig,
)
from bonsai.models.qwen3.modeling import ShardingCfg


class TestMiMoAudioLayerOutputs(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        jax.config.update("jax_default_matmul_precision", "float32")
        jax.config.update("jax_platforms", "cpu")

        model_name = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_ckpt_path = snapshot_download(model_name)

        config_path = os.path.join(model_ckpt_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        config_kwargs = {k: v for k, v in config_dict.items() if k in MiMoAudioConfig.__dataclass_fields__}
        config_kwargs["shd_cfg"] = ShardingCfg.no_sharding()
        cls.bonsai_config = MiMoAudioConfig(**config_kwargs)

        cls.args = MiMoAudioArguments(
            model_name_or_path=model_ckpt_path,
            sosp_idx=cls.tokenizer.convert_tokens_to_ids("<|sosp|>"),
            eosp_idx=cls.tokenizer.convert_tokens_to_ids("<|eosp|>"),
            sostm_idx=cls.tokenizer.convert_tokens_to_ids("<|sostm|>"),
            eostm_idx=cls.tokenizer.convert_tokens_to_ids("<|eostm|>"),
            eot_idx=cls.tokenizer.convert_tokens_to_ids("<|eot|>"),
            empty_idx=cls.tokenizer.convert_tokens_to_ids("<|empty|>"),
        )

        torch_config = TorchMiMoAudioConfig.from_pretrained(model_ckpt_path)
        cls.torch_model = (
            TorchMiMoAudio.from_pretrained(
                model_ckpt_path, config=torch_config, args=asdict(cls.args), torch_dtype=torch.float32
            )
            .eval()
            .cpu()
        )

        cls.nnx_model = params.create_model_with_weights(
            model_path=model_ckpt_path,
            config=cls.bonsai_config,
            args=cls.args,
            rngs=nnx.Rngs(0),
            dtype=jnp.float32,
            mesh=None,
        )

        cls.batch_size = 1
        cls.num_input_tokens = 5
        cls.group_size = cls.bonsai_config.group_size
        cls.audio_channels = cls.bonsai_config.audio_channels
        cls.tol = 1e-3

    def _init_cache(self, batch_size, token_len):
        return self.nnx_model.model.init_cache(
            cfg=self.nnx_model.qwen2_config,
            batch_size=batch_size,
            token_len=token_len,
            generate_steps=0,
            dtype=jnp.float32,
        )

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

    def test_text_embedder(self):
        tx = torch.randint(0, self.torch_model.config.vocab_size, size=(self.batch_size, self.num_input_tokens))
        jx = jnp.array(tx.cpu().detach().numpy())
        jy = self.nnx_model.model.embedder.embedding.value[jx]
        with torch.no_grad():
            ty = self.torch_model.model.embed_tokens(tx)
        self._compare(jy, ty)

    def test_speech_embeddings(self):
        for ch in range(self.audio_channels):
            vocab_size = self.nnx_model.speech_vocab_sizes[ch]
            tx = torch.randint(0, vocab_size, size=(self.batch_size, self.num_input_tokens))
            jx = jnp.array(tx.cpu().detach().numpy())
            jy = self.nnx_model.speech_embeddings[ch](jx)
            with torch.no_grad():
                ty = self.torch_model.speech_embeddings[ch](tx)
            self._compare(jy, ty)

    def test_main_decoder_layer(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.hidden_size)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        cache = self._init_cache(self.batch_size, self.num_input_tokens)
        segment_ids = jnp.ones((self.batch_size, self.num_input_tokens))
        jy = self.nnx_model.model.layers[0](jx, cache[0], segment_ids)

        cache_position = torch.arange(0, self.num_input_tokens, device=tx.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.torch_model.model.rotary_emb(tx, position_ids)

        with torch.no_grad():
            ty_output = self.torch_model.model.layers[0].to(torch.float32)(
                tx,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=DynamicCache(),
                cache_position=cache_position,
            )
            ty = ty_output[0] if isinstance(ty_output, tuple) else ty_output

        self._compare(jy, ty)

    def test_all_main_decoder_layers(self):
        cache = self._init_cache(self.batch_size, self.num_input_tokens)
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.hidden_size)

        for layer_idx, (nm, tm, nc) in enumerate(
            zip(self.nnx_model.model.layers, self.torch_model.model.layers, cache)
        ):
            jx = jax.random.normal(jax.random.key(layer_idx), shape=shape)
            tx = torch.tensor(np.array(jx, dtype=np.float32))
            segment_ids = jnp.ones((self.batch_size, self.num_input_tokens))
            jy = nm(jx, nc, segment_ids)

            cache_position = torch.arange(0, self.num_input_tokens, device=tx.device)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = self.torch_model.model.rotary_emb(tx, position_ids)

            with torch.no_grad():
                ty_output = tm.to(torch.float32)(
                    tx,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=DynamicCache(),
                    cache_position=cache_position,
                )
                ty = ty_output[0] if isinstance(ty_output, tuple) else ty_output

            self._compare(jy, ty)

    def test_main_rms_norm(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.hidden_size)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        jy = self.nnx_model.model.layers[0].input_layernorm(jx)
        with torch.no_grad():
            ty = self.torch_model.model.layers[0].input_layernorm(tx)
        self._compare(jy, ty)

    def test_main_self_attn(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.hidden_size)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        cache = self._init_cache(self.batch_size, self.num_input_tokens)
        segment_ids = jnp.ones((self.batch_size, self.num_input_tokens), dtype=jnp.float32)
        jy = self.nnx_model.model.layers[0].attn(jx, cache[0], segment_ids)

        cache_position = torch.arange(0, self.num_input_tokens, device=tx.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.torch_model.model.rotary_emb(tx, position_ids)
        attention_mask = create_causal_mask(
            config=self.torch_model.config,
            input_embeds=tx,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=DynamicCache(),
            position_ids=position_ids,
        )

        with torch.no_grad():
            ty = self.torch_model.model.layers[0].self_attn.to(torch.float32)(
                tx,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=DynamicCache(),
                cache_position=cache_position,
            )[0]
        self._compare(jy, ty)

    def test_main_mlp(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.hidden_size)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        jy = self.nnx_model.model.layers[0].mlp(jx)
        with torch.no_grad():
            ty = self.torch_model.model.layers[0].mlp.to(torch.float32)(tx)
        self._compare(jy, ty)

    def test_lm_head(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.hidden_size)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        jy = self.nnx_model.lm_head(jx)
        with torch.no_grad():
            ty = self.torch_model.lm_head.to(torch.float32)(tx)
        self._compare(jy, ty)

    def test_local_transformer_lm_heads(self):
        for ch in range(self.audio_channels):
            shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.local_dim)
            jx = jax.random.normal(jax.random.key(ch), shape=shape)
            tx = torch.tensor(np.array(jx, dtype=np.float32))
            jy = self.nnx_model.local_transformer_lm_heads[ch](jx)
            with torch.no_grad():
                ty = self.torch_model.local_transformer_lm_heads[ch].to(torch.float32)(tx)
            self._compare(jy, ty)

    def test_speech_group_downcast(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.input_local_dim * self.group_size)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        jy = self.nnx_model.speech_group_downcast(jx)
        with torch.no_grad():
            ty = self.torch_model.speech_group_downcast.to(torch.float32)(tx)
        self._compare(jy, ty)

    def test_hidden_states_downcast(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.hidden_size)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        jy = self.nnx_model.hidden_states_downcast(jx)
        with torch.no_grad():
            ty = self.torch_model.hidden_states_downcast.to(torch.float32)(tx)
        self._compare(jy, ty)

    def test_local_transformer_layer(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.local_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        cache = self.nnx_model.local_transformer.init_cache(
            cfg=self.nnx_model.local_qwen2_config,
            batch_size=self.batch_size,
            token_len=self.num_input_tokens,
            generate_steps=0,
            dtype=jnp.float32,
        )
        segment_ids = jnp.ones((self.batch_size, self.num_input_tokens))
        jy = self.nnx_model.local_transformer.layers[0](jx, cache[0], segment_ids)

        cache_position = torch.arange(0, self.num_input_tokens, device=tx.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.torch_model.local_transformer.rotary_emb(tx, position_ids)

        with torch.no_grad():
            ty_output = self.torch_model.local_transformer.layers[0].to(torch.float32)(
                tx,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=DynamicCache(),
                cache_position=cache_position,
            )
            ty = ty_output[0] if isinstance(ty_output, tuple) else ty_output

        self._compare(jy, ty)

    def test_input_local_transformer_layer(self):
        shape = (self.batch_size, self.num_input_tokens, self.bonsai_config.input_local_dim)
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))

        cache = self.nnx_model.input_local_transformer.init_cache(
            cfg=self.nnx_model.input_local_qwen2_config,
            batch_size=self.batch_size,
            token_len=self.num_input_tokens,
            generate_steps=0,
            dtype=jnp.float32,
        )
        segment_ids = jnp.ones((self.batch_size, self.num_input_tokens))
        jy = self.nnx_model.input_local_transformer.layers[0](jx, cache[0], segment_ids)

        cache_position = torch.arange(0, self.num_input_tokens, device=tx.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.torch_model.input_local_transformer.rotary_emb(tx, position_ids)
        attention_mask = torch.ones(
            (self.batch_size, 1, self.num_input_tokens, self.num_input_tokens), dtype=torch.float32, device=tx.device
        )

        with torch.no_grad():
            ty_output = self.torch_model.input_local_transformer.layers[0].to(torch.float32)(
                tx,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=DynamicCache(),
                cache_position=cache_position,
            )
            ty = ty_output[0] if isinstance(ty_output, tuple) else ty_output

        self._compare(jy, ty)

    def test_apply_input_local_transformer(self):
        shape = (
            self.batch_size,
            self.num_input_tokens // self.group_size,
            self.group_size,
            self.bonsai_config.input_local_dim,
        )
        jx = jax.random.normal(jax.random.key(0), shape=shape)
        tx = torch.tensor(np.array(jx, dtype=np.float32))
        jy = self.nnx_model.apply_input_local_transformer(jx, cache=None)
        with torch.no_grad():
            ty = self.torch_model.apply_input_local_transformer(tx)
        self._compare(jy, ty)

    def test_prepare_input_embeds(self):
        num_groups = 3
        input_shape = (self.batch_size, self.audio_channels + 1, num_groups * self.group_size)
        input_ids_np = np.random.randint(0, 1000, input_shape, dtype=np.int32)
        for ch in range(self.audio_channels):
            input_ids_np[:, ch + 1, :] = np.random.randint(
                0, self.nnx_model.speech_vocab_sizes[ch], (self.batch_size, num_groups * self.group_size)
            )

        input_ids_jax = jnp.array(input_ids_np)
        input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long)

        def text_embed_fn_jax(x):
            return self.nnx_model.model.embedder.embedding.value[x]

        jax_embeds = self.nnx_model._prepare_input_embeds(input_ids_jax, text_embed_fn_jax)

        with torch.no_grad():
            torch_embeds = self.torch_model._prepare_input_embeds(input_ids_torch)

        self._compare(jax_embeds, torch_embeds)

    def test_full_forward(self):
        num_groups = 3
        input_shape = (self.batch_size, self.audio_channels + 1, num_groups * self.group_size)
        input_ids_np = np.random.randint(0, 1000, input_shape, dtype=np.int32)
        for ch in range(self.audio_channels):
            input_ids_np[:, ch + 1, :] = np.random.randint(
                0, self.nnx_model.speech_vocab_sizes[ch], (self.batch_size, num_groups * self.group_size)
            )

        input_ids_jax = jnp.array(input_ids_np)
        input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long)

        cache_jax = self._init_cache(self.batch_size, num_groups)
        text_logits_jax, local_hidden_jax, _ = self.nnx_model.forward(input_ids_jax, cache_jax)

        with torch.no_grad():
            attention_mask = torch.ones((self.batch_size, num_groups), dtype=torch.bool, device=input_ids_torch.device)
            position_ids = torch.arange(num_groups).unsqueeze(0).expand(self.batch_size, -1)
            cache_position = torch.arange(num_groups)
            outputs_torch = self.torch_model(
                input_ids=input_ids_torch,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            text_logits_torch = outputs_torch.text_logits
            local_hidden_torch = outputs_torch.local_hidden_states

        self._compare(text_logits_jax, text_logits_torch)
        self._compare(local_hidden_jax, local_hidden_torch)


if __name__ == "__main__":
    absltest.main()
