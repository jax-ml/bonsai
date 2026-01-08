import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from jax import P
from jax.sharding import AxisType
from tqdm import trange
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import token_type_ids_mask_function

from bonsai.models.gemma3 import modeling, params

# used for skipping smaller tests
SKIP_INTERMEDIATE_TESTS: bool = False

# used to set highest precision on matrix multiplication for testing
jax.config.update("jax_default_matmul_precision", "highest")


def check_hf_token():
    try:
        access_token = os.environ["HF_TOKEN"]
        AutoProcessor.from_pretrained("google/gemma-3-4b-it", token=access_token, use_fast=False)
    except Exception as e:
        print("Failed to access HF_TOKEN or download Processor:")
        print(e)
        return True
    return False


@unittest.skipIf(check_hf_token(), "Skipping TestModuleForwardPasses due to HF_TOKEN failure.")
class TestModuleForwardPasses(absltest.TestCase):
    # Using this for faster testing. This way we can avoid reloading the model.
    # Make sure not to modify the Gemma3 model in inconsistent ways between tests.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_name: str = "google/gemma-3-4b-it"
        cls.torch_device = "cpu"
        access_token = os.environ["HF_TOKEN"]

        # attempt model download
        cls.processor = AutoProcessor.from_pretrained(cls.model_name, token=access_token, use_fast=False)
        cls.torch_model = (
            Gemma3ForConditionalGeneration.from_pretrained(cls.model_name, dtype="auto")
            .to(device=cls.torch_device, dtype=torch.float32)
            .eval()
        )
        cls.torch_config = cls.torch_model.config

        cls.mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)
        cls.bonsai_config = modeling.ModelConfig.gemma3_4b_it(norm_dtype=jnp.float32)
        model_ckpt_path = snapshot_download(cls.model_name, token=access_token)
        cls.bonsai_model = params.create_gemma3_from_pretrained(model_ckpt_path, cls.bonsai_config, mesh=cls.mesh)

    def _upgrade_dtypes(self):
        self.bonsai_model.embed_tokens.weight.embedding.value = (
            self.bonsai_model.embed_tokens.weight.embedding.value.astype(jnp.float32)
        )
        return

    def _make_torch_input(self):
        # returns model inputs:
        # KEY               SHAPE                           DTYPE
        # input_ids         torch.Size([1, 281])            int64
        # attention_mask    torch.Size([1, 281])            int64
        # token_type_ids    torch.Size([1, 281])            int64
        # pixel_values      torch.Size([1, 3, 896, 896])    bfloat16 -> float32
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    },
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ]

        out = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        out["pixel_values"] = out["pixel_values"].to(dtype=torch.float32)

        return {k: v.to(device=self.torch_device) for k, v in out.items()}

    def _make_bonsai_input(self, torch_inputs):
        out = dict()
        for k, v in torch_inputs.items():
            tmp = v.detach().cpu().numpy()
            if k == "pixel_values":
                tmp = np.permute_dims(tmp, (0, 2, 3, 1))
            out[k] = tmp
        return out

    # This should be correct for unbatched inputs
    # Adapted from transformers/models/gemma3/modeling_gemma3.py
    def _process_torch_inputs(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **lm_kwargs,
    ):
        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.torch_config.image_token_id >= self.torch_config.text_config.vocab_size:
            special_image_mask = input_ids == self.torch_config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.torch_model.model.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Merge text and images
        if pixel_values is not None:
            image_features = self.torch_model.model.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.torch_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.torch_config.get_text_config(),
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            is_prefill = (
                not use_cache
                or past_key_values is None
                or not past_key_values.is_initialized
                or pixel_values is not None
            )
            if token_type_ids is not None and is_prefill:
                is_image = (token_type_ids == 1).to(cache_position.device)
                new_image_start = is_image & ~torch.nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
                image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
                image_group_ids = torch.where(
                    is_image, image_group_ids, torch.full_like(token_type_ids, -1, device=is_image.device)
                )
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device), image_group_ids, self.torch_config.mm_tokens_per_image
                )

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        return dict(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

    # This should be correct for unbatched inputs
    # Adapted from transformers/models/gemma3/modeling_gemma3.py
    def _process_torch_inputs_for_decoder_text_model(
        self,
        attn_type,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        training = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not training:
            past_key_values = DynamicCache(config=self.torch_model.model.config.text_config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }
        position_embeddings_global = self.torch_model.model.language_model.rotary_emb(inputs_embeds, position_ids)
        position_embeddings_local = self.torch_model.model.language_model.rotary_emb_local(inputs_embeds, position_ids)
        return dict(
            hidden_states=inputs_embeds,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=causal_mask_mapping[attn_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    # Vision tests
    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_image_emb(self):
        tm = self.torch_model.model.vision_tower.vision_model.embeddings
        nm = self.bonsai_model.vision_tower.embeddings

        t_inputs = self._make_torch_input()
        n_inputs = self._make_bonsai_input(t_inputs)
        tx = t_inputs["pixel_values"]
        nx = n_inputs["pixel_values"]

        with torch.no_grad():
            ty = tm(tx)
        ny = nm(nx)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_siglip_encoder_layer(self):
        tm = self.torch_model.model.vision_tower.vision_model.encoder.layers[0]
        nm = self.bonsai_model.vision_tower.encoder.layers[0]

        tx = torch.randn((1, 4096, 1152), device=self.torch_device)
        nx = tx.detach().cpu().numpy()

        with torch.no_grad():
            ty = tm(tx, None)
        ny = nm(nx, None)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_vision_model(self):
        # only have deviations on .0567% of the entries and on order 7e-3
        tm = self.torch_model.model.vision_tower
        nm = self.bonsai_model.vision_tower

        t_inputs = self._make_torch_input()
        n_inputs = self._make_bonsai_input(t_inputs)
        tx = t_inputs["pixel_values"]
        nx = n_inputs["pixel_values"]

        with torch.no_grad():
            ty = tm(tx).last_hidden_state
        ny = nm(nx)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-2, atol=1e-2)

    # Language tests
    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_text_embedding(self):
        self._upgrade_dtypes()
        tm = self.torch_model.model.language_model.embed_tokens
        nm = self.bonsai_model.embed_tokens

        torch.testing.assert_close(torch.tensor(nm.weight.embedding.value), tm.weight.cpu())
        torch.testing.assert_close(torch.tensor(nm.embed_scale), tm.embed_scale.cpu())

        t_inputs = self._make_torch_input()
        n_inputs = self._make_bonsai_input(t_inputs)
        tx = t_inputs["input_ids"]
        nx = n_inputs["input_ids"]

        with torch.no_grad():
            ty = tm(tx)
        ny = nm(nx)

        np.testing.assert_allclose(ny, ty.detach().cpu().numpy())

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_attn_projs(self):
        tm = self.torch_model.model.language_model.layers[0].self_attn
        nm = self.bonsai_model.language_model.layers[0].self_attn

        tx = torch.randn((1, 281, 2560), device=self.torch_device)
        nx = tx.detach().cpu().numpy()

        ty = tm.q_proj(tx)
        ny = nm.q_proj(nx, out_sharding=P())
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4, err_msg="q")

        ty = tm.k_proj(tx)
        ny = nm.k_proj(nx, out_sharding=P())
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4, err_msg="k")

        ty = tm.v_proj(tx)
        ny = nm.v_proj(nx, out_sharding=P())
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4, err_msg="v")

        tx = torch.randn((1, 281, 2048), device=self.torch_device)
        nx = tx.detach().cpu().numpy()
        ty = tm.o_proj(tx)
        ny = nm.o_proj(nx, out_sharding=P())
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-4, atol=1e-4, err_msg="o")

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_attn_norms(self):
        tm = self.torch_model.model.language_model.layers[0].self_attn
        nm = self.bonsai_model.language_model.layers[0].self_attn

        tx = torch.randn((1, 281, 2048), device=self.torch_device).reshape(1, 281, -1, 256)
        nx = tx.detach().cpu().numpy()

        np.testing.assert_allclose(
            nm.q_norm.scale.value, tm.q_norm.weight.detach().cpu().numpy(), err_msg="q_norm weights"
        )

        ty = tm.q_norm(tx)
        ny = nm.q_norm(nx)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5, err_msg="q")

        tx = torch.randn((1, 281, 1024), device=self.torch_device).reshape(1, 281, -1, 256)
        nx = tx.detach().cpu().numpy()

        ty = tm.k_norm(tx)
        ny = nm.k_norm(nx)
        np.testing.assert_allclose(ny, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5, err_msg="k")

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_sin_cos(self):
        batch_size, seq_len, dim = 2, 10, 256
        hidden_states = torch.ones((batch_size, seq_len, dim))
        jp = jnp.stack([jnp.arange(seq_len), jnp.arange(seq_len)])

        # local uses default
        rt = self.bonsai_config.text_config.rope_slide_theta
        js, jc = modeling._generate_pos_embeddings(jp, dim, rope_theta=rt, factor=1.0)
        rot_emb = self.torch_model.model.language_model.rotary_emb_local
        tc, ts = rot_emb(hidden_states, torch.tensor(jp))
        tc, ts = tc[:, :, : dim // 2], ts[:, :, : dim // 2]
        torch.testing.assert_close(torch.tensor(js), ts)
        torch.testing.assert_close(torch.tensor(jc), tc)

        # global uses linear
        rt = self.bonsai_config.text_config.rope_full_theta
        js, jc = modeling._generate_pos_embeddings(jp, dim, rope_theta=rt, factor=8.0)
        rot_emb = self.torch_model.model.language_model.rotary_emb
        tc, ts = rot_emb(hidden_states, torch.tensor(jp))
        tc, ts = tc[:, :, : dim // 2], ts[:, :, : dim // 2]
        torch.testing.assert_close(torch.tensor(js), ts)
        torch.testing.assert_close(torch.tensor(jc), tc)

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_text_decoder_layers(self):
        first_t_inputs = self._make_torch_input()
        start_t_inputs = self._process_torch_inputs(**first_t_inputs)

        for test_layer in trange(34):
            # Models
            tm = self.torch_model.model.language_model.layers[test_layer]
            nm = self.bonsai_model.language_model.layers[test_layer]
            attn_type = tm.attention_type

            # Inputs
            t_inputs = self._process_torch_inputs_for_decoder_text_model(attn_type, **start_t_inputs)
            nx = t_inputs["hidden_states"].detach().cpu().numpy()
            batch_size, num_tokens, _ = nx.shape
            nnx_cache = modeling.init_cache(
                cfg=self.bonsai_config, batch_size=batch_size, token_len=num_tokens, generate_steps=1, dtype=jnp.float32
            )
            n_tti = first_t_inputs["token_type_ids"].detach().cpu().numpy()

            if attn_type == "full_attention":
                mask = modeling.make_causal_mask(nnx_cache[test_layer], n_tti, out_sharding=P())
            else:
                mask = modeling.make_window_mask(nnx_cache[test_layer], n_tti, 1024, out_sharding=P())

            # run models
            ty = tm(**t_inputs)
            ny = nm(nx, nnx_cache[test_layer], jnp.ones((batch_size, num_tokens)), mask=mask)

            t_inputs["hidden_states"] = ty[0]

            found_exception = False
            try:
                np.testing.assert_allclose(
                    ny, ty[0].detach().cpu().numpy(), rtol=5e-3, atol=5e-3, err_msg=f"{test_layer}"
                )
            except Exception as e:
                print(e)
                found_exception = True
        assert not found_exception, "FOUND EXCEPTION"

    # multi modal tests

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_multi_modal_projector(self):
        t_inputs = self._make_torch_input()
        tm = self.torch_model.model
        nm = self.bonsai_model.multi_modal_projector

        tx = tm.vision_tower(t_inputs["pixel_values"]).last_hidden_state
        nx = tx.detach().cpu().numpy()

        ty = tm.multi_modal_projector(tx)
        ny = nm(nx)

        torch.testing.assert_close(torch.tensor(ny), ty, rtol=1e-4, atol=1e-4)

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_text_image_merge(self):
        nm = self.bonsai_model
        t_inputs = self._make_torch_input()
        t_out = self._process_torch_inputs(**t_inputs)

        # answer is input_embeds
        t_ans = t_out["inputs_embeds"]

        tmp = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        n_text = nm.embed_tokens(tmp)

        # return
        n_img = jnp.array(np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)))
        n_img = nm.vision_tower(n_img)
        n_img = nm.multi_modal_projector(n_img)
        n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())

        n_ans = modeling.batched_merge_modalities(n_img, n_text, n_tti)

        np.testing.assert_allclose(n_ans, t_ans.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)

    @unittest.skipIf(SKIP_INTERMEDIATE_TESTS, "Done")
    def test_masks(self):
        # Make a really long input so we can test the sliding window
        # This only tests for the pre-fill stage
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    },
                    {"type": "text", "text": "Describe this image in detail." + "hello " * 1500},
                ],
            },
        ]

        t_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        t_inputs["pixel_values"] = t_inputs["pixel_values"].to(dtype=torch.float32)

        batch_size, num_tokens = t_inputs["input_ids"].shape
        n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())
        gen_steps = 10
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, gen_steps)
        n_mask = modeling.make_causal_mask(cache[0], n_tti, out_sharding=P())

        # Full attention
        t_inputs = self._process_torch_inputs(**t_inputs)
        t_mask = t_inputs["attention_mask"]["full_attention"]
        size_for_comp = t_mask.shape[-1]

        np.testing.assert_allclose(n_mask[:, :, :, :size_for_comp], t_mask.detach().cpu().numpy())

        # Sliding attention
        t_mask = t_inputs["attention_mask"]["sliding_attention"]
        n_mask = modeling.make_window_mask(
            cache[0], n_tti, self.bonsai_config.text_config.sliding_window, out_sharding=P()
        )

        np.testing.assert_allclose(n_mask[:, :, :, :size_for_comp], t_mask.detach().cpu().numpy())

    @unittest.skip("Skipping - this test is just to observe errors over full model evaluation")
    def test_full_in_order(self):
        tm = self.torch_model.model
        nm = self.bonsai_model

        # Torch inputs
        t_inputs = self._make_torch_input()

        # NNX inputs
        n_img = jnp.array(np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)))
        n_text = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())
        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        # Get masks
        n_causal_mask = modeling.make_causal_mask(cache[0], n_tti, out_sharding=P())
        n_sliding_mask = modeling.make_window_mask(
            cache[0], n_tti, self.bonsai_config.text_config.sliding_window, out_sharding=P()
        )

        # text embeds
        t_inputs_embeds = tm.language_model.embed_tokens(t_inputs["input_ids"])
        n_inputs_embeds = nm.embed_tokens(n_text)
        np.testing.assert_allclose(n_inputs_embeds, t_inputs_embeds.detach().cpu().numpy(), err_msg="text emb")

        # Vision part
        t_vis = tm.vision_tower(t_inputs["pixel_values"]).last_hidden_state
        n_vis = nm.vision_tower(n_img)
        # Mismatched elements: 4608354 / 4718592 (97.7%)
        # Max absolute difference among violations: 0.00756264
        # Max relative difference among violations: 15.521739
        np.testing.assert_allclose(n_vis, t_vis.detach().cpu().numpy(), rtol=1e-3, atol=1e-3, err_msg="vis tower")

        # MM Proj part
        t_img_feat = tm.multi_modal_projector(t_vis)
        n_img_feat = nm.multi_modal_projector(n_vis)
        # Mismatched elements: 648574 / 655360 (99%)
        # Max absolute difference among violations: 0.00063944
        # Max relative difference among violations: 20.392141
        np.testing.assert_allclose(
            n_img_feat, t_img_feat.detach().cpu().numpy(), rtol=1e-3, atol=1e-3, err_msg="mm proj"
        )

        # Merging part
        special_image_mask = tm.get_placeholder_mask(
            t_inputs["input_ids"], inputs_embeds=t_inputs_embeds, image_features=t_img_feat
        )
        t_inputs_embeds = t_inputs_embeds.masked_scatter(special_image_mask, t_img_feat)
        n_inputs_embeds = modeling.batched_merge_modalities(n_img_feat, n_inputs_embeds, n_tti)
        # Mismatched elements: 648574 / 719360 (90.2%)
        # Max absolute difference among violations: 0.00063944
        # Max relative difference among violations: 20.392141
        np.testing.assert_allclose(
            n_inputs_embeds, t_inputs_embeds.detach().cpu().numpy(), rtol=1e-3, atol=1e-3, err_msg="merge"
        )

        # Text part in order
        t_inputs["output_hidden_states"] = True
        t_text_inputs = self._process_torch_inputs(**t_inputs)
        t_hidden_states = tm.language_model(**t_text_inputs).hidden_states
        assert len(t_hidden_states) - 1 == len(nm.language_model.layers), (
            f"{len(t_hidden_states)} vs {len(nm.language_model.layers)}"
        )

        # check inputs
        nx = n_inputs_embeds

        n_hidden_states = []
        for i, layer in enumerate(nm.language_model.layers):
            attn_type = tm.language_model.layers[i].attention_type
            n_mask = n_causal_mask if attn_type == "full_attention" else n_sliding_mask
            n_hidden_states.append(nx)
            nx = layer(nx, cache[i], segment_ids, n_mask)
        nx = nm.language_model.norm(nx)
        n_hidden_states.append(nx)

        for i, (nval, tval) in enumerate(zip(n_hidden_states, t_hidden_states)):
            try:
                np.testing.assert_allclose(nval, tval.detach().cpu().numpy(), err_msg=f"text {i}")
            except Exception as e:
                print(e)
                found_error = True
        assert not found_error, "Found errors in text decoder layers"
        # NOTE: some errors are expected here since errors compound with layer

    def test_full(self):
        tm = self.torch_model
        nm = self.bonsai_model

        t_inputs = self._make_torch_input()

        n_img = jnp.array(np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)))
        n_text = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())
        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        ny = nm(n_text, n_img, cache, segment_ids, n_tti)
        ty = tm(**t_inputs)

        torch.testing.assert_close(torch.tensor(ny), ty.logits, rtol=5e-2, atol=5e-2)

    @unittest.skip("TODO")
    def test_full_batched(self):
        tm = self.torch_model
        nm = self.bonsai_model

        t_inputs = self._make_torch_input()

        n_img = jnp.array(np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)))
        n_text = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
        n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())

        # Test simple batching
        n_img = jnp.concat([n_img, n_img])
        n_text = jnp.concat([n_text, n_text])
        n_tti = jnp.concat([n_tti, n_tti])

        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        ny = nm(n_text, n_img, cache, segment_ids, n_tti)
        ty = tm(**t_inputs)

        torch.testing.assert_close(torch.tensor(ny)[0:1], ty.logits, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(torch.tensor(ny)[1:2], ty.logits, rtol=5e-2, atol=5e-2)

        raise NotImplementedError("Need to get more complex batched inputs working")
        # When doing batching, prompts have >= 0 images (not all same) -> change batched_merge_modalities
        #   for this, we might also need to keep track of where images came from
        # We also need to update the left padding to deal with different padding for each prompt


if __name__ == "__main__":
    absltest.main()
