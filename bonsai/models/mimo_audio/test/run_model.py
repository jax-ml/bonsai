#!/usr/bin/env python3

import os
import json
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download


def load_audio_tokenizer(tokenizer_path: str):
    from bonsai.models.mimo_audio.mimo_audio_tokenizer_configuration import MiMoAudioTokenizerConfig
    from bonsai.models.mimo_audio.mimo_audio_tokenizer_params import load_tokenizer_weights_from_safetensors

    config_path = os.path.join(tokenizer_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict["use_sharding"] = False
    config = MiMoAudioTokenizerConfig(**config_dict)
    safetensors_path = os.path.join(tokenizer_path, "model.safetensors")

    tokenizer_model = load_tokenizer_weights_from_safetensors(
        config=config,
        safetensors_path=safetensors_path,
        dtype=jnp.float32,
        mesh=None,
        rngs=nnx.Rngs(0),
    )

    return tokenizer_model, config


def load_main_model(model_path: str):
    from bonsai.models.mimo_audio.mimo_audio_configuration import MiMoAudioConfig, MiMoAudioArguments
    from bonsai.models.mimo_audio.params import create_model_with_weights
    from transformers import AutoTokenizer

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_kwargs = {k: v for k, v in config_dict.items() if k in MiMoAudioConfig.__dataclass_fields__}
    config = MiMoAudioConfig(**config_kwargs)

    text_tokenizer = AutoTokenizer.from_pretrained(model_path)

    args = MiMoAudioArguments(
        model_name_or_path=model_path,
        sosp_idx=text_tokenizer.convert_tokens_to_ids("<|sosp|>"),
        eosp_idx=text_tokenizer.convert_tokens_to_ids("<|eosp|>"),
        sostm_idx=text_tokenizer.convert_tokens_to_ids("<|sostm|>"),
        eostm_idx=text_tokenizer.convert_tokens_to_ids("<|eostm|>"),
        eot_idx=text_tokenizer.convert_tokens_to_ids("<|eot|>"),
        empty_idx=text_tokenizer.convert_tokens_to_ids("<|empty|>"),
    )
    model = create_model_with_weights(
        model_path=model_path,
        config=config,
        args=args,
        rngs=nnx.Rngs(0),
        mesh=None,
    )

    return model, config, args, text_tokenizer


def insert_between(tokens: list, group_size: int, fill_value: int) -> list:
    if group_size <= 1:
        return tokens

    result = []
    for token in tokens:
        result.append(token)
        result.extend([fill_value] * (group_size - 1))

    return result


def run_inference(
    main_model,
    tokenizer_model,
    text_tokenizer,
    config,
    args,
    tokenizer_config,
    text_to_speak: str,
    max_steps: int = 100,
    output_dir: str = "test_outputs",
):
    from bonsai.models.mimo_audio.modeling import forward_jit, MiMoSampler
    from bonsai.models.mimo_audio.mimo_audio_configuration import MiMoSamplerConfig

    audio_channels = main_model.audio_channels
    group_size = main_model.group_size
    batch_size = 1

    tts_template = "Turn this writing into audio"
    chat_text = f"<|im_start|>user\n{tts_template}: {text_to_speak}<|im_end|>\n<|im_start|>assistant\n<|sostm|>"

    text_tokens_raw = text_tokenizer.encode(chat_text)
    text_tokens_with_spacing = insert_between(text_tokens_raw, group_size, -100)

    num_groups = len(text_tokens_with_spacing) // group_size
    if len(text_tokens_with_spacing) % group_size != 0:
        text_tokens_with_spacing.extend([-100] * (group_size - len(text_tokens_with_spacing) % group_size))
        num_groups = len(text_tokens_with_spacing) // group_size

    input_shape = (batch_size, audio_channels + 1, num_groups * group_size)
    input_ids = jnp.zeros(input_shape, dtype=jnp.int32)

    input_ids = input_ids.at[0, 0, :].set(jnp.array(text_tokens_with_spacing))

    for ch in range(1, audio_channels + 1):
        channel_empty_id = main_model.speech_empty_ids[ch - 1]
        audio_empty_tokens = jnp.full((num_groups * group_size,), channel_empty_id, dtype=jnp.int32)
        input_ids = input_ids.at[0, ch, :].set(audio_empty_tokens)

    cache = main_model.model.init_cache(
        main_model.qwen2_config,
        batch_size,
        num_groups,
        generate_steps=max_steps,
        dtype=jnp.bfloat16,
    )

    text_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.6, top_p=1.0, do_sample=True))
    audio_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.9, top_p=0.95, do_sample=True))

    pad_id = text_tokenizer.pad_token_id
    text_logits, local_hidden_states, cache = forward_jit(main_model, input_ids, cache, pad_id)

    generated_text_tokens = []
    generated_audio_tokens_list = []

    rng_key = jax.random.key(42)
    empty_idx = args.empty_idx

    for step in range(max_steps):
        key, subkey = jax.random.split(rng_key)
        logits_2d = text_logits[0, 0:1, :]
        next_text_token = text_sampler.sample(logits_2d, subkey)
        next_text_token_int = int(next_text_token[0])
        generated_text_tokens.append(next_text_token_int)

        if next_text_token_int == args.eostm_idx:
            break
        if next_text_token_int == text_tokenizer.eos_token_id:
            break

        audio_tokens = None

        if next_text_token_int != empty_idx:
            for t in range(group_size):
                audio_tokens_step = jnp.array(main_model.speech_empty_ids)
                generated_audio_tokens_list.append(audio_tokens_step)
        else:
            key, subkey = jax.random.split(key)
            audio_tokens = main_model.local_forward(local_hidden_states, subkey, audio_sampler)

            for t in range(group_size):
                audio_tokens_step = audio_tokens[0, t, :]
                generated_audio_tokens_list.append(audio_tokens_step)

        rng_key = key

        next_input = jnp.zeros((batch_size, audio_channels + 1, group_size), dtype=jnp.int32)

        for i in range(group_size):
            next_input = next_input.at[0, 0, i].set(next_text_token[0])

        if audio_tokens is None:
            for ch in range(audio_channels):
                channel_empty_id = main_model.speech_empty_ids[ch]
                for i in range(group_size):
                    next_input = next_input.at[0, ch + 1, i].set(channel_empty_id)
        else:
            for ch in range(audio_channels):
                for i in range(group_size):
                    next_input = next_input.at[0, ch + 1, i].set(audio_tokens[0, i, ch])

        text_logits, local_hidden_states, cache = forward_jit(main_model, next_input, cache, pad_id)

    generated_text = text_tokenizer.decode(generated_text_tokens, skip_special_tokens=True)
    print(f"text token output: {generated_text}")

    audio_tokens_array = jnp.stack(generated_audio_tokens_list, axis=0).T

    speech_empty_ids = main_model.speech_empty_ids
    is_real_audio_mask = jnp.zeros(audio_tokens_array.shape[1], dtype=bool)

    for ch in range(audio_channels):
        empty_id = speech_empty_ids[ch]
        not_empty = audio_tokens_array[ch, :] != empty_id
        is_real_audio_mask = is_real_audio_mask | not_empty

    audio_tokens_array = audio_tokens_array[:, is_real_audio_mask]
    decoded_audio = tokenizer_model.decode(audio_tokens_array)
    os.makedirs(output_dir, exist_ok=True)

    import soundfile as sf

    audio_path = os.path.join(output_dir, "generated_audio.wav")
    audio_np = np.array(decoded_audio[0, 0, :])
    sample_rate = tokenizer_config.sampling_rate
    sf.write(audio_path, audio_np, sample_rate)

    print(f"\n wav file saved: {audio_path}")


def main():
    model_name = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
    tokenizer_name = "XiaomiMiMo/MiMo-Audio-Tokenizer"

    model_path = snapshot_download(model_name)
    tokenizer_path = snapshot_download(tokenizer_name)

    tokenizer_model, tokenizer_config = load_audio_tokenizer(tokenizer_path)
    main_model, config, args, text_tokenizer = load_main_model(model_path)

    text_to_speak = (
        "And now here is my secret, a very simple secret:It is only with the heart that one can see rightly;"
        "What is essential is invisible to the eye.It's the time you wasted for your rose that makes your rose so important."
        "Men have forgotten this truth, but you must not forget it.You become responsible for what you have tamed."
        "You are responsible for your rose..."
    )
    run_inference(
        main_model=main_model,
        tokenizer_model=tokenizer_model,
        text_tokenizer=text_tokenizer,
        config=config,
        args=args,
        tokenizer_config=tokenizer_config,
        text_to_speak=text_to_speak,
        max_steps=300,
    )


if __name__ == "__main__":
    main()
