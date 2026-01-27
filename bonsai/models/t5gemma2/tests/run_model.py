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

"""Test script for T5Gemma2 model (multimodal).

Usage:
    python -m bonsai.models.t5gemma2.tests.run_model
    python -m bonsai.models.t5gemma2.tests.run_model --demo image
    python -m bonsai.models.t5gemma2.tests.run_model --demo translate
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import requests
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer

from bonsai.models.t5gemma2 import modeling, params


# =============================================================================
# Special tokens from modeling.py
# =============================================================================

BOS_TOKEN = modeling.BOS_TOKEN  # 2
EOS_TOKEN = modeling.EOS_TOKEN  # 1
NEW_LINE_TOKEN = modeling.NEW_LINE_TOKEN  # 108
START_OF_IMAGE_TOKEN = modeling.START_OF_IMAGE_TOKEN  # 255999
END_OF_IMAGE_TOKEN = modeling.END_OF_IMAGE_TOKEN  # 256000
IMAGE_PLACEHOLDER_TOKEN = modeling.IMAGE_PLACEHOLDER_TOKEN  # 256001


# =============================================================================
# Image token utilities (from text.py, inlined for standalone usage)
# =============================================================================


def _get_new_text_positions(*, offset_on: np.ndarray, offset_by: int) -> np.ndarray:
    """Create the positions of the new tokens."""
    offset = np.cumsum(offset_on, axis=-1) * offset_by
    new_positions = np.arange(offset_on.shape[-1]) + offset
    new_positions -= offset_by * offset_on
    return new_positions


def _insert_sequence(
    tokens: np.ndarray,
    *,
    at: int,
    sequence: list[int],
    max_num_images: int,
) -> np.ndarray:
    """Inserts a sequence of tokens at all occurrences of a specific token."""
    original_dim = tokens.ndim
    if original_dim == 1:
        tokens = tokens[None, :]

    batch_size, length = tokens.shape
    mm_tokens_to_insert = np.array(sequence)
    offset_by = len(mm_tokens_to_insert) - 1
    length_with_mm = length + max_num_images * offset_by
    mm_start = tokens == at

    new_tokens = np.zeros((batch_size, length_with_mm), dtype=np.int64)
    new_text_pos = _get_new_text_positions(offset_on=mm_start, offset_by=offset_by)
    np.put_along_axis(new_tokens, new_text_pos, tokens, axis=1)

    batch_indices_to_zero, _ = np.where(mm_start)
    new_pos_to_zero = new_text_pos[mm_start]
    if batch_indices_to_zero.size > 0:
        new_tokens[batch_indices_to_zero, new_pos_to_zero] = 0

    batch_indices, seq_indices = np.nonzero(mm_start)

    if batch_indices.size > 0:
        intra_batch_img_idx = np.cumsum(mm_start, axis=1)[mm_start] - 1
        final_img_start_pos = seq_indices + intra_batch_img_idx * offset_by
        indices_to_insert = final_img_start_pos[:, None] + np.arange(
            len(mm_tokens_to_insert),
        )
        new_tokens[batch_indices[:, None], indices_to_insert] = mm_tokens_to_insert

    if original_dim == 1:
        new_tokens = np.squeeze(new_tokens)
    return new_tokens


def add_extra_tokens_for_images(
    tokens: np.ndarray | list,
    *,
    new_line_token: int,
    start_of_image_token: int,
    end_of_image_token: int,
    image_placeholder_token: int,
    num_placeholder_tokens_per_image: int = 256,
    max_num_images: int = 1,
) -> np.ndarray:
    """Add extra image tokens to text tokens.

    Expands <start_of_image> token into the full image token sequence:
    \\n <start_of_image> [256 x placeholder] <end_of_image> \\n
    """
    mm_tokens = [
        new_line_token,
        start_of_image_token,
        *[image_placeholder_token] * num_placeholder_tokens_per_image,
        end_of_image_token,
        new_line_token,
    ]
    if not isinstance(tokens, np.ndarray):
        tokens = np.asarray(tokens)

    return _insert_sequence(
        at=start_of_image_token,
        sequence=mm_tokens,
        tokens=tokens,
        max_num_images=max_num_images,
    )


# =============================================================================
# Tokenization utilities
# =============================================================================


def tokenize(tokenizer, text: str) -> jnp.ndarray:
    """Tokenize text with BOS token prepended.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Input text to tokenize.

    Returns:
        Token IDs as jnp.ndarray of shape [1, seq_len].
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    # Prepend BOS token
    token_ids = [BOS_TOKEN] + token_ids
    return jnp.array([token_ids], dtype=jnp.int32)


def create_multimodal_input(
    tokenizer,
    text: str,
    image_path: str | None = None,
    *,
    image_size: int = 896,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Create batched input_ids and pixel_values for the model.

    Handles both text tokenization and image preprocessing. When an image is
    provided and text contains <start_of_image>, the function:
    1. Loads and preprocesses the image
    2. Tokenizes the text
    3. Expands <start_of_image> into the full image token sequence

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Text prompt (may contain <start_of_image> marker).
        image_path: Optional path to image file or URL.
        image_size: Target image size (default 896 for T5Gemma2).

    Returns:
        Tuple of (input_ids, pixel_values):
        - input_ids: Token IDs of shape [1, seq_len]
        - pixel_values: Image array of shape [1, 1, H, W, C] or None if no image
    """
    # Tokenize text with BOS
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = [BOS_TOKEN] + token_ids

    # Process image if provided and text contains image marker
    pixel_values = None
    if image_path is not None and "<start_of_image>" in text:
        # Load image
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)

        # Convert to RGB and resize
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)

        # Normalize to [-1, 1] and format as [1, 1, H, W, C]
        image_array = np.array(image, dtype=np.float32) / 127.5 - 1.0
        pixel_values = jnp.array(image_array[None, None, ...], dtype=jnp.float32)

        # Expand <start_of_image> into full image token sequence
        token_ids = add_extra_tokens_for_images(
            token_ids,
            new_line_token=NEW_LINE_TOKEN,
            start_of_image_token=START_OF_IMAGE_TOKEN,
            end_of_image_token=END_OF_IMAGE_TOKEN,
            image_placeholder_token=IMAGE_PLACEHOLDER_TOKEN,
            max_num_images=1,
        )

    # Format as batched array [1, seq_len]
    input_ids = jnp.array([token_ids], dtype=jnp.int32)

    return input_ids, pixel_values


# =============================================================================
# Generation
# =============================================================================


def greedy_generate(
    model: modeling.T5Gemma2,
    encoder_input_ids: jnp.ndarray,
    max_new_tokens: int = 50,
    eos_token_ids: int | list[int] = EOS_TOKEN,
    pixel_values: jnp.ndarray | None = None,
    use_cache: bool = True,
) -> jnp.ndarray:
    """Greedy decoding for T5Gemma2.

    Args:
        model: T5Gemma2 model instance.
        encoder_input_ids: Encoder input token IDs [B, L].
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_ids: End of sequence token ID(s).
        pixel_values: Optional image pixel values [B, N, H, W, C].
        use_cache: Whether to use KV cache for faster decoding.

    Returns:
        Generated token IDs [B, num_generated_tokens].
    """
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    batch_size = encoder_input_ids.shape[0]
    encoder_seq_len = encoder_input_ids.shape[1]

    # Create encoder attention mask
    encoder_mask = jnp.ones_like(encoder_input_ids, dtype=jnp.bool_)

    # Encode input (with optional images)
    encoder_outputs = model.encoder(
        encoder_input_ids,
        attention_mask=encoder_mask,
        images=pixel_values,
    )

    # Initialize cache if using cached decoding
    if use_cache:
        model.init_cache(
            batch_size=batch_size,
            max_decode_length=max_new_tokens + 1,
            encoder_seq_length=encoder_seq_len,
        )

    # Start with BOS token
    decoder_input_ids = jnp.full((batch_size, 1), BOS_TOKEN, dtype=jnp.int32)
    generated_ids = []

    for step in range(max_new_tokens):
        if use_cache:
            if step == 0:
                # First step: prefill with BOS
                decoder_outputs = model.decoder(
                    decoder_input_ids,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=encoder_mask,
                    decode=True,
                )
            else:
                # Subsequent steps: feed only the new token
                decoder_outputs = model.decoder(
                    next_token[:, None],
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=encoder_mask,
                    decode=True,
                )
        else:
            # No cache: recompute full sequence each step
            decoder_outputs = model.decoder(
                decoder_input_ids,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=encoder_mask,
                decode=False,
            )

        # Get logits from the last position
        last_hidden = decoder_outputs[:, -1:, :]
        # Use embedding weights for output projection (tied embeddings)
        embed_table = model.decoder.embedder.embedding[...]
        logits = jnp.einsum("btd,vd->btv", last_hidden, embed_table)

        # Greedy selection
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_ids.append(next_token)

        if not use_cache:
            decoder_input_ids = jnp.concatenate(
                [decoder_input_ids, next_token[:, None]],
                axis=1,
            )

        # Check for EOS
        is_eos = jnp.any(jnp.array([next_token == eos for eos in eos_token_ids]))
        if is_eos:
            break

    return jnp.stack(generated_ids, axis=1)


# =============================================================================
# Demo functions
# =============================================================================


def run_model():
    """Run T5Gemma2 model with example prompts."""
    # Download model checkpoint
    model_ckpt_path = snapshot_download("google/t5gemma-2-270m-270m")
    config = modeling.T5Gemma2Config.t5gemma2_270m_270m(with_vision=True)

    # Example queries for translation (text-only)
    queries = [
        "Translate English to French: Hello, how are you?",
        "Translate English to German: The weather is nice today.",
    ]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)

    print("Loading model from checkpoint...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config)
    print("Model loaded!")

    generate_steps = 32

    for query in queries:
        print(f"\nInput: {query}")

        # Tokenize
        tokens = tokenize(tokenizer, query)
        print(f"Token shape: {tokens.shape}")

        # Generate
        generated = greedy_generate(
            model,
            tokens,
            max_new_tokens=generate_steps,
            use_cache=True,
        )

        # Decode output
        output_tokens = jax.device_get(generated[0])
        # Find EOS and truncate
        eos_indices = np.where(output_tokens == EOS_TOKEN)[0]
        if eos_indices.size > 0:
            output_tokens = output_tokens[: eos_indices[0]]

        decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"Output: {decoded}\n")


def run_multimodal_demo():
    """Run a multimodal demo with images."""
    model_ckpt_path = snapshot_download("google/t5gemma-2-270m-270m")
    config = modeling.T5Gemma2Config.t5gemma2_270m_270m(with_vision=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)

    print("Loading model from checkpoint...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config)
    print("Model loaded!")

    # Test images with prompts
    image_prompts = [
        (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            "<start_of_image> in this image, there is",
        ),
    ]

    print("=" * 80)
    print("Multimodal Demo - Image Captioning")
    print("=" * 80)

    for image_url, text in image_prompts:
        print(f"\nImage: {image_url[:60]}...")
        print(f"Prompt: {text}")

        # Create multimodal input
        input_ids, pixel_values = create_multimodal_input(
            tokenizer, text, image_path=image_url
        )

        print(f"Input IDs shape: {input_ids.shape}")
        if pixel_values is not None:
            print(f"Pixel values shape: {pixel_values.shape}")

        # Generate
        generated = greedy_generate(
            model,
            input_ids,
            max_new_tokens=30,
            pixel_values=pixel_values,
            use_cache=True,
        )

        output_tokens = jax.device_get(generated[0])
        decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"Output: {decoded}")
        print("-" * 80)

    print("\nMultimodal demo complete!")


def run_translation_demo():
    """Run a translation demo with few-shot examples."""
    model_ckpt_path = snapshot_download("google/t5gemma-2-270m-270m")
    config = modeling.T5Gemma2Config.t5gemma2_270m_270m(with_vision=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)

    print("Loading model from checkpoint...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config)
    print("Model loaded!")

    # Few-shot translation prompt (English to Chinese)
    prompt = """Translate the text from English to Chinese (zh_CN).

    English: Hook the vacuum up to it, it helps keep the dust down, and you can prep your concrete that way also. We prefer to do it with the hand grinders ourself, so either way will work pretty good. Usually if it's got an epoxy coating already, the hand grinders work a little better, but this thing works really good too. So after we do that, we fix all the cracks, fix all the divots with our patch repair material, and then we grind them smooth. And then we clean the concrete and get it ready for the first coating. This is going to be a 100% solids epoxy, so it goes on in its different procedures, different stages.
    Chinese (zh_CN): 把吸尘器连在上面，这样可以减少灰尘，你还可以顺便处理一下混凝土。我们更喜欢用手持式研磨机，两种方法的效果都不错。通常，如果表面已经有环氧树脂涂层的话，手持式研磨机的效果会更好一些，但这台机器的效果也很好。处理完之后，我们用修补材料把所有的裂缝和坑洼都补好，再把它们磨平。然后清洁混凝土表面，为涂刷第一层涂料做好准备。我们这里用的是 100% 固体环氧树脂，所以涂刷的时候会有不同的步骤和阶段。

    English: The Zoroastrian text, Vendidad, states that Yima built an underground city on the orders of the god Ahura Mazda, to protect his people from a catastrophic winter. Much like the account of Noah in the Bible, Yima was instructed to collect pairs of the best animals and people as well as the best seeds in order to reseed the Earth after the winter cataclysm. This was before the last Ice Age, 110,000 years ago.
    Chinese (zh_CN): 琐罗亚斯德教经典《万迪达德》中说，Yima 奉阿胡拉·马兹达神之命建造了一座地下城市，以保护他的人民躲避一场灾难性的寒冬。就像《圣经》中诺亚的故事一样，Yima 被指示收集最好的动物、人类以及最好的种子，以便在冬季灾难过后重新在地球上播种。这发生在最后一个冰河时代之前，也就是 11 万年前。

    English: Okay, so let me explain. All right, so the problem is, if you look inside there... You see the wood siding? There's the old siding, and it butts up to the shingles there. And then I put this over it. And what happens is the dirt collects there, to that flashing.
    Chinese (zh_CN): 好的，我来解释一下。问题是，你们看里面……看到木头墙板了吗？那是原来的墙板，紧挨着那边的瓦。然后我把这个盖在上面。结果灰尘就堆积在那儿，堆积到泛水板上。

    English: Hey guys, Thunder E here, and welcome to the video you've been waiting for. I am talking about gaming on the ASUS ROG Phone 5. Now, the ROG Phone series is well known for its gaming powers, but in this video, we're going to find out if the ROG Phone 5 is truly taking back the crown as the king of gaming phones.
    Chinese (zh_CN): 大家好，我是雷霆 E，欢迎收看大家期待已久的视频。今天要评测的是华硕 ROG Phone 5 的游戏性能。ROG Phone 系列手机一直以其强大的游戏性能而闻名，那么，ROG Phone 5 能否真正加冕"游戏手机之王"？我们拭目以待。

    English: It is December 1997, and the Imperial Sugar Company is acquiring a new production site at Port Wentworth from Savannah Foods and Industries Incorporated. There is nothing really of note here. It was doing what businesses do, and that is acquiring to expand. The site has been home to food production and processing since the early 1900s. Savannah Industries Incorporated began construction of granulated sugar production facilities at Port Wentworth during the 1910s, completing it in 1917.
    Chinese (zh_CN): 那是 1997 年 12 月，帝国糖业公司正从萨凡纳食品和工业有限公司手中收购位于温特沃斯港的一个新生产基地。这的确没什么值得注意的，它做的只是一家公司都会做的事情，那就是通过收购来扩张。该基地自 20 世纪初以来一直是食品生产和加工的场所。萨凡纳工业有限公司在 20 世纪 10 年代开始在温特沃斯港建造砂糖生产设施，并于 1917 年竣工。

    English: Time for the Scotty Kilmer channel. Does your car have faded paint on it? Then stay tuned, because today I'm going to show you how to polish off faded paint. And all it takes is a bucket of water, a polisher, and a bottle of this Meguiar's Ultimate Compound.
    Chinese (zh_CN):"""

    print("=" * 60)
    print("Translation Demo: English to Chinese (zh_CN)")
    print("=" * 60)

    tokens = tokenize(tokenizer, prompt)
    print(f"\nInput length: {tokens.shape[1]} tokens")

    print("\nGenerating (max 100 tokens, stops at EOS or newline)...")
    generated = greedy_generate(
        model,
        tokens,
        max_new_tokens=100,
        eos_token_ids=[EOS_TOKEN, NEW_LINE_TOKEN],
        use_cache=True,
    )

    output_tokens = jax.device_get(generated[0])
    decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)

    print(f"\n{'=' * 60}")
    print("Generated Translation:")
    print("=" * 60)
    print(decoded)
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run T5Gemma2 model tests")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["image", "translate"],
        default=None,
        help="Run a demo: 'image' for multimodal, 'translate' for translation",
    )
    args = parser.parse_args()

    if args.demo == "image":
        run_multimodal_demo()
    elif args.demo == "translate":
        run_translation_demo()
    else:
        print("Running default text-only demo...")
        print("Use --demo image for multimodal or --demo translate for translation\n")
        run_model()


if __name__ == "__main__":
    main()
