import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers import UMT5EncoderModel as TorchUMT5EncoderModel

from bonsai.models.umt5.modeling import UMT5Config, UMT5EncoderModel
from bonsai.models.umt5.params import create_model_from_safe_tensors, load_model_config


def compare_outputs(jax_output: jax.Array, torch_output, name: str, rtol: float = 1e-3, atol: float = 1e-5):
    """Compare JAX and PyTorch outputs and report differences.

    Args:
        jax_output: Output from JAX model
        torch_output: Output from PyTorch model (torch.Tensor)
        name: Name of the output being compared
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    import torch

    if torch_output.dtype == torch.bfloat16:
        torch_output = torch_output.float()

    # Convert PyTorch to numpy
    if isinstance(torch_output, torch.Tensor):
        torch_np = torch_output.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_output)

    # Convert JAX to numpy
    jax_np = np.array(jax_output)

    print(f"\n{'=' * 80}")
    print(f"Comparing: {name}")
    print(f"{'=' * 80}")
    print(f"JAX shape:   {jax_np.shape}")
    print(f"Torch shape: {torch_np.shape}")
    print(f"JAX dtype:   {jax_np.dtype}")
    print(f"Torch dtype: {torch_np.dtype}")

    # Check shapes match
    if jax_np.shape != torch_np.shape:
        print("❌ Shape mismatch!")
        return False

    # Compute differences
    abs_diff = np.abs(jax_np - torch_np)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-10)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)

    print("\nStatistics:")
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"  Mean relative difference: {mean_rel_diff:.2e}")

    print(f"\nJAX output range:   [{np.min(jax_np):.4f}, {np.max(jax_np):.4f}]")
    print(f"Torch output range: [{np.min(torch_np):.4f}, {np.max(torch_np):.4f}]")

    # Check if within tolerance
    close = np.allclose(jax_np, torch_np, rtol=rtol, atol=atol)

    if close:
        print(f"\n✅ Outputs match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"\n❌ Outputs do NOT match (rtol={rtol}, atol={atol})")
        # Show some mismatched locations
        mismatch_mask = ~np.isclose(jax_np, torch_np, rtol=rtol, atol=atol)
        n_mismatches = np.sum(mismatch_mask)
        print(f"  Number of mismatches: {n_mismatches} / {jax_np.size} ({100 * n_mismatches / jax_np.size:.2f}%)")

    return close


def check_weight_loading(jax_model: UMT5EncoderModel, torch_model: TorchUMT5EncoderModel):
    # 1. Embedding weights
    # torch_model is WanT5EncoderModel, torch_model.model is T5Encoder
    jax_emb = jax_model.encoder.embed_tokens.embedding.get_value()
    hf_emb = torch_model.encoder.embed_tokens.weight.float().detach().cpu().numpy()

    print("Embedding weights:")
    print(f"  Shapes: jax={jax_emb.shape}, torch={hf_emb.shape}")
    print(f"  Max diff: {np.abs(jax_emb - hf_emb).max():.2e}")

    # PyTorch: encoder.block[0].layer[0].SelfAttention.q.weight
    for layer in range(len(jax_model.encoder.block)):
        jax_attn_q = jax_model.encoder.block[layer].layer[0].SelfAttention.q.kernel.get_value()
        torch_attn_q = torch_model.encoder.block[layer].layer[0].SelfAttention.q.weight.float().detach().cpu().numpy()

        print("\nSelf Attention Query Weight:")
        print(f"  Shapes: jax={jax_attn_q.shape}, torch={torch_attn_q.shape}")
        print(f"  Max diff: {np.abs(jax_attn_q - torch_attn_q.T).max():.2e}")

        jax_attn_k = jax_model.encoder.block[layer].layer[0].SelfAttention.k.kernel.get_value()
        torch_attn_k = torch_model.encoder.block[layer].layer[0].SelfAttention.k.weight.float().detach().cpu().numpy()

        print("\nSelf Attention Key Weight:")
        print(f"  Shapes: jax={jax_attn_k.shape}, torch={torch_attn_k.shape}")
        print(f"  Max diff: {np.abs(jax_attn_k - torch_attn_k.T).max():.2e}")

        jax_attn_v = jax_model.encoder.block[layer].layer[0].SelfAttention.v.kernel.get_value()
        torch_attn_v = torch_model.encoder.block[layer].layer[0].SelfAttention.v.weight.float().detach().cpu().numpy()

        print("\nSelf Attention Value Weight:")
        print(f"  Shapes: jax={jax_attn_v.shape}, torch={torch_attn_v.shape}")
        print(f"  Max diff: {np.abs(jax_attn_v - torch_attn_v.T).max():.2e}")

        jax_attn_o = jax_model.encoder.block[layer].layer[0].SelfAttention.o.kernel.get_value()
        torch_attn_o = torch_model.encoder.block[layer].layer[0].SelfAttention.o.weight.float().detach().cpu().numpy()

        print("\nSelf Attention o Weight:")
        print(f"  Shapes: jax={jax_attn_o.shape}, torch={torch_attn_o.shape}")
        print(f"  Max diff: {np.abs(jax_attn_o - torch_attn_o.T).max():.2e}")

        jax_attn_pos = (
            jax_model.encoder.block[layer].layer[0].SelfAttention.relative_attention_bias.embedding.get_value()
        )
        torch_attn_pos = (
            torch_model.encoder.block[layer]
            .layer[0]
            .SelfAttention.relative_attention_bias.weight.float()
            .detach()
            .cpu()
            .numpy()
        )

        print("\nSelf Attention Position Embedding Weight:")
        print(f"  Shapes: jax={jax_attn_pos.shape}, torch={torch_attn_pos.shape}")
        print(f"  Max diff: {np.abs(jax_attn_pos - torch_attn_pos).max():.2e}")

        jax_attn_norm = jax_model.encoder.block[layer].layer[0].layer_norm.scale.get_value()
        torch_attn_norm = torch_model.encoder.block[layer].layer[0].layer_norm.weight.float().detach().cpu().numpy()

        print("\nSelf Attention RMS Norm weight:")
        print(f"  Shapes: jax={jax_attn_norm.shape}, torch={torch_attn_norm.shape}")
        print(f"  Max diff: {np.abs(jax_attn_norm - torch_attn_norm).max():.2e}")

        jax_ffn_norm = jax_model.encoder.block[layer].layer[1].layer_norm.scale.get_value()
        torch_ffn_norm = torch_model.encoder.block[layer].layer[1].layer_norm.weight.float().detach().cpu().numpy()

        print("\nFFN RMS Norm weight:")
        print(f"  Shapes: jax={jax_ffn_norm.shape}, torch={torch_ffn_norm.shape}")
        print(f"  Max diff: {np.abs(jax_ffn_norm - torch_ffn_norm).max():.2e}")

        jax_ffn_gate = jax_model.encoder.block[layer].layer[1].DenseReluDense.wi_0.kernel.get_value()
        torch_ffn_gate = (
            torch_model.encoder.block[layer].layer[1].DenseReluDense.wi_0.weight.float().detach().cpu().numpy()
        )

        print("\nFFN Gate Weight:")
        print(f"  Shapes: jax={jax_ffn_gate.shape}, torch={torch_ffn_gate.shape}")
        print(f"  Max diff: {np.abs(jax_ffn_gate - torch_ffn_gate.T).max():.2e}")

        jax_ffn_fc1 = jax_model.encoder.block[layer].layer[1].DenseReluDense.wi_1.kernel.get_value()
        torch_ffn_fc1 = (
            torch_model.encoder.block[layer].layer[1].DenseReluDense.wi_1.weight.float().detach().cpu().numpy()
        )

        print("\nFFN FC1 Weight:")
        print(f"  Shapes: jax={jax_ffn_fc1.shape}, torch={torch_ffn_fc1.shape}")
        print(f"  Max diff: {np.abs(jax_ffn_fc1 - torch_ffn_fc1.T).max():.2e}")

        jax_ffn_fc2 = jax_model.encoder.block[layer].layer[1].DenseReluDense.wo.kernel.get_value()
        torch_ffn_fc2 = (
            torch_model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.float().detach().cpu().numpy()
        )

        print("\nFFN FC2 Weight:")
        print(f"  Shapes: jax={jax_ffn_fc2.shape}, torch={torch_ffn_fc2.shape}")
        print(f"  Max diff: {np.abs(jax_ffn_fc2 - torch_ffn_fc2.T).max():.2e}")


def test_t5_encoder():
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    model_conf = load_model_config(model_ckpt_path + "/text_encoder")
    jax_t5 = create_model_from_safe_tensors(
        file_dir=model_ckpt_path + "/text_encoder",
        cfg=model_conf,
    )

    hf_t5 = TorchUMT5EncoderModel.from_pretrained(
        model_ckpt_path,
        subfolder="text_encoder",
    )

    check_weight_loading(jax_t5, hf_t5)

    prompt = "A beautiful sunset over the ocean with waves crashing on the shore"
    max_length = 512
    jax_inputs = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="np")
    torch_inputs = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    print(f"{jax_inputs.attention_mask.shape=}")

    pytorch_output = hf_t5(input_ids=torch_inputs.input_ids, attention_mask=torch_inputs.attention_mask)
    jax_output = jax_t5(input_ids=jax_inputs.input_ids, attention_mask=jax_inputs.attention_mask)

    torch_embeddings = pytorch_output.last_hidden_state

    seq_len = torch_inputs.attention_mask.gt(0).sum(dim=1).long()

    print(torch_embeddings.shape, jax_output.shape)
    compare_outputs(jax_output[:, :seq_len, :], torch_embeddings[:, :seq_len, :], "UMT5")


if __name__ == "__main__":
    test_t5_encoder()
