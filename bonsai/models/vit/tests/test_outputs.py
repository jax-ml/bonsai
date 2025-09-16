import os

import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import ViTConfig, ViTForImageClassification, ViTModel

from bonsai.models.vit import params
from bonsai.models.vit.modeling import Embeddings, TransformerEncoder

# TODO: Generate data with jax for determinism
# TODO: Construct objects and transfer weights before to save time
RTOL: float = 1e-3
ATOL: float = 1e-3
BATCH_SIZE: int = 32
CHANNELS: int = 3
HEIGHT: int = 224
WIDTH: int = 224

HIDDEN_SHAPE: tuple[int] = (BATCH_SIZE, 197, 768)


def t2j(arr: torch.Tensor) -> jnp.ndarray:
    """Converts a torch.Tensor to a jnp.ndarray"""
    return jnp.array(arr.cpu().detach().numpy())


## EMBEDDINGS
class ViTTest(absltest.TestCase):
    def test_embeddings(self):
        # Get models
        configuration = ViTConfig()
        torch_model = ViTModel(configuration)
        torch_emb = torch_model.embeddings
        img_size, patch_size = configuration.image_size, configuration.patch_size
        nnx_emb = Embeddings(
            (img_size,) * 2,
            (patch_size,) * 2,
            configuration.num_channels,
            configuration.hidden_size,
            configuration.hidden_dropout_prob,
            rngs=nnx.Rngs(0),
        )

        # Construct data
        torch_input_data = torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))
        nnx_input_data = jnp.permute_dims(t2j(torch_input_data), (0, 2, 3, 1))

        # Transfer weights
        t_state = torch_emb.state_dict()
        nnx_emb.projection.kernel.value = t2j(t_state["patch_embeddings.projection.weight"]).transpose(2, 3, 1, 0)
        nnx_emb.projection.bias.value = t2j(t_state["patch_embeddings.projection.bias"])
        nnx_emb.cls_token.value = t2j(t_state["cls_token"])
        nnx_emb.pos_embeddings.value = t2j(t_state["position_embeddings"])

        # Compute forward passes
        with torch.no_grad():
            torch_emb_out = torch_emb(torch_input_data)
        nnx_emb_out = nnx_emb(nnx_input_data)

        # compare
        np.testing.assert_allclose(nnx_emb_out, t2j(torch_emb_out), rtol=RTOL, atol=ATOL)
        # np.allclose()

    ## ENCODINGS
    def test_layers(self):
        # Get models
        configuration = ViTConfig()
        torch_model = ViTModel(configuration)
        torch_enc = torch_model.encoder.layer[0]
        # nnx_enc = ViTLayerNNX(configuration, rngs=nnx.Rngs(0))
        nnx_enc = TransformerEncoder(
            configuration.num_attention_heads,
            configuration.hidden_size,
            configuration.intermediate_size,
            configuration.hidden_dropout_prob,
            configuration.layer_norm_eps,
            rngs=nnx.Rngs(0),
        )

        # Construct data
        torch_input_data = torch.randn(HIDDEN_SHAPE)
        nnx_input_data = t2j(torch_input_data)

        ## CONSTANTS
        embed_dim, num_heads, head_dim = 768, 12, 64

        # Transfer weights
        t_state = torch_enc.state_dict()
        nnx_enc.attention.query.kernel.value = t2j(t_state["attention.attention.query.weight"]).T.reshape(
            embed_dim, num_heads, head_dim
        )
        nnx_enc.attention.query.bias.value = t2j(t_state["attention.attention.query.bias"]).reshape(num_heads, head_dim)
        nnx_enc.attention.key.kernel.value = t2j(t_state["attention.attention.key.weight"]).T.reshape(
            embed_dim, num_heads, head_dim
        )
        nnx_enc.attention.key.bias.value = t2j(t_state["attention.attention.key.bias"]).reshape(num_heads, head_dim)
        nnx_enc.attention.value.kernel.value = t2j(t_state["attention.attention.value.weight"]).T.reshape(
            embed_dim, num_heads, head_dim
        )
        nnx_enc.attention.value.bias.value = t2j(t_state["attention.attention.value.bias"]).reshape(num_heads, head_dim)
        nnx_enc.attention.out.kernel.value = t2j(t_state["attention.output.dense.weight"]).T.reshape(
            num_heads, head_dim, embed_dim
        )
        nnx_enc.attention.out.bias.value = t2j(t_state["attention.output.dense.bias"])
        nnx_enc.linear1.kernel.value = t2j(t_state["intermediate.dense.weight"]).T
        nnx_enc.linear1.bias.value = t2j(t_state["intermediate.dense.bias"])
        nnx_enc.linear2.kernel.value = t2j(t_state["output.dense.weight"]).T
        nnx_enc.linear2.bias.value = t2j(t_state["output.dense.bias"])
        nnx_enc.layernorm_before.scale.value = t2j(t_state["layernorm_before.weight"])
        nnx_enc.layernorm_before.bias.value = t2j(t_state["layernorm_before.bias"])
        nnx_enc.layernorm_after.scale.value = t2j(t_state["layernorm_after.weight"])
        nnx_enc.layernorm_after.bias.value = t2j(t_state["layernorm_after.bias"])

        # Compute forward passes
        with torch.no_grad():
            torch_enc_out = torch_enc(torch_input_data)
        nnx_enc_out = nnx_enc(nnx_input_data)

        # Compare
        np.testing.assert_allclose(nnx_enc_out, t2j(torch_enc_out), rtol=RTOL, atol=ATOL)

    def test_full(self):
        model_name = "google/vit-base-patch16-224"
        MODEL_CP_PATH = "/tmp/models-bonsai/" + model_name.split("/")[1]

        if not os.path.isdir(MODEL_CP_PATH):
            snapshot_download(model_name, local_dir=MODEL_CP_PATH)

        safetensors_path = os.path.join(MODEL_CP_PATH, "model.safetensors")
        bonsai_model = params.create_vit_from_pretrained(safetensors_path)

        baseline_model = ViTForImageClassification.from_pretrained(model_name)

        torch_input_data = torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))
        nnx_input_data = jnp.permute_dims(t2j(torch_input_data), (0, 2, 3, 1))

        with torch.no_grad():
            torch_out = baseline_model(torch_input_data).logits
        nnx_out = bonsai_model(nnx_input_data)

        # Compare
        np.testing.assert_allclose(nnx_out, t2j(torch_out), rtol=RTOL, atol=0.1)


if __name__ == "__main__":
    absltest.main()
