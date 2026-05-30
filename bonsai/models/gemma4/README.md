# Gemma 4 Model

This is the implementation of the Gemma 4 architecture in JAX and Flax NNX for `bonsai`.

## Architecture Details

Gemma 4 introduces a hybrid MoE (Mixture of Experts) and hybrid attention pattern:
- **Attention Pattern**: A repeating cycle of 5 Local Sliding Window attention layers followed by 1 Global attention layer (`LOCAL_SLIDING`, `LOCAL_SLIDING`, `LOCAL_SLIDING`, `LOCAL_SLIDING`, `LOCAL_SLIDING`, `GLOBAL`).
- **Mixture of Experts**: Combines a top-k routed expert module with a single persistent, wider shared expert module.
- **Normalization**: Utilizes specialized zero-scale RMSNorm layers within MoE gating mechanisms alongside standard offset-scale RMSNorms (`1 + scale`) throughout the model. Furthermore, Query and Key embeddings use RMSNorm.
- **Logit Soft-capping**: Output logits are optionally soft-capped (usually value 30.0) before softmax.

## Configuration

The default base configuration configures:
- Hybrid attention logic with independent relative RoPE frequency parameters (`global_rope_proportion`, `local_rope_proportion`).
- The necessary layer size parameters: `num_hidden_layers`, `hidden_size`, `intermediate_size`.
- The MoE parameters: `num_experts`, `num_shared_experts`, and `num_experts_per_tok`.

## Example Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx
from bonsai.models.gemma4 import Gemma4Config, Gemma4ForCausalLM

# Initialize base configuration
config = Gemma4Config.gemma4_base()

# Initialize model
rngs = nnx.Rngs(0)
model = Gemma4ForCausalLM(config, rngs=rngs)

# Forward pass
input_ids = jnp.array([[1, 2, 3, 4]])
positions = jnp.array([[0, 1, 2, 3]])

logits = model(input_ids, positions=positions)
print(logits.shape) # (1, 4, 256000)
```
