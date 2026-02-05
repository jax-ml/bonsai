# Graph Attention Network (GAT)

A JAX/Flax implementation of the Graph Attention Network (GAT) based on [Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Veličković et al.

This implementation uses `flax.nnx` for explicit state management.

## Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx
from bonsai.models.gat.modeling import GAT

# Configuration
key = jax.random.key(0)
N, F, C = 10, 5, 2

# Instantiate Model
model = GAT(
    in_features=F,
    hidden_features=8,
    out_features=C,
    num_heads=2,
    dropout_rng=key,
    dropout_prob=0.6,
    alpha=0.2
)

# Dummy Data
key, k1, k2 = jax.random.split(key, 3)
x = jax.random.normal(k1, (N, F))
adj = jax.random.bernoulli(k2, 0.3, (N, N)).astype(jnp.float32) + jnp.eye(N)
adj = jnp.clip(adj, 0.0, 1.0)

# Forward Pass
logits = model(x, adj, training=False)
```

## Validation

To reproduce the results on the Cora dataset (~83% accuracy):

```bash
python bonsai/models/gat/tests/GAT_cora_validation.py
```
