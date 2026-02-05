import jax
import jax.numpy as jnp
from flax import nnx
from bonsai.models.gat.modeling import GAT

def test_gat_forward_pass():
    print("Initializing GAT model...")
    # 1. Configuration
    key = jax.random.key(0)
    N, F, C = 10, 5, 2  # 10 nodes, 5 features, 2 classes
    
    # 2. Instantiate Model
    model = GAT(
        in_features=F,
        hidden_features=8,
        out_features=C,
        num_heads=2,
        dropout_rng=key,
        dropout_prob=0.6,
        alpha=0.2
    )
    
    print("Model initialized successfully.")

    # 3. Create Dummy Data
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (N, F))
    # Adjacency matrix (binary, symmetric + self-loops)
    adj = jax.random.bernoulli(k2, 0.3, (N, N)).astype(jnp.float32)
    adj = adj + jnp.eye(N)
    adj = jnp.clip(adj, 0.0, 1.0)
    
    print(f"Input features shape: {x.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")

    # 4. Forward Pass
    print("Running forward pass...")
    try:
        logits = model(x, adj, training=False)
        print(f"Logits shape: {logits.shape}")
        
        assert logits.shape == (N, C), f"Expected logits shape {(N, C)}, but got {logits.shape}"
        print("Forward pass successful!")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_gat_forward_pass()
