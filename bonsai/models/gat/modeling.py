import jax
import jax.numpy as jnp
from flax import nnx


class GATLayer(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        rngs: nnx.Rngs,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nnx.Linear(in_features, num_heads * out_features, use_bias=False, rngs=rngs)
        # Attention parameter 'a'
        # shape: (num_heads, 2 * out_features)
        self.a = nnx.Param(nnx.initializers.glorot_uniform()(rngs.params(), (num_heads, 2 * out_features), jnp.float32))

        self.bias = nnx.Param(jnp.zeros((num_heads * out_features if concat else out_features,)))

        self.leaky_relu = lambda x: jax.nn.leaky_relu(x, negative_slope=alpha)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, h: jax.Array, adj: jax.Array, training: bool = True) -> jax.Array:
        # h: (N, in_features)
        # adj: (N, N)
        N = h.shape[0]

        # Linear transformation
        Wh = self.W(h)  # (N, num_heads * out_features)
        Wh = Wh.reshape(N, self.num_heads, self.out_features)  # (N, num_heads, out_features)

        # Prepare attention mechanism
        # Concatenate Wh_i and Wh_j -> (N, N, num_heads, 2 * out_features)
        # However, to avoid O(N^2) memory for features, we can do:
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        #      = LeakyReLU((a_1)^T Wh_i + (a_2)^T Wh_j)

        a1 = self.a.value[:, : self.out_features]  # (num_heads, out_features)
        a2 = self.a.value[:, self.out_features :]  # (num_heads, out_features)

        # Calculate (a_1)^T Wh_i -> (N, num_heads)
        # Wh: (N, num_heads, out_features)
        # a1: (num_heads, out_features)
        # We want inner product over out_features for each head
        attn_1 = jnp.einsum("nho,ho->nh", Wh, a1)  # (N, num_heads)
        attn_2 = jnp.einsum("nho,ho->nh", Wh, a2)  # (N, num_heads)

        # Broadcast add -> (N, N, num_heads)
        e = attn_1[:, None, :] + attn_2[None, :, :]
        e = self.leaky_relu(e)

        # Masked attention
        # adj is assumed to be 0 for no edge, 1 for edge (including self-loop)
        # We want to mask where adj is 0
        zero_vec = -9e15 * jnp.ones_like(e)
        attention = jnp.where(adj[..., None] > 0, e, zero_vec)

        # Softmax over neighbors (dim 1)
        attention = jax.nn.softmax(attention, axis=1)  # (N, N, num_heads)

        # Apply dropout to attention coefficients
        attention = self.dropout(attention, deterministic=not training)

        # Aggregation: h'_i = sum_j alpha_ij W h_j
        # attention: (N, N, num_heads)
        # Wh: (N, num_heads, out_features)
        # Result: (N, num_heads, out_features)
        h_prime = jnp.einsum("ijh,jho->iho", attention, Wh)

        if self.concat:
            # Concatenate heads -> (N, num_heads * out_features)
            output = h_prime.reshape(N, self.num_heads * self.out_features)
        else:
            # Average heads -> (N, out_features)
            output = jnp.mean(h_prime, axis=1)

        return output + self.bias.value


class GAT(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_heads: int,
        dropout_rng: jax.Array,
        dropout_prob: float = 0.6,
        alpha: float = 0.2,
        concat_hidden: bool = True,
        num_layers: int = 2,  # Defaults to 2 layers as per paper/spec
        num_out_heads: int = 1,
    ):
        self.dropout_prob = dropout_prob
        self.layers = nnx.List([])
        rngs = nnx.Rngs(dropout_rng)

        # Input/Hidden Layers
        # Usually GAT has layers 1 to N-1
        current_dim = in_features

        # First layer (and subsequent hidden layers if num_layers > 2)
        for _ in range(num_layers - 1):
            self.layers.append(
                GATLayer(
                    in_features=current_dim,
                    out_features=hidden_features,
                    num_heads=num_heads,
                    rngs=rngs,
                    dropout=dropout_prob,
                    alpha=alpha,
                    concat=True,
                )
            )
            current_dim = hidden_features * num_heads

        # Output Layer
        self.layers.append(
            GATLayer(
                in_features=current_dim,
                out_features=out_features,
                num_heads=num_out_heads,
                rngs=rngs,
                dropout=dropout_prob,
                alpha=alpha,
                concat=False,  # Paper averages the last layer
            )
        )

        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, x: jax.Array, adj: jax.Array, training: bool = True) -> jax.Array:
        h = x
        # Apply dropout to input features
        h = self.dropout(h, deterministic=not training)

        for i, layer in enumerate(self.layers):
            h = layer(h, adj, training=training)
            # Apply elu and dropout for hidden layers
            if i < len(self.layers) - 1:
                h = jax.nn.elu(h)
                h = self.dropout(h, deterministic=not training)

        # Final layer usually is softmax for classification, but we return logits
        return h
