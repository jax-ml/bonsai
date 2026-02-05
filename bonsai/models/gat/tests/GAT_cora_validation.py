import os
import urllib.request
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from bonsai.models.gat.modeling import GAT
from bonsai.models.gat.params import GATConfig

# Configuration
# Use a temporary directory for data to avoid polluting the repo
DATA_DIR = os.path.join(os.getcwd(), "data", "cora")
CORA_CONTENT_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.content"
CORA_CITES_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites"


def download_cora():
    os.makedirs(DATA_DIR, exist_ok=True)
    content_path = os.path.join(DATA_DIR, "cora.content")
    cites_path = os.path.join(DATA_DIR, "cora.cites")

    if not os.path.exists(content_path):
        print(f"Downloading {CORA_CONTENT_URL}...")
        urllib.request.urlretrieve(CORA_CONTENT_URL, content_path)

    if not os.path.exists(cites_path):
        print(f"Downloading {CORA_CITES_URL}...")
        urllib.request.urlretrieve(CORA_CITES_URL, cites_path)

    return content_path, cites_path


def load_data():
    content_path, cites_path = download_cora()

    # Load content
    # Format: paper_id, word_attributes..., label
    content = np.genfromtxt(content_path, dtype=np.dtype(str))
    idx = content[:, 0].astype(np.int32)
    features = content[:, 1:-1].astype(np.float32)
    labels_str = content[:, -1]

    # Map labels to integers
    unique_labels = sorted(set(labels_str))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels_str], dtype=np.int32)

    # Map paper IDs to 0-N
    idx_map = {id: i for i, id in enumerate(idx)}

    # Load cites
    # Format: cited_paper_id, citing_paper_id
    edges_unordered = np.genfromtxt(cites_path, dtype=np.int32)

    edges = []
    for edge in edges_unordered:
        if edge[0] in idx_map and edge[1] in idx_map:
            edges.append([idx_map[edge[0]], idx_map[edge[1]]])
    edges = np.array(edges)

    N = features.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    # Symmetric adjacency matrix
    adj = adj + adj.T
    adj = np.clip(adj, 0, 1)

    # Add self-loops
    adj = adj + np.eye(N)

    # Row-normalize features
    features_sum = features.sum(axis=1, keepdims=True)
    features_sum = np.where(features_sum == 0, 1, features_sum)
    features = features / features_sum

    # Standard split indices (based on common Cora implementations)
    # We'll use 20 nodes per class for training, 500 for val, 1000 for test
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)

    # For reproducibility, we use a class-balanced split
    np.random.seed(42)
    for i in range(len(unique_labels)):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        train_mask[indices[:20]] = True
        val_mask[indices[20 : 20 + 71]] = True  # 71 * 7 = 497
        test_mask[indices[20 + 71 : 20 + 71 + 143]] = True  # 143 * 7 = 1001

    return (
        jnp.array(features),
        jnp.array(adj),
        jnp.array(labels),
        jnp.array(train_mask),
        jnp.array(val_mask),
        jnp.array(test_mask),
    )


def loss_fn(model, x, adj, labels, mask, training):
    logits = model(x, adj, training=training)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    # Cross entropy only on masked nodes
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1)


@nnx.jit
def train_step(model, optimizer, x, adj, labels, mask):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, x, adj, labels, mask, True)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, adj, labels, mask):
    logits = model(x, adj, training=False)
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum((preds == labels) * mask)
    total = jnp.sum(mask)
    accuracy = correct / jnp.maximum(total, 1)
    loss = loss_fn(model, x, adj, labels, mask, False)
    return loss, accuracy


def main():
    print("Loading Cora data...")
    x, adj, labels, train_mask, val_mask, test_mask = load_data()
    print(f"Data loaded. Features: {x.shape}, Nodes: {x.shape[0]}, Edges: {jnp.sum(adj > 0)}")

    key = jax.random.key(42)
    model_key, _ = jax.random.split(key)

    config = GATConfig(
        in_features=x.shape[1],
        hidden_features=8,
        out_features=int(jnp.max(labels) + 1),
        num_heads=8,
        num_out_heads=1,
        dropout_prob=0.6,
        alpha=0.2,
    )

    model = GAT(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        out_features=config.out_features,
        num_heads=config.num_heads,
        dropout_rng=model_key,
        dropout_prob=config.dropout_prob,
        alpha=config.alpha,
        num_out_heads=config.num_out_heads,
    )

    # Standard Adam optimizer for GAT
    # Paper uses lr=0.005 and weight_decay=5e-4
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.005), wrt=nnx.Param)

    print("Starting training...")
    best_val_acc = 0
    for epoch in range(1, 201):
        loss = train_step(model, optimizer, x, adj, labels, train_mask)

        if epoch % 10 == 0:
            val_loss, val_acc = eval_step(model, x, adj, labels, val_mask)
            print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    test_loss, test_acc = eval_step(model, x, adj, labels, test_mask)
    print("\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    if test_acc >= 0.80:
        print("SUCCESS: Accuracy is above 80%")
    else:
        print("FAILURE: Accuracy is below 80%")


if __name__ == "__main__":
    main()
