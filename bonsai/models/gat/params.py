from dataclasses import dataclass, field


@dataclass
class GATConfig:
    """Configuration class for Graph Attention Network (GAT)."""

    in_features: int = field(metadata={"help": "Dimension of input node features"})
    hidden_features: int = field(default=8, metadata={"help": "Dimension of hidden features PER HEAD"})
    out_features: int = field(default=7, metadata={"help": "Dimension of output features (classes)"})
    num_heads: int = field(default=8, metadata={"help": "Number of attention heads for hidden layers"})
    num_out_heads: int = field(default=1, metadata={"help": "Number of attention heads for output layer"})
    num_layers: int = field(default=2, metadata={"help": "Number of GAT layers"})
    dropout_prob: float = field(default=0.6, metadata={"help": "Dropout probability"})
    alpha: float = field(default=0.2, metadata={"help": "LeakyReLU negative slope"})
