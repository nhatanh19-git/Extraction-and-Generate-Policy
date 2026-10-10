"""Configuration for BiLSTM model."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BiLSTMConfig:
    """Hyperparameters and configuration for BiLSTM model."""
    # Data
    batch_size: int = 16
    
    # Model
    embedding_dim: int = 300  # FastText/GloVe standard
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.5
    use_crf: bool = True
    freeze_embeddings: bool = False
    
    # Training
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 5
    seed: int = 42
    clip_grad_norm: float = 5.0
    
    # Paths
    fasttext_model_path: Optional[str] = None
    glove_model_path: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
