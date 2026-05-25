from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    data_path: str = "data/sample_data.csv"
    adj_distance_path: str = "data/adj_distance.npy"
    adj_hydrological_path: str = "data/adj_hydrological.npy"
    checkpoint_path: str = "best_model.pt"
    time_col: str = "timestamp"
    site_col: str = "site_id"
    target_col: str = "TN"
    feature_cols: Optional[List[str]] = None
    num_nodes: int = 17
    input_dim: int = 8
    seq_len: int = 30
    pred_len: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    hidden_dim: int = 64
    gcn_layers: int = 2
    gcn_dropout: float = 0.1
    mamba_d_state: int = 16
    mamba_conv_kernel: int = 3
    mamba_expand: int = 2
    mamba_dropout: float = 0.1
    bilstm_hidden: int = 64
    bilstm_layers: int = 1
    attn_heads: int = 4
    attn_dropout: float = 0.1
    fusion_dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    patience: int = 12
    seed: int = 42
    device: str = "cpu"
    add_self_loops: bool = True
    strict_missing: bool = True
    standardize_target: bool = True
    num_workers: int = 0
    pin_memory: bool = False

    def finalize(self):
        if self.feature_cols is None:
            self.feature_cols = [
                "water_temperature",
                "pH",
                "dissolved_oxygen",
                "conductivity",
                "turbidity",
                "permanganate_index",
                "ammonia_nitrogen",
                "total_phosphorus",
            ]
        self.input_dim = len(self.feature_cols)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.hidden_dim % self.attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn_heads")
        return self


def get_config() -> Config:
    return Config().finalize()
