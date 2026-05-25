import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, *features):
        if len(features) == 0:
            raise ValueError("FusionBlock requires at least one input feature")
        merged = torch.cat(features, dim=-1)
        projected = self.projection(merged)
        gate = self.gate(merged)
        return self.norm(gate * projected + (1.0 - gate) * projected)


class ResidualFusionBlock(nn.Module):
    def __init__(self, primary_dim, residual_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fusion = FusionBlock(primary_dim + residual_dim, output_dim, dropout=dropout)

    def forward(self, primary, residual):
        return self.fusion(primary, residual)


class MultiScaleFusion(nn.Module):
    def __init__(self, dims, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = sum(dims)
        self.output_dim = output_dim
        self.fusion = FusionBlock(self.input_dim, output_dim, dropout=dropout)

    def forward(self, features):
        return self.fusion(*features)
