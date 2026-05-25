import torch
import torch.nn as nn

from .dc_gcn import DCGCN
from .mamba_block import MambaTemporalBlock
from .bilstm_attention import BiLSTMAttentionBlock
from .fusion import FusionBlock


class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.e2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        e1_seq = self.e1(x)
        e2_seq = self.e2(x)
        e1_last = e1_seq[:, -1]
        e2_last = e2_seq[:, -1]
        return e1_last, e2_seq, e2_last


class StageEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        gcn_layers=2,
        gcn_dropout=0.1,
        mamba_d_state=16,
        mamba_conv_kernel=3,
        mamba_expand=2,
        mamba_dropout=0.1,
        bilstm_hidden=64,
        bilstm_layers=1,
        attn_heads=4,
        attn_dropout=0.1,
    ):
        super().__init__()
        self.gcn = DCGCN(hidden_dim, num_layers=gcn_layers, dropout=gcn_dropout)
        self.mamba = MambaTemporalBlock(
            d_model=hidden_dim,
            d_state=mamba_d_state,
            d_conv=mamba_conv_kernel,
            expand=mamba_expand,
            dropout=mamba_dropout,
            layers=1,
        )
        self.bilstm_attention = BiLSTMAttentionBlock(
            input_dim=hidden_dim,
            hidden_dim=bilstm_hidden,
            num_layers=bilstm_layers,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )

    def forward(self, seq, node_feature, adj_flow, adj_dist):
        gcn_feature = self.gcn(node_feature, adj_flow, adj_dist)
        mamba_feature, mamba_seq = self.mamba(seq)
        bilstm_feature, bilstm_seq = self.bilstm_attention(seq)
        return gcn_feature, mamba_feature, bilstm_feature, mamba_seq, bilstm_seq


class WaterQualityModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        pred_len=1,
        gcn_layers=2,
        gcn_dropout=0.1,
        mamba_d_state=16,
        mamba_conv_kernel=3,
        mamba_expand=2,
        mamba_dropout=0.1,
        bilstm_hidden=64,
        bilstm_layers=1,
        attn_heads=4,
        attn_dropout=0.1,
        fusion_dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.embedding = FeatureEmbedding(input_dim, hidden_dim, dropout=fusion_dropout)
        self.stage1 = StageEncoder(
            hidden_dim=hidden_dim,
            gcn_layers=gcn_layers,
            gcn_dropout=gcn_dropout,
            mamba_d_state=mamba_d_state,
            mamba_conv_kernel=mamba_conv_kernel,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            bilstm_hidden=bilstm_hidden,
            bilstm_layers=bilstm_layers,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
        )
        self.stage1_fusion = FusionBlock(hidden_dim * 4, hidden_dim, dropout=fusion_dropout)
        self.stage2_input = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.stage2 = StageEncoder(
            hidden_dim=hidden_dim,
            gcn_layers=gcn_layers,
            gcn_dropout=gcn_dropout,
            mamba_d_state=mamba_d_state,
            mamba_conv_kernel=mamba_conv_kernel,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            bilstm_hidden=bilstm_hidden,
            bilstm_layers=bilstm_layers,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
        )
        self.stage2_fusion = FusionBlock(hidden_dim * 7, hidden_dim, dropout=fusion_dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x, adj_flow, adj_dist):
        if x.dim() != 4:
            raise ValueError("WaterQualityModel expects x with shape [B, T, N, F]")
        e1_last, e2_seq, e2_last = self.embedding(x)
        h_gcn1, h_mamba1, h_bilstm1, _, _ = self.stage1(e2_seq, e2_last, adj_flow, adj_dist)
        z1 = self.stage1_fusion(e2_last, h_gcn1, h_mamba1, h_bilstm1)
        stage2_seed = self.stage2_input(torch.cat([e2_last, z1], dim=-1))
        stage2_seq = e2_seq + stage2_seed.unsqueeze(1)
        h_gcn2, h_mamba2, h_bilstm2, _, _ = self.stage2(stage2_seq, stage2_seed, adj_flow, adj_dist)
        u = self.stage2_fusion(e2_last, h_gcn1, h_mamba1, h_bilstm1, h_gcn2, h_mamba2, h_bilstm2)
        v = torch.cat([u, e1_last], dim=-1)
        return self.regressor(v)


def build_model_from_config(config):
    return WaterQualityModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        pred_len=config.pred_len,
        gcn_layers=config.gcn_layers,
        gcn_dropout=config.gcn_dropout,
        mamba_d_state=config.mamba_d_state,
        mamba_conv_kernel=config.mamba_conv_kernel,
        mamba_expand=config.mamba_expand,
        mamba_dropout=config.mamba_dropout,
        bilstm_hidden=config.bilstm_hidden,
        bilstm_layers=config.bilstm_layers,
        attn_heads=config.attn_heads,
        attn_dropout=config.attn_dropout,
        fusion_dropout=config.fusion_dropout,
    )
