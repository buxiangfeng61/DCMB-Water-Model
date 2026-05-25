import torch
import torch.nn as nn


class BiLSTMAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        if (hidden_dim * 2) % num_heads != 0:
            raise ValueError("2 * hidden_dim must be divisible by num_heads")
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(input_dim)
        self.context_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("BiLSTMAttentionBlock expects x with shape [B, T, N, D]")
        batch, steps, nodes, channels = x.shape
        h = x.permute(0, 2, 1, 3).contiguous().view(batch * nodes, steps, channels)
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, need_weights=False)
        projected = self.projection(attn_out)
        gate = self.context_gate(torch.cat([projected, h], dim=-1))
        mixed = gate * projected + (1.0 - gate) * h
        mixed = self.norm(mixed)
        seq = mixed.view(batch, nodes, steps, channels).permute(0, 2, 1, 3).contiguous()
        last = seq[:, -1]
        return last, seq


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, seq):
        weights = torch.softmax(self.score(seq), dim=1)
        return torch.sum(weights * seq, dim=1), weights


class EnhancedBiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = BiLSTMAttentionBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pool = TemporalAttentionPooling(input_dim)

    def forward(self, x):
        last, seq = self.encoder(x)
        batch, steps, nodes, channels = seq.shape
        pooled_in = seq.permute(0, 2, 1, 3).contiguous().view(batch * nodes, steps, channels)
        pooled, weights = self.pool(pooled_in)
        pooled = pooled.view(batch, nodes, channels)
        return 0.5 * (last + pooled), seq
