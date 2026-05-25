import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScan(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u, delta, B_t, C_t):
        channels = u.shape[-1]
        A = -torch.exp(self.A_log).view(1, 1, channels)
        D = self.D.view(1, 1, channels)
        decay = torch.exp(torch.clamp(delta * A, min=-20.0, max=0.0))
        drive = delta * B_t * u
        prefix_decay = torch.cumprod(torch.clamp(decay, min=1e-6, max=1.0), dim=1)
        normalized_drive = drive / torch.clamp(prefix_decay, min=1e-6)
        state = prefix_decay * torch.cumsum(normalized_drive, dim=1)
        return C_t * state + D * u


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.1):
        super().__init__()
        inner_dim = d_model * expand
        self.d_model = d_model
        self.inner_dim = inner_dim
        self.input_projection = nn.Linear(d_model, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=inner_dim,
            bias=True,
        )
        self.delta_projection = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Softplus(),
        )
        self.B_projection = nn.Linear(inner_dim, inner_dim)
        self.C_projection = nn.Linear(inner_dim, inner_dim)
        self.scan = SelectiveScan(inner_dim)
        self.output_projection = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        z = self.input_projection(x)
        u, gate = torch.chunk(z, 2, dim=-1)
        u = self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        gate = F.silu(gate)
        delta = self.delta_projection(u)
        B_t = self.B_projection(u)
        C_t = self.C_projection(u)
        y = self.scan(u, delta, B_t, C_t)
        y = y * gate
        y = self.output_projection(y)
        y = self.dropout(y)
        y = self.norm(residual + y)
        return y


class MambaTemporalBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.1, layers=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(layers)
        ])

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("MambaTemporalBlock expects x with shape [B, T, N, D]")
        batch, steps, nodes, channels = x.shape
        h = x.permute(0, 2, 1, 3).contiguous().view(batch * nodes, steps, channels)
        for block in self.blocks:
            h = block(h)
        seq = h.view(batch, nodes, steps, channels).permute(0, 2, 1, 3).contiguous()
        last = seq[:, -1]
        return last, seq


class GatedTemporalMixer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.left = nn.Linear(d_model, d_model)
        self.right = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = torch.tanh(self.left(x)) * torch.sigmoid(self.right(x))
        y = self.out(y)
        y = self.dropout(y)
        return self.norm(x + y)
