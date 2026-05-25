import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activation=True, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adjacency):
        if x.dim() != 3:
            raise ValueError("GraphConvolution expects x with shape [B, N, C]")
        if adjacency.dim() != 2:
            raise ValueError("GraphConvolution expects adjacency with shape [N, N]")
        support = torch.einsum("ij,bjc->bic", adjacency, x)
        out = torch.matmul(support, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.activation:
            out = F.gelu(out)
        out = self.dropout(out)
        return out


class CrossGraphLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.flow_graph = GraphConvolution(in_dim, hidden_dim, dropout=dropout, activation=True)
        self.dist_graph = GraphConvolution(in_dim, hidden_dim, dropout=dropout, activation=True)
        self.flow_norm = nn.LayerNorm(hidden_dim)
        self.dist_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_flow, h_dist, adj_flow, adj_dist):
        next_flow = self.flow_graph(h_flow, adj_flow)
        next_dist = self.dist_graph(h_dist, adj_dist)
        next_flow = self.flow_norm(next_flow)
        next_dist = self.dist_norm(next_dist)
        return next_flow, next_dist


class DCGCN(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers = []
        layers.append(CrossGraphLayer(hidden_dim, hidden_dim, dropout=dropout))
        for _ in range(1, num_layers):
            layers.append(CrossGraphLayer(hidden_dim * 2, hidden_dim, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.cross_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x, adj_flow, adj_dist):
        h_flow = x
        h_dist = x
        history_flow = []
        history_dist = []
        for index, layer in enumerate(self.layers):
            if index == 0:
                h_flow, h_dist = layer(h_flow, h_dist, adj_flow, adj_dist)
            else:
                z_flow = torch.cat([h_flow, h_dist], dim=-1)
                z_dist = torch.cat([h_dist, h_flow], dim=-1)
                h_flow, h_dist = layer(z_flow, z_dist, adj_flow, adj_dist)
            history_flow.append(h_flow)
            history_dist.append(h_dist)
        h_pair = torch.cat([h_flow, h_dist], dim=-1)
        gate = self.cross_gate(h_pair)
        gated = h_pair * gate
        out = self.output_projection(gated)
        return out


class StackedDCGCN(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1, blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([DCGCN(hidden_dim, num_layers=num_layers, dropout=dropout) for _ in range(blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(blocks)])

    def forward(self, x, adj_flow, adj_dist):
        h = x
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + block(h, adj_flow, adj_dist))
        return h
