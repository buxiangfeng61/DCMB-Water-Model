import numpy as np
import torch


EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value)))


def build_distance_adjacency(coordinates, threshold_km=85.0):
    coordinates = list(coordinates)
    n = len(coordinates)
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = haversine_distance_km(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
            if dist <= threshold_km:
                adjacency[i, j] = 1.0
    return adjacency


def build_hydrological_adjacency(num_nodes, edges, undirected=True):
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        adjacency[int(i), int(j)] = 1.0
        if undirected:
            adjacency[int(j), int(i)] = 1.0
    return adjacency


def normalize_adjacency(adjacency, add_self_loops=True):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    matrix = adjacency.copy()
    if add_self_loops:
        matrix = matrix + np.eye(matrix.shape[0], dtype=np.float32)
    degree = matrix.sum(axis=1)
    degree = np.clip(degree, 1e-12, None)
    inv_sqrt = np.power(degree, -0.5)
    norm = inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    return norm.astype(np.float32)


class GraphBundle:
    def __init__(self, adj_flow, adj_dist):
        self.adj_flow = adj_flow
        self.adj_dist = adj_dist

    def to(self, device):
        self.adj_flow = self.adj_flow.to(device)
        self.adj_dist = self.adj_dist.to(device)
        return self


def load_graphs(adj_distance_path, adj_hydrological_path, device="cpu", add_self_loops=True):
    adj_dist = np.load(adj_distance_path, allow_pickle=False)
    adj_flow = np.load(adj_hydrological_path, allow_pickle=False)
    if adj_dist.shape != adj_flow.shape:
        raise ValueError("adj_distance and adj_hydrological must have the same shape")
    adj_dist = normalize_adjacency(adj_dist, add_self_loops=add_self_loops)
    adj_flow = normalize_adjacency(adj_flow, add_self_loops=add_self_loops)
    adj_dist = torch.tensor(adj_dist, dtype=torch.float32, device=device)
    adj_flow = torch.tensor(adj_flow, dtype=torch.float32, device=device)
    return GraphBundle(adj_flow=adj_flow, adj_dist=adj_dist)


def save_adjacency(path, adjacency):
    np.save(path, np.asarray(adjacency, dtype=np.float32))


def validate_graph_shape(adjacency, num_nodes):
    adjacency = np.asarray(adjacency)
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(f"Expected adjacency shape {(num_nodes, num_nodes)}, got {adjacency.shape}")
    return True
