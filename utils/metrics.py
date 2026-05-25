import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(y_true, y_pred):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true, y_pred):
    y_true = to_numpy(y_true).reshape(-1)
    y_pred = to_numpy(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true, y_pred):
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        self.y_true.append(to_numpy(y_true))
        self.y_pred.append(to_numpy(y_pred))

    def compute(self):
        if len(self.y_true) == 0:
            return {"MSE": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
        return compute_metrics(np.concatenate(self.y_true, axis=0), np.concatenate(self.y_pred, axis=0))
