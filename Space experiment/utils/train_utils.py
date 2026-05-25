from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
import numpy as np
import torch
import torch.nn as nn

from .data_loader import inverse_target
from .metrics import compute_metrics


@dataclass
class EarlyStopping:
    patience: int = 10
    best_value: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False

    def update(self, value):
        improved = value < self.best_value
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return improved


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


def config_to_dict(config):
    if is_dataclass(config):
        return asdict(config)
    return dict(config.__dict__)


def run_one_epoch(model, loader, graphs, config, optimizer=None):
    train = optimizer is not None
    model.train(mode=train)
    criterion = nn.MSELoss()
    loss_meter = AverageMeter()
    all_true = []
    all_pred = []
    for x, y in loader:
        x = x.to(config.device)
        y = y.to(config.device)
        with torch.set_grad_enabled(train):
            pred = model(x, graphs.adj_flow, graphs.adj_dist)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip is not None and config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
        loss_meter.update(loss.item(), x.shape[0])
        pred_inv = inverse_target(pred, loader.dataset_target_scaler if hasattr(loader, "dataset_target_scaler") else config._target_scaler, config.standardize_target)
        true_inv = inverse_target(y, loader.dataset_target_scaler if hasattr(loader, "dataset_target_scaler") else config._target_scaler, config.standardize_target)
        all_pred.append(pred_inv)
        all_true.append(true_inv)
    metrics = compute_metrics(np.concatenate(all_true, axis=0), np.concatenate(all_pred, axis=0))
    return loss_meter.avg, metrics


def attach_target_scaler(loaders, scaler):
    for loader in loaders:
        loader.dataset_target_scaler = scaler
    return loaders


def save_checkpoint(path, model, optimizer, config, epoch, metrics):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config_to_dict(config),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def format_metrics(prefix, metrics):
    return f"{prefix}_MSE={metrics['MSE']:.6f} {prefix}_MAE={metrics['MAE']:.6f} {prefix}_RMSE={metrics['RMSE']:.6f} {prefix}_R2={metrics['R2']:.6f}"
