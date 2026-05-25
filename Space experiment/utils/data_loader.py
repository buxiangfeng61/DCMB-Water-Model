from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_scaler: Standardizer
    target_scaler: Standardizer
    site_order: List[str]
    feature_cols: List[str]
    timestamps: List


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def infer_feature_columns(frame, time_col, site_col, target_col):
    excluded = {time_col, site_col, target_col}
    return [col for col in frame.columns if col not in excluded]


def load_table(path, time_col, site_col, target_col, feature_cols=None, strict_missing=True):
    frame = pd.read_csv(path)
    if feature_cols is None:
        feature_cols = infer_feature_columns(frame, time_col, site_col, target_col)
    required = [time_col, site_col] + list(feature_cols) + [target_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = frame[required].copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame[site_col] = frame[site_col].astype(str)
    for col in feature_cols + [target_col]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if strict_missing:
        frame = frame.dropna(axis=0, how="any").copy()
    return frame, list(feature_cols)


def table_to_arrays(frame, time_col, site_col, feature_cols, target_col):
    site_order = sorted(frame[site_col].astype(str).unique().tolist())
    timestamps = sorted(frame[time_col].unique().tolist())
    time_index = pd.Index(timestamps, name=time_col)
    site_index = pd.Index(site_order, name=site_col)
    features = []
    for col in feature_cols:
        pivot = frame.pivot_table(index=time_col, columns=site_col, values=col, aggfunc="mean")
        pivot = pivot.reindex(index=time_index, columns=site_index)
        features.append(pivot.to_numpy(dtype=np.float32)[..., None])
    feature_array = np.concatenate(features, axis=-1)
    target_pivot = frame.pivot_table(index=time_col, columns=site_col, values=target_col, aggfunc="mean")
    target_pivot = target_pivot.reindex(index=time_index, columns=site_index)
    target_array = target_pivot.to_numpy(dtype=np.float32)[..., None]
    return feature_array, target_array, timestamps, site_order


def fit_standardizer(array):
    mean = np.nanmean(array, axis=(0, 1), keepdims=True)
    std = np.nanstd(array, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def make_windows(features, targets, seq_len, pred_len, start_index, end_index):
    xs = []
    ys = []
    for target_start in range(start_index, end_index - pred_len + 1):
        source_start = target_start - seq_len
        source_end = target_start
        if source_start < 0:
            continue
        x = features[source_start:source_end]
        y = targets[target_start:target_start + pred_len]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        y = np.transpose(y, (1, 0, 2)).squeeze(-1)
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    if len(xs) == 0:
        raise ValueError("No valid windows were created")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_indices(total_steps, train_ratio, val_ratio, seq_len, pred_len):
    train_pool_end = int(total_steps * train_ratio)
    val_size = int(train_pool_end * val_ratio)
    train_end = max(seq_len + pred_len + 1, train_pool_end - val_size)
    val_end = train_pool_end
    if val_end <= train_end:
        val_end = min(total_steps - pred_len, train_end + max(pred_len + 1, 1))
    if total_steps - val_end < pred_len + 1:
        raise ValueError("Not enough time steps for the test split")
    return train_end, val_end


def build_dataloaders(config):
    frame, feature_cols = load_table(
        path=config.data_path,
        time_col=config.time_col,
        site_col=config.site_col,
        target_col=config.target_col,
        feature_cols=config.feature_cols,
        strict_missing=config.strict_missing,
    )
    features, targets, timestamps, site_order = table_to_arrays(
        frame=frame,
        time_col=config.time_col,
        site_col=config.site_col,
        feature_cols=feature_cols,
        target_col=config.target_col,
    )
    total_steps = features.shape[0]
    train_end, val_end = split_indices(total_steps, config.train_ratio, config.val_ratio, config.seq_len, config.pred_len)
    feature_scaler = fit_standardizer(features[:train_end])
    target_scaler = fit_standardizer(targets[:train_end])
    features_scaled = feature_scaler.transform(features)
    targets_scaled = target_scaler.transform(targets) if config.standardize_target else targets.copy()
    train_x, train_y = make_windows(features_scaled, targets_scaled, config.seq_len, config.pred_len, config.seq_len, train_end)
    val_x, val_y = make_windows(features_scaled, targets_scaled, config.seq_len, config.pred_len, train_end, val_end)
    test_x, test_y = make_windows(features_scaled, targets_scaled, config.seq_len, config.pred_len, val_end, total_steps)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        WindowDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        WindowDataset(val_x, val_y),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        WindowDataset(test_x, test_y),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        site_order=site_order,
        feature_cols=feature_cols,
        timestamps=timestamps,
    )


def inverse_target(y, target_scaler, standardize_target=True):
    if isinstance(y, torch.Tensor):
        array = y.detach().cpu().numpy()
    else:
        array = np.asarray(y)
    if not standardize_target:
        return array
    converted = np.transpose(array, (0, 2, 1))[..., None]
    restored = target_scaler.inverse_transform(converted)
    return np.transpose(restored.squeeze(-1), (0, 2, 1))
