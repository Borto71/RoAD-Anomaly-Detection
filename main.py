#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

# --- import your model ---
from models.cnn_1d import RoADCNN1D as CNN1D


# -------------------------
# Reproducibility helpers
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Data loading utilities
# -------------------------
def list_csv_files(root: Path) -> List[Path]:
    """Recursively list all CSV files under root (data/dataset/<subset>/*.csv)."""
    return sorted(root.rglob("*.csv"))


def read_csv_return_features_labels(
    csv_path: Path, label_col: str = "anomaly"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read a CSV and return (X, y, feature_names).

    - If label_col exists, y is that column, binarized to {0,1}.
    - If label_col is absent, y is zeros.
    - Features NEVER include label_col.
    """
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    if label_col in cols:
        feature_cols = [c for c in cols if c != label_col]
        y_raw = df[label_col].astype(float).values
        y = (y_raw > 0.5).astype(np.float32)
    else:
        feature_cols = cols
        y = np.zeros(len(df), dtype=np.float32)

    X = df[feature_cols].values.astype(np.float32)
    return X, y, feature_cols


def fit_scaler_on_train_files(
    train_files: List[Path], feature_cols_ref: List[str], label_col: str = "anomaly"
) -> StandardScaler:
    """
    Fit StandardScaler on TRAIN files only (concatenate along time),
    only on features (no label).
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    chunks = []
    for f in tqdm(train_files, desc="Fitting scaler on train"):
        X, _, feat_cols = read_csv_return_features_labels(f, label_col=label_col)
        assert (
            feat_cols == feature_cols_ref
        ), f"Feature columns mismatch in {f.name}"
        chunks.append(X)
    X_all = np.concatenate(chunks, axis=0)
    scaler.fit(X_all)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


# -------------------------
# Windowed Dataset (with RAM cache)
# -------------------------
class WindowedSeqDataset(Dataset):
    """
    Convert per-file time series into overlapping windows (C, T) plus a window label.

    Label policy (NUOVA):
      - Calcola la percentuale di frame anomali nella finestra.
      - window_label = 1 se percentuale >= min_anomaly_ratio, altrimenti 0.
      - Se il file non ha etichette, la finestra è 0 (normale).

    Questo evita che una singola anomalia "contamini" centinaia di finestre.
    """

    def __init__(
        self,
        files: List[Path],
        scaler: StandardScaler,
        feature_cols: List[str],
        label_col: str = "anomaly",
        window_size: int = 128,
        step_size: int = 64,
        min_anomaly_ratio: float = 0.2,
        progress_desc: str = "Indexing windows",
        cache_in_ram: bool = True,
    ):
        self.files = files
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.step_size = step_size
        self.min_anomaly_ratio = min_anomaly_ratio
        self.cache_in_ram = cache_in_ram

        self.index: List[Tuple[int, int, int]] = []  # (file_idx, start, end)
        self._cache_meta: List[Tuple[int, int]] = []  # (n_frames, n_feats) per file
        self._data_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}

        total_windows = 0
        for i, f in enumerate(tqdm(self.files, desc=progress_desc)):
            X, y, feat_cols = read_csv_return_features_labels(
                f, label_col=self.label_col
            )
            assert (
                feat_cols == self.feature_cols
            ), f"Feature columns mismatch in {f.name}"

            # Pre-cache scaled data once per file (if enabled)
            if self.cache_in_ram:
                X_scaled = apply_scaler(X, self.scaler)
                self._data_cache[f] = (X_scaled, y)

            n = X.shape[0]
            self._cache_meta.append((n, X.shape[1]))
            if n < self.window_size:
                continue

            starts = range(0, n - self.window_size + 1, self.step_size)
            for s in starts:
                e = s + self.window_size
                self.index.append((i, s, e))
                total_windows += 1

        tqdm.write(
            f"{progress_desc}: indexed {total_windows} windows from {len(self.files)} files."
        )
        # Group = file path
        self.groups = np.array(
            [self.files[i].as_posix() for (i, _, _) in self.index]
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        file_idx, s, e = self.index[idx]
        f = self.files[file_idx]

        if self.cache_in_ram and f in self._data_cache:
            X_full, y_full = self._data_cache[f]
        else:
            X_full, y_full, _ = read_csv_return_features_labels(
                f, label_col=self.label_col
            )
            X_full = apply_scaler(X_full, self.scaler)

        Xw = X_full[s:e]  # (T, C)

        if len(y_full) >= e:
            yw = y_full[s:e]
        else:
            yw = np.zeros(e - s, dtype=np.float32)

        # percentuale di frame anomali nella finestra
        if len(yw) > 0:
            anomaly_ratio = float(np.mean(yw > 0.5))
        else:
            anomaly_ratio = 0.0

        label = float(anomaly_ratio >= self.min_anomaly_ratio)

        # (T, C) -> (C, T) per Conv1d
        Xw = torch.from_numpy(Xw.astype(np.float32)).permute(1, 0)
        y = torch.tensor(label, dtype=torch.float32)
        return Xw, y


# -------------------------
# Training & evaluation
# -------------------------
def train_one_epoch(
    model, loader, criterion, optimizer, device, amp_scaler
) -> Dict[str, float]:
    model.train()
    losses = []

    for X, y in tqdm(loader, desc="Train", leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits, _ = model(X)  # expects (B, 1)
            logits = logits.squeeze(-1)  # (B,)
            loss = criterion(logits, y)

        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        losses.append(loss.item())

    return {"loss": float(np.mean(losses)) if losses else float("nan")}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    losses = []
    y_true_all, y_score_all = [], []

    for X, y in tqdm(loader, desc="Val", leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits, _ = model(X)
            logits = logits.squeeze(-1)
            loss = criterion(logits, y)

        losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_score_all.append(probs)
        y_true_all.append(y.detach().cpu().numpy())

    if not y_true_all:
        return {
            "loss": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "acc@0.5": float("nan"),
            "prec@0.5": float("nan"),
            "rec@0.5": float("nan"),
            "f1@0.5": float("nan"),
            "best_f1": float("nan"),
            "best_th": 0.5,
            "acc_class0": float("nan"),
            "acc_class1": float("nan"),
            "support_class0": 0,
            "support_class1": 0,
            "cm_tn": 0,
            "cm_fp": 0,
            "cm_fn": 0,
            "cm_tp": 0,
        }

    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)

    # Best threshold (for info)
    best_f1, best_th = -1.0, 0.5
    threshs = np.linspace(0.05, 0.95, 19)
    for th in threshs:
        y_pred_th = (y_score >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_th, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_th = f1, th

    # Fixed threshold 0.5
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    def _safe_metric(fn, *args, **kwargs):
        try:
            return float(fn(*args, **kwargs))
        except Exception:
            return float("nan")

    auroc = _safe_metric(roc_auc_score, y_true, y_score)
    auprc = _safe_metric(average_precision_score, y_true, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        for i, row_label in enumerate([0, 1][: cm.shape[0]]):
            for j, col_label in enumerate([0, 1][: cm.shape[1]]):
                if row_label == 0 and col_label == 0:
                    tn = cm[i, j]
                elif row_label == 0 and col_label == 1:
                    fp = cm[i, j]
                elif row_label == 1 and col_label == 0:
                    fn = cm[i, j]
                elif row_label == 1 and col_label == 1:
                    tp = cm[i, j]

    support_class0 = int(tn + fp)
    support_class1 = int(tp + fn)

    acc_class0 = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    acc_class1 = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auroc": auroc,
        "auprc": auprc,
        "acc@0.5": float(acc),
        "prec@0.5": float(prec),
        "rec@0.5": float(rec),
        "f1@0.5": float(f1),
        "best_f1": float(best_f1),
        "best_th": float(best_th),
        "acc_class0": acc_class0,
        "acc_class1": acc_class1,
        "support_class0": support_class0,
        "support_class1": support_class1,
        "cm_tn": int(tn),
        "cm_fp": int(fp),
        "cm_fn": int(fn),
        "cm_tp": int(tp),
    }


# -------------------------
# Orchestration
# -------------------------
def main():
    seed_everything(42)

    # GPU preferences
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("Using CPU (no CUDA available)")

    data_root = Path("data/dataset")
    assert data_root.exists(), "data/dataset folder not found. Export CSVs first."

    # 1) Discover files and reference feature order from FIRST file
    all_csv = list_csv_files(data_root)
    assert len(all_csv) > 0, "No CSV found under data/dataset/"
    X0, _, feature_cols = read_csv_return_features_labels(
        all_csv[0], label_col="anomaly"
    )
    n_features = X0.shape[1]
    tqdm.write(f"Discovered {len(all_csv)} CSV files. Detected {n_features} features.")

    # 2) Split at FILE level with stratification by presence of anomaly
    def file_has_anomaly(p: Path) -> int:
        _, y, _ = read_csv_return_features_labels(p, "anomaly")
        return int(np.any(y > 0.5))

    groups = np.array([p.as_posix() for p in all_csv])
    strat = np.array(
        [file_has_anomaly(p) for p in tqdm(all_csv, desc="Scanning anomaly presence")]
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(all_csv, strat, groups))
    train_files = [all_csv[i] for i in train_idx]
    val_files = [all_csv[i] for i in val_idx]
    tqdm.write(
        f"Split -> train files: {len(train_files)}, val files: {len(val_files)}"
    )

    # 3) Fit scaler ONLY on train files; save it
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True, parents=True)
    scaler = fit_scaler_on_train_files(
        train_files, feature_cols, label_col="anomaly"
    )

    import joblib

    joblib.dump(
        {"scaler": scaler, "feature_cols": feature_cols},
        artifacts / "scaler.joblib",
    )
    tqdm.write("Scaler saved to artifacts/scaler.joblib")

    # 4) Build datasets / loaders (windowed) — caching ON
    window_size = 128
    step_size = 64
    min_anomaly_ratio = 0.2  # almeno il 20% di frame anomali nella finestra
    batch_size = 256

    ds_train = WindowedSeqDataset(
        train_files,
        scaler,
        feature_cols,
        label_col="anomaly",
        window_size=window_size,
        step_size=step_size,
        min_anomaly_ratio=min_anomaly_ratio,
        progress_desc="Indexing windows (train)",
        cache_in_ram=True,
    )
    ds_val = WindowedSeqDataset(
        val_files,
        scaler,
        feature_cols,
        label_col="anomaly",
        window_size=window_size,
        step_size=step_size,
        min_anomaly_ratio=min_anomaly_ratio,
        progress_desc="Indexing windows (val)",
        cache_in_ram=True,
    )
    tqdm.write(f"Train windows: {len(ds_train)}, Val windows: {len(ds_val)}")

    # 5) Compute class imbalance on TRAIN windows for pos_weight
    y_train_windows = []
    for i in tqdm(range(len(ds_train)), desc="Computing class balance (train windows)"):
        _, y = ds_train[i]
        y_train_windows.append(float(y.item()))
    y_train_windows = np.array(y_train_windows)
    n_pos = float((y_train_windows > 0.5).sum())
    n_neg = float((y_train_windows <= 0.5).sum())
    pos_weight_value = max(n_neg / (n_pos + 1e-8), 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
    tqdm.write(
        f"Class balance (train windows) -> pos: {int(n_pos)}, "
        f"neg: {int(n_neg)}, pos_weight: {pos_weight.item():.2f}"
    )

    num_workers = min(12, os.cpu_count() or 4)
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # 6) Model / loss / optimizer / scheduler
    model = CNN1D(in_channels=n_features, base_channels=64, dropout=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 7) Training loop
    num_epochs = 30
    best_val = -1.0
    best_ckpt = artifacts / "best_model.pt"
    history = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tqdm.write(f"\nEpoch {epoch}/{num_epochs}")

        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, amp_scaler)
        ev = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(ev["loss"])

        entry = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": ev["loss"],
            "val_auroc": ev["auroc"],
            "val_auprc": ev["auprc"],
            "val_acc@0.5": ev["acc@0.5"],
            "val_prec@0.5": ev["prec@0.5"],
            "val_rec@0.5": ev["rec@0.5"],
            "val_f1@0.5": ev["f1@0.5"],
            "val_best_f1": ev["best_f1"],
            "val_best_th": ev["best_th"],
            "val_acc_class0": ev["acc_class0"],
            "val_acc_class1": ev["acc_class1"],
            "val_support_class0": ev["support_class0"],
            "val_support_class1": ev["support_class1"],
            "val_tn": ev["cm_tn"],
            "val_fp": ev["cm_fp"],
            "val_fn": ev["cm_fn"],
            "val_tp": ev["cm_tp"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": round(time.time() - t0, 2),
        }
        history.append(entry)

        tqdm.write(
            f"[Epoch {epoch:02d}] "
            f"train_loss={entry['train_loss']:.4f} "
            f"val_loss={entry['val_loss']:.4f} "
            f"auprc={entry['val_auprc']:.4f} "
            f"f1@0.5={entry['val_f1@0.5']:.4f} "
            f"best_f1={entry['val_best_f1']:.4f} th*={entry['val_best_th']:.2f} "
            f"acc0={entry['val_acc_class0']:.4f} "
            f"acc1={entry['val_acc_class1']:.4f} "
            f"lr={entry['lr']:.2e} "
            f"time={entry['epoch_time_sec']:.1f}s"
        )

        # Model selection: prefer AUPRC (imbalance) with fallback to best F1
        auprc = 0.0 if math.isnan(ev["auprc"]) else ev["auprc"]
        score = 0.7 * auprc + 0.3 * ev["best_f1"]
        if score > best_val:
            best_val = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_channels": n_features,
                    "window_size": window_size,
                    "feature_cols": feature_cols,
                    "best_val_score": best_val,
                    "epoch": epoch,
                },
                best_ckpt,
            )
            tqdm.write(
                f"  ↳ Saved new best checkpoint to {best_ckpt} (score={best_val:.4f})"
            )

        pd.DataFrame(history).to_csv(artifacts / "training_log.csv", index=False)
        with open(artifacts / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)

    # 8) Save final weights as well
    torch.save(model.state_dict(), artifacts / "last_model_weights.pt")
    tqdm.write("\nTraining finished.")
    tqdm.write(f"- Best model: {best_ckpt}")
    tqdm.write(f"- Logs: {artifacts/'training_log.csv'} and {artifacts/'training_log.json'}")
    tqdm.write(f"- Scaler: {artifacts/'scaler.joblib'}")


if __name__ == "__main__":
    main()
