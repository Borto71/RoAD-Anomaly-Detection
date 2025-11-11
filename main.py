import os
import glob
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from models.cnn_1d import build_model


# Utilities

def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across numpy, random, and torch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_csvs(folder: str) -> List[str]:
    """Return all CSV file paths in a given folder."""
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def load_csv(path: str) -> np.ndarray:
    """Load CSV without header into a float32 numpy array."""
    df = pd.read_csv(path, header = None)
    return df.to_numpy(dtype = np.float32)


def windowed_views(
    X: np.ndarray, y: Optional[np.ndarray], win: int, hop: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Slice a (T, C) array into overlapping windows of size `win` and stride `hop`."""
    n = X.shape[0]
    if n < win:
        empty_X = np.empty((0, win, X.shape[1]), dtype = np.float32)
        empty_y = None if y is None else np.empty((0,), dtype = np.float32)
        return empty_X, empty_y

    starts = np.arange(0, n - win + 1, hop, dtype = np.int32)
    Xw = np.stack([X[s : s + win] for s in starts], axis = 0)

    if y is None:
        yw = None
    else:
        yw = np.stack([y[s : s + win].max() for s in starts], axis = 0).astype(np.float32)

    return Xw, yw


def fit_minmax_on_training(training_csvs: List[str]) -> MinMaxScaler:
    """Fit a MinMaxScaler on all training files without loading them entirely in memory."""
    scaler = MinMaxScaler()
    for p in training_csvs:
        arr = load_csv(p)
        feats = arr[:, :86] if arr.shape[1] >= 86 else arr
        scaler.partial_fit(feats)
    return scaler


def summarize_labels(y: np.ndarray, name: str) -> None:
    """Print label distribution summary for quick inspection."""
    if y.size == 0:
        print(f"{name}: empty")
        return
    uniq, cnts = np.unique(y, return_counts=True)
    print(f"{name}: total = {y.size}  labels = {dict(zip(uniq.tolist(), cnts.tolist()))}")


# Dataset

class RoADWindows(Dataset):
    """
    Sliding-window dataset for the RoAD multivariate time series.

    - Features: first 86 columns.
    - Label: column 87 if available, binarized to {0,1}.
    - Returns (C, T) tensors ready for 1D CNN input.
    """

    def __init__(
        self,
        csv_paths: List[str],
        win: int,
        hop: int,
        scaler: Optional[MinMaxScaler] = None,
        assume_normal_if_no_label: bool = True,
    ):
        self.win = win
        self.hop = hop
        self.scaler = scaler
        self.assume_normal_if_no_label = assume_normal_if_no_label

        Xw_list, yw_list = [], []

        for p in csv_paths:
            arr = load_csv(p)
            if arr.shape[1] < 86:
                continue

            feats = arr[:, :86].astype(np.float32)
            lbls = arr[:, 86].astype(np.float32) if arr.shape[1] >= 87 else None
            if lbls is not None:
                lbls = (lbls > 0).astype(np.float32)

            if self.scaler is not None:
                feats = self.scaler.transform(feats)

            Xw, yw = windowed_views(feats, lbls, win = self.win, hop = self.hop)
            if Xw.shape[0] == 0:
                continue

            if yw is None:
                if not self.assume_normal_if_no_label:
                    continue
                yw = np.zeros((Xw.shape[0],), dtype = np.float32)

            Xw_list.append(Xw)
            yw_list.append(yw)

        if len(Xw_list) == 0:
            self.X = np.empty((0, 86, self.win), dtype = np.float32)
            self.y = np.empty((0,), dtype = np.float32)
        else:
            X = np.concatenate(Xw_list, axis = 0)
            y = np.concatenate(yw_list, axis = 0)
            self.X = np.transpose(X, (0, 2, 1)).astype(np.float32)
            self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype = torch.float32)


# Positive samples dataset

class _InMemoryWindows(Dataset):
    """Simple wrapper for in-memory feature and label arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): 
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype = torch.float32)


def build_positive_ds(
    subset_paths: Dict[str, str],
    window: int,
    hop: int,
    scaler: Optional[MinMaxScaler],
) -> Optional[Dataset]:
    """Collect only windows with label = 1 from all anomaly subsets."""
    pos_X, pos_y = [], []
    for name, path in subset_paths.items():
        csvs = list_csvs(path)
        if not csvs:
            continue
        ds = RoADWindows(
            csv_paths = csvs, win = window, hop = hop, scaler = scaler,
            assume_normal_if_no_label = False
        )
        if len(ds) == 0:
            continue
        mask = (ds.y == 1.0)
        if mask.any():
            pos_X.append(ds.X[mask])
            pos_y.append(ds.y[mask])
    if not pos_X:
        return None
    X = np.concatenate(pos_X, axis = 0)
    y = np.concatenate(pos_y, axis = 0)
    summarize_labels(y, "train(positives)")
    return _InMemoryWindows(X, y)


# Training and evaluation

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler_amp: Optional[torch.amp.GradScaler],
    criterion: nn.Module,
) -> float:
    """Single training epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad(set_to_none = True)

        if scaler_amp is not None and device.type == "cuda":
            with torch.amp.autocast("cuda", enabled = True):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    subset_loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, dict]:
    """Compute window-level metrics on anomaly subsets."""
    model.eval()
    results: Dict[str, dict] = {}

    for name, loader in subset_loaders.items():
        all_logits, all_labels = [], []

        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            all_logits.append(logits.squeeze(1).cpu().numpy())
            all_labels.append(y.numpy())

        if len(all_logits) == 0:
            results[name] = {"samples": 0}
            continue

        scores = np.concatenate(all_logits, axis = 0)
        labels = np.concatenate(all_labels, axis = 0).astype(np.float32)
        unique = np.unique(labels)

        # Handle single-class subsets gracefully
        if unique.size < 2:
            results[name] = {
                "samples": int(labels.shape[0]),
                "ROC-AUC": None,
                "PR-AUC": None,
                "BestF1": None,
                "BestThresh": None,
                "F1@0.5": None,
                "note": f"single-class labels in {name} (labels = {unique.tolist()})",
            }
            continue

        roc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))

        ps, rs, ths = precision_recall_curve(labels, scores)
        f1s = 2 * ps * rs / (ps + rs + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        best_f1 = float(f1s[best_idx])
        best_th = float(ths[best_idx - 1]) if 0 < best_idx < len(ths) else 0.0

        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
        preds_05 = (probs >= 0.5).astype(np.float32)
        f1_at_05 = float(f1_score(labels, preds_05, zero_division=0))

        results[name] = {
            "samples": int(labels.shape[0]),
            "ROC-AUC": roc,
            "PR-AUC": pr_auc,
            "BestF1": best_f1,
            "BestThresh": best_th,
            "F1@0.5": f1_at_05,
        }

    return results


# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = "data/dataset")
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--epochs", type = int, default = 20)
    parser.add_argument("--window", type = int, default = 512)
    parser.add_argument("--hop", type = int, default = 128)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--weight_decay", type = float, default = 1e-4)
    parser.add_argument("--save_path", type = str, default = "road_cnn.pt")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--no_cuda", action = "store_true")
    parser.add_argument("--no_amp", action = "store_true")
    parser.add_argument("--no_fit_scaler", action = "store_true")
    parser.add_argument("--val_split", type = float, default = 0.1)
    parser.add_argument("--pos_frac", type = float, default = 0.2,
                        help = "Approximate fraction of positive samples per epoch.")
    parser.add_argument("--pos_weight", type = float, default = None,
                        help="Manual pos_weight for BCEWithLogitsLoss.")
    parser.add_argument("--auto_pos_weight", action = "store_true",
                        help = "Automatically compute pos_weight = n_neg/n_pos.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    # Collect CSVs
    train_dir = os.path.join(args.data_dir, "training")
    train_csvs = list_csvs(train_dir)
    if len(train_csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")

    anomaly_subsets = ["collision", "control", "weight", "velocity"]
    subset_paths = {s: os.path.join(args.data_dir, s) for s in anomaly_subsets}

    # Fit MinMaxScaler
    scaler = None
    if not args.no_fit_scaler:
        scaler = fit_minmax_on_training(train_csvs)

    # Normal training data
    full_train_ds = RoADWindows(train_csvs, args.window, args.hop, scaler, assume_normal_if_no_label=True)
    n = len(full_train_ds)
    perm = np.random.permutation(n)
    val_n = max(1, int(args.val_split * n))
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]

    train_norm_ds = torch.utils.data.Subset(full_train_ds, tr_idx.tolist())
    val_subset = torch.utils.data.Subset(full_train_ds, val_idx.tolist())

    # Build positive dataset
    pos_train_ds = build_positive_ds(subset_paths, args.window, args.hop, scaler)

    # Merge negatives and positives
    if pos_train_ds is not None and len(pos_train_ds) > 0:
        mixed_train = ConcatDataset([train_norm_ds, pos_train_ds])
        n_neg = len(train_norm_ds)
        n_pos = len(pos_train_ds)
        w_neg = (1.0 - args.pos_frac) / max(1, n_neg)
        w_pos = args.pos_frac / max(1, n_pos)
        weights = np.concatenate([
            np.full(n_neg, w_neg, dtype=np.float32),
            np.full(n_pos, w_pos, dtype=np.float32)
        ])
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples = n_neg + n_pos, replacement = True)
        train_loader = DataLoader(mixed_train, batch_size = args.batch_size, sampler = sampler, drop_last = False)
    else:
        print("WARNING: No positive windows found; training on normals only.")
        train_loader = DataLoader(train_norm_ds, batch_size=args.batch_size, shuffle = True, drop_last = False)

    val_loader = DataLoader(val_subset, batch_size = args.batch_size, shuffle = False, drop_last = False)

    # Compute pos_weight if requested
    pos_weight_tensor = None
    if args.pos_weight is not None:
        pos_weight_tensor = torch.tensor([float(args.pos_weight)], device=device)
    elif args.auto_pos_weight and pos_train_ds is not None and len(pos_train_ds) > 0:
        n_neg = len(train_norm_ds)
        n_pos = len(pos_train_ds)
        ratio = max(1.0, n_neg / max(1, n_pos))
        pos_weight_tensor = torch.tensor([ratio], device=device)
        print(f"Auto pos_weight = {ratio:.2f}")

    # Model and optimizer
    model = build_model(in_channels = 86, base_channels = 64, dropout = 0.1, return_frame_logits = False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    scaler_amp = torch.amp.GradScaler("cuda", enabled=(not args.no_amp and device.type == "cuda"))

    best_val = float("inf")
    best_state = None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler_amp, criterion)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                logits, _ = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if len(val_losses) else train_loss
        scheduler.step(val_loss)

        print(f"[Epoch {epoch:03d}] train_loss = {train_loss:.6f}  val_loss = {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "scaler": scaler,
                "config": vars(args),
                "epoch": epoch,
            }

    if best_state is not None:
        torch.save(best_state, args.save_path)
        print(f"Saved best model to: {args.save_path}")

    # Evaluation on anomaly subsets
    eval_loaders = {}
    for name, path in subset_paths.items():
        csvs = list_csvs(path)
        if len(csvs) == 0:
            continue
        ds = RoADWindows(csvs, args.window, args.hop, scaler, assume_normal_if_no_label = False)
        if len(ds) == 0:
            continue
        eval_loaders[name] = DataLoader(ds, batch_size = args.batch_size, shuffle = False, drop_last = False)

    if len(eval_loaders) == 0:
        print("No anomaly subsets found for evaluation.")
        return

    results = evaluate(model, eval_loaders, device)
    print("\n=== Evaluation on anomaly subsets (window-level) ===")
    for name, res in results.items():
        if res.get("samples", 0) == 0:
            print(f"- {name}: no samples")
            continue
        roc = res.get("ROC-AUC")
        pr = res.get("PR-AUC")
        best_f1 = res.get("BestF1")
        best_th = res.get("BestThresh")
        f1_05 = res.get("F1@0.5")

    results = evaluate(model, eval_loaders, device)
    print("\n=== Evaluation on anomaly subsets (window-level) ===")
    for name, res in results.items():
        if res.get("samples", 0) == 0:
            print(f"- {name}: no samples")
            continue
        roc = res.get("ROC-AUC")
        pr = res.get("PR-AUC")
        best_f1 = res.get("BestF1")
        best_th = res.get("BestThresh")
        f1_05 = res.get("F1@0.5")
        note = res.get("note", "")
        print(
            f"- {name:9s} | N={res['samples']:7d} | "
            f"ROC-AUC={None if roc is None else f'{roc:.4f}'} | "
            f"PR-AUC={None if pr is None else f'{pr:.4f}'} | "
            f"BestF1={None if best_f1 is None else f'{best_f1:.4f}'} "
            f"@th={None if best_th is None else f'{best_th:.3f}'} | "
            f"F1@0.5={None if f1_05 is None else f'{f1_05:.4f}'} "
            f"{'| ' + note if note else ''}"
        )


if __name__ == "__main__":
    main()
