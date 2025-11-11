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
from torch.utils.data import Dataset, DataLoader

# Import your model factory; keep path consistent with your repo layout.
from models.cnn_1d import build_model


# ------------------------------- Utils ---------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set all relevant seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_csvs(folder: str) -> List[str]:
    """Return a sorted list of CSV paths inside a folder."""
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def load_csv(path: str) -> np.ndarray:
    """Load a CSV with no header into a float32 numpy array."""
    df = pd.read_csv(path, header=None)
    return df.to_numpy(dtype=np.float32)


def windowed_views(
    X: np.ndarray, y: Optional[np.ndarray], win: int, hop: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Cut a (T, C) array into overlapping windows of length `win` with step `hop`.
    If y is provided, window label is max in the window (weak supervision).
    Returns:
      Xw: (W, win, C)
      yw: (W,) or None
    """
    n = X.shape[0]
    if n < win:
        empty_X = np.empty((0, win, X.shape[1]), dtype=np.float32)
        empty_y = None if y is None else np.empty((0,), dtype=np.float32)
        return empty_X, empty_y

    starts = np.arange(0, n - win + 1, hop, dtype=np.int32)
    Xw = np.stack([X[s : s + win] for s in starts], axis=0)

    if y is None:
        yw = None
    else:
        yw = np.stack([y[s : s + win].max() for s in starts], axis=0).astype(np.float32)

    return Xw, yw


def fit_minmax_on_training(training_csvs: List[str]) -> MinMaxScaler:
    """
    Fit a MinMax scaler on all training files.
    We use partial_fit to avoid holding everything in memory at once.
    """
    scaler = MinMaxScaler()
    for p in training_csvs:
        arr = load_csv(p)
        feats = arr[:, :86] if arr.shape[1] >= 86 else arr
        scaler.partial_fit(feats)
    return scaler


# ------------------------------ Dataset --------------------------------------


class RoADWindows(Dataset):
    """
    A dataset that loads CSVs, scales features, and returns sliding windows.

    - Features: first 86 columns.
    - Label: column 87 if present; will be binarized to {0,1} via (label > 0).
    - If no label column exists and assume_normal_if_no_label=True, windows are 0.
    - Returns (C, T) tensors for 1D CNN input.
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
                # Skip malformed files
                continue

            feats = arr[:, :86].astype(np.float32)

            # If there is a label column, binarize it to {0,1}
            lbls = arr[:, 86].astype(np.float32) if arr.shape[1] >= 87 else None
            if lbls is not None:
                lbls = (lbls > 0).astype(np.float32)

            # Apply scaling if provided
            if self.scaler is not None:
                feats = self.scaler.transform(feats)

            Xw, yw = windowed_views(feats, lbls, win=self.win, hop=self.hop)
            if Xw.shape[0] == 0:
                continue

            if yw is None:
                if not self.assume_normal_if_no_label:
                    # Skip files with no labels if we require labels
                    continue
                yw = np.zeros((Xw.shape[0],), dtype=np.float32)

            Xw_list.append(Xw)
            yw_list.append(yw)

        if len(Xw_list) == 0:
            self.X = np.empty((0, 86, self.win), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.float32)
        else:
            X = np.concatenate(Xw_list, axis=0)  # (N, win, C)
            y = np.concatenate(yw_list, axis=0)  # (N,)
            # Convert to channels-first (N, C, T)
            self.X = np.transpose(X, (0, 2, 1)).astype(np.float32)
            self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (C, T)
        y = self.y[idx]  # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# --------------------------- Train / Evaluate --------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler_amp: Optional[torch.amp.GradScaler] = None,
) -> float:
    """One training epoch with optional AMP."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0

    for x, y in loader:
        x = x.to(device)  # (B, C, T)
        y = y.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad(set_to_none=True)

        if scaler_amp is not None and device.type == "cuda":
            # Use new torch.amp autocast to avoid deprecation warnings
            with torch.amp.autocast("cuda", enabled=True):
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
    """
    Evaluate on anomaly subsets. Uses logits directly as scores to avoid
    numerical issues with exp/sigmoid. Handles single-class cases gracefully.
    """
    model.eval()
    results: Dict[str, dict] = {}

    for name, loader in subset_loaders.items():
        all_logits, all_labels = [], []

        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)  # (B, 1)
            all_logits.append(logits.squeeze(1).cpu().numpy())
            all_labels.append(y.numpy())

        if len(all_logits) == 0:
            results[name] = {"samples": 0}
            continue

        scores = np.concatenate(all_logits, axis=0)  # logits as scores
        labels = np.concatenate(all_labels, axis=0).astype(np.float32)

        # If only one class is present, AUC/PR are undefined. Report None.
        unique = np.unique(labels)
        if unique.size < 2:
            results[name] = {
                "samples": int(labels.shape[0]),
                "ROC-AUC": None,
                "PR-AUC": None,
                "BestF1": None,
                "BestThresh": None,
                "F1@0.5": None,
                "note": f"single-class labels in {name} (labels={unique.tolist()})",
            }
            continue

        # ROC/PR AUC can consume raw scores (no need for sigmoid).
        roc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))

        # Choose a threshold by maximizing F1 on the precision-recall curve.
        ps, rs, ths = precision_recall_curve(labels, scores)
        f1s = 2 * ps * rs / (ps + rs + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        best_f1 = float(f1s[best_idx])
        # precision_recall_curve returns thresholds of shape (len(ps)-1)
        best_th = float(ths[best_idx - 1]) if 0 < best_idx < len(ths) else 0.0

        # F1@0.5 using a numerically safe sigmoid (clipped logits)
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


# --------------------------------- Main --------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dataset", help="Base folder with subset subdirectories.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_path", type = str, default = "checkpoint/road_cnn.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU.")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--no_fit_scaler", action="store_true", help="Skip MinMax fit on training subset.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training windows for validation.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    # Collect CSVs
    train_dir = os.path.join(args.data_dir, "training")
    anomaly_subsets = ["collision", "control", "weight", "velocity"]
    subset_paths = {s: os.path.join(args.data_dir, s) for s in anomaly_subsets}

    train_csvs = list_csvs(train_dir)
    if len(train_csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in: {train_dir}")

    # Fit scaler on training if requested
    scaler = None
    if not args.no_fit_scaler:
        scaler = fit_minmax_on_training(train_csvs)

    # Build training dataset from training subset only (all windows are normal)
    full_train_ds = RoADWindows(
        csv_paths=train_csvs,
        win=args.window,
        hop=args.hop,
        scaler=scaler,
        assume_normal_if_no_label=True,
    )
    if len(full_train_ds) == 0:
        raise RuntimeError("No training windows created. Check window/hop sizes or data integrity.")

    # Simple hold-out split for validation
    n = len(full_train_ds)
    perm = np.random.permutation(n)
    val_n = max(1, int(args.val_split * n))
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]

    train_subset = torch.utils.data.Subset(full_train_ds, tr_idx.tolist())
    val_subset = torch.utils.data.Subset(full_train_ds, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Build evaluation loaders (these have labels)
    eval_loaders: Dict[str, DataLoader] = {}
    for name, path in subset_paths.items():
        csvs = list_csvs(path)
        if len(csvs) == 0:
            continue
        ds = RoADWindows(
            csv_paths=csvs,
            win=args.window,
            hop=args.hop,
            scaler=scaler,
            assume_normal_if_no_label=False,  # require labels for evaluation
        )
        if len(ds) == 0:
            continue
        eval_loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Model and optimizer
    model = build_model(in_channels=86, base_channels=64, dropout=0.1, return_frame_logits=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

   
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler_amp=scaler_amp)

        # Validation on normal windows (lower BCE is better here)
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
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

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "scaler": scaler,  # keep the fitted scaler for downstream use
                "config": vars(args),
                "epoch": epoch,
            }

    # Save best state (if any)
    if best_state is not None:
        torch.save(best_state, args.save_path)
        print(f"Saved best model to: {args.save_path}")

    # Final evaluation on anomaly subsets (if present)
    if len(eval_loaders) == 0:
        print("No anomaly subset CSVs found for evaluation.")
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
