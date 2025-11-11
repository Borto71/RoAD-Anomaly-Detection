import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.cnn_1d import build_model



# Basic, reproducible defaults

SEED = 42
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
DATASET_ROOT = "./data/dataset"  
TRAIN_DIR = "training"
VAL_DIR = "control"


def set_seed(seed: int = 42):
    """
    Set all relevant RNG seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(split_dir: str):
    """
    Load one split (e.g., training or control) from a directory containing .npy files.

    Each .npy is expected to be a 2D array with shape (T, F+1):
      - columns 0..F-1: features
      - column F: label per row (time step)

    We convert each file into ONE sample:
      - X_i: all rows, all feature columns -> shape (T, F)
      - y_i: a single class integer (rounded median of last column)

    Then we pad sequences in this split to the split's max length (T_max),
    returning:
      X: (N, T_max, F), float32
      y: (N,), int64
    """
    X_list, y_list = [], []
    max_T = 0
    F_ref = None

    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith(".npy"):
            continue

        path = os.path.join(split_dir, fname)
        arr = np.load(path)

        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"{fname}: expected 2D array with at least 2 columns (features + label), got {arr.shape}"
            )

        # Split features/labels
        Xi = arr[:, :-1]  # (T, F)
        # Single label per file (rounded median of the per-row labels)
        yi = int(np.round(np.median(arr[:, -1])))

        # Feature-count consistency within the split
        if F_ref is None:
            F_ref = Xi.shape[1]
        elif Xi.shape[1] != F_ref:
            raise ValueError(
                f"{fname}: inconsistent feature count in split: {Xi.shape[1]} vs expected {F_ref}"
            )

        X_list.append(Xi.astype(np.float32))
        y_list.append(yi)
        max_T = max(max_T, Xi.shape[0])

    if len(X_list) == 0:
        raise RuntimeError(f"No .npy files found in: {split_dir}")

    # Pad to T_max within this split
    N = len(X_list)
    F = F_ref
    X = np.zeros((N, max_T, F), dtype=np.float32)
    for i, Xi in enumerate(X_list):
        T = Xi.shape[0]
        X[i, :T, :] = Xi

    y = np.array(y_list, dtype=np.int64)
    return X, y


def harmonize_features(Xa: np.ndarray, Xb: np.ndarray):
    """
    Ensure Xa and Xb (both shaped (N, T, F)) have the same number of features (F).
    If different, we either truncate extra columns or pad missing columns with zeros.

    Returns (Xa_h, Xb_h) with matching F.
    """
    Fa = Xa.shape[2]
    Fb = Xb.shape[2]
    if Fa == Fb:
        return Xa, Xb

    print(f"[WARN] Feature mismatch: splitA={Fa}, splitB={Fb}. Harmonizing...")

    if Fa > Fb:
        # Pad B to Fa
        pad = np.zeros((Xb.shape[0], Xb.shape[1], Fa - Fb), dtype=Xb.dtype)
        Xb = np.concatenate([Xb, pad], axis=2)
    else:
        # Pad A to Fb
        pad = np.zeros((Xa.shape[0], Xa.shape[1], Fb - Fa), dtype=Xa.dtype)
        Xa = np.concatenate([Xa, pad], axis=2)

    return Xa, Xb


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    """
    Convert (N, T, F) numpy arrays into a PyTorch DataLoader with shape (B, C, T),
    where C=F for Conv1d input.
    """
    # (N, T, F) -> (N, F, T)
    X = torch.from_numpy(X).permute(0, 2, 1).contiguous()
    y = torch.from_numpy(y)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Basic training loop for one epoch: forward, loss, backward, step.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)             # shape (B, num_classes)
        loss = criterion(logits, yb)   # CrossEntropyLoss expects (B, C) and targets (B,)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


def evaluate(model, loader, device):
    """
    Simple evaluation: compute accuracy over the whole loader.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()

    accuracy = correct / max(1, total)
    return accuracy


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------
    # 1) Load data
    # -----------------------------
    print("Loading data...")
    train_dir = os.path.join(DATASET_ROOT, TRAIN_DIR)
    val_dir = os.path.join(DATASET_ROOT, VAL_DIR)

    X_train, y_train = load_split(train_dir)
    X_val, y_val = load_split(val_dir)

    # Ensure class IDs are consecutive ints starting from 0
    # (maps any arbitrary labels to [0..K-1] consistently across splits)
    print("Normalizing class IDs...")
    classes = np.unique(np.concatenate([y_train, y_val], axis=0))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([class_to_idx[c] for c in y_train], dtype=np.int64)
    y_val = np.array([class_to_idx[c] for c in y_val], dtype=np.int64)

    # Harmonize feature count (F) across splits if needed
    X_train, X_val = harmonize_features(X_train, X_val)

    # -----------------------------
    # 2) Build loaders
    # -----------------------------
    train_loader = make_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    in_channels = X_train.shape[2]         # F
    num_classes = int(len(np.unique(y_train)))
    print(f"in_channels={in_channels} | num_classes={num_classes}")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # -----------------------------
    # 3) Initialize model / loss / optimizer
    # -----------------------------
    print("Initializing model...")
    model = build_model(in_channels=in_channels, num_classes=num_classes).to(device)

    # Sanity check: the model must output (B, num_classes)
    xb_sanity, yb_sanity = next(iter(train_loader))
    with torch.no_grad():
        out = model(xb_sanity.to(device))
    assert out.ndim == 2 and out.shape[1] == num_classes, \
        f"Model output must be (B, {num_classes}), got {tuple(out.shape)}"

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------------
    # 4) Train
    # -----------------------------
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: loss = {loss:.4f}")

    # -----------------------------
    # 5) Evaluate
    # -----------------------------
    print("Evaluating...")
    val_acc = evaluate(model, val_loader, device)
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
