import os
import time
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Custom imports
from RoADDataset import Dataset as RoADDataset
from models.cnn_1d import build_model

# ---------------------------------------------------------
# 1. DATASET WINDOWING
# ---------------------------------------------------------
class RoADWindowedDataset(Dataset):
    def __init__(self, records, window_size, stride, num_features=86, include_label=False):
        """
        Creates sliding windows from the list of records.
        """
        self.records = records
        self.window_size = window_size
        self.stride = stride
        self.num_features = num_features
        self.include_label = include_label
        self.index_map = [] 

        # Build an index map to handle multiple recordings of different lengths
        for rec_idx, arr in enumerate(records):
            N = arr.shape[0]
            if N < window_size: continue
            for start in range(0, N - window_size + 1, stride):
                self.index_map.append((rec_idx, start))

    def __len__(self): 
        return len(self.index_map)
    
    def __getitem__(self, idx): 
        rec_idx, start = self.index_map[idx]
        arr = self.records[rec_idx]
        
        # Extract window and transpose to (Channels, Time) for PyTorch Conv1d
        window_data = arr[start : start + self.window_size, :self.num_features]
        x_tensor = torch.from_numpy(window_data.T.astype(np.float32)) 

        if not self.include_label:
            return x_tensor

        # If labels are present (usually the last column), check if any anomaly exists in the window
        if arr.shape[1] > self.num_features:
            window_labels = arr[start : start + self.window_size, self.num_features]
            label_value = 1.0 if window_labels.max() > 0 else 0.0
        else:
            label_value = 0.0
            
        return x_tensor, torch.tensor(label_value, dtype=torch.float32)

# ---------------------------------------------------------
# 2. UTILS
# ---------------------------------------------------------
def compute_minmax_incremental(train_records, num_features=86):
    """
    Computes global min and max across all training records incrementally
    to avoid loading everything into memory at once.
    """
    print("Computing Min/Max incrementally...")
    global_min = np.full(num_features, np.inf, dtype=np.float32)
    global_max = np.full(num_features, -np.inf, dtype=np.float32)
    
    for rec in train_records:
        arr = np.asarray(rec)[:, :num_features]
        if arr.shape[0] == 0: continue
        global_min = np.minimum(global_min, arr.min(axis=0))
        global_max = np.maximum(global_max, arr.max(axis=0))
        
    return global_min, global_max

def normalize_records(records, feat_min, feat_max, num_features=86):
    """
    Normalizes features to the range [-1, 1].
    """
    norm_recs = []
    eps = 1e-8
    den = (feat_max - feat_min) + eps
    
    for rec in records:
        arr = np.asarray(rec)
        feats = arr[:, :num_features]
        # Scale to [-1, 1]
        feats_scaled = 2.0 * ((feats - feat_min) / den) - 1.0
        
        # Keep labels untouched if they exist
        if arr.shape[1] > num_features:
            arr_scaled = np.concatenate([feats_scaled, arr[:, num_features:]], axis=1)
        else:
            arr_scaled = feats_scaled
        norm_recs.append(arr_scaled)
        
    return norm_recs

def compute_center(train_loader, model, device):
    """
    Initializes the hypersphere center 'c' for Deep SVDD 
    by taking the mean of the embeddings from the initial forward pass.
    """
    print("Computing Deep SVDD center c...")
    model.eval()
    
    # Dummy pass to get output shape
    dummy = torch.zeros(1, 86, 256).to(device)
    c = torch.zeros(model.forward_features(dummy).shape[1]).to(device)
    n_samples = 0
    
    with torch.no_grad():
        for x in train_loader:
            x = x.to(device)
            z = model.forward_features(x)
            c += z.sum(dim=0)
            n_samples += z.shape[0]
            
    c /= n_samples
    
    # Avoid zero values to prevent numerical instability
    c[(abs(c) < 0.1) & (c < 0)] = -0.1
    c[(abs(c) < 0.1) & (c > 0)] = 0.1
    
    return c.detach()

# ---------------------------------------------------------
# 3. FEATURE IMPORTANCE (PERMUTATION)
# ---------------------------------------------------------
def compute_permutation_importance(model, loader, c, device, feature_names=None):
    """
    Calculates feature importance by shuffling one channel at a time
    and measuring the increase in Loss (distance from center).
    Higher loss increase = more important feature.
    """
    print("\nComputing Permutation Feature Importance...")
    model.eval()
    
    # 1. Compute Base Loss (using a subset for speed)
    original_losses = []
    inputs = []
    
    max_batches = 10 
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            if isinstance(batch, list): x = batch[0]
            else: x = batch
            x = x.to(device)
            inputs.append(x)
            
            z = model.forward_features(x)
            loss = ((z - c) ** 2).sum(dim=1)
            original_losses.append(loss.cpu().numpy())
            
    inputs = torch.cat(inputs, dim=0) # (N, C, T)
    base_loss = np.concatenate(original_losses).mean()
    
    print(f"Base Loss: {base_loss:.6f}")
    
    importances = []
    num_features = inputs.shape[1] # C
    
    for f_idx in range(num_features):
        # Permutation: shuffle values of this channel across the batch
        # This breaks the temporal and statistical structure of THIS specific feature
        x_perm = inputs.clone()
        idx = torch.randperm(x_perm.shape[0])
        x_perm[:, f_idx, :] = x_perm[idx, f_idx, :] 
        
        with torch.no_grad():
            z = model.forward_features(x_perm)
            loss = ((z - c) ** 2).sum(dim=1).mean().item()
        
        # Importance = how much the loss increased
        imp = loss - base_loss
        importances.append(imp)
        
        if (f_idx+1) % 10 == 0:
            print(f"Processed feature {f_idx+1}/{num_features}...")

    # Save results to dataframe
    if feature_names is None or len(feature_names) != num_features:
        feature_names = [f"feat_{i}" for i in range(num_features)]
        
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    df_imp = df_imp.sort_values(by="importance", ascending=False)
    return df_imp

# ---------------------------------------------------------
# 4. TRAINING & SAVE LOGIC
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, c, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x in loader:
        x = x.to(device)
        optimizer.zero_grad()
        
        # Deep SVDD Loss: Minimize distance to center c
        z = model.forward_features(x)
        loss = ((z - c)**2).sum(dim=1).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)

def collect_inference_results(model, loader, c, device):
    """
    Runs the model on the entire dataset and returns scores and labels.
    """
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.forward_features(x)
            # Anomaly score is the squared distance from center
            dist = ((z - c)**2).sum(dim=1)
            scores.append(dist.cpu().numpy())
            labels.append(y.numpy())
            
    if len(scores) == 0: return np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(labels)

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs_cnn", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nSaving outputs to: {run_dir}")

    # Configuration
    WINDOW_SIZE = 256
    STRIDE = 128
    NUM_FEAT = 86
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Dataset
    ds = RoADDataset(normalize=False)
    
    # Get feature names if available in the dataset object
    feat_names = ds.feature_names if hasattr(ds, 'feature_names') else None

    # Load & Normalize Data
    train_recs = list(ds.sets["training"]) + list(ds.sets.get("control", []))
    coll_recs = list(ds.sets.get("collision", []))
    weight_recs = list(ds.sets.get("weight", []))
    vel_recs = list(ds.sets.get("velocity", []))
    
    # Compute scaling parameters on training data only
    fmin, fmax = compute_minmax_incremental(train_recs, NUM_FEAT)
    
    # Apply normalization
    train_norm = normalize_records(train_recs, fmin, fmax)
    coll_norm = normalize_records(coll_recs, fmin, fmax)
    weight_norm = normalize_records(weight_recs, fmin, fmax)
    vel_norm = normalize_records(vel_recs, fmin, fmax)
    ctrl_norm = normalize_records(ds.sets.get("control", []), fmin, fmax)

    # Data Loaders
    train_loader = DataLoader(RoADWindowedDataset(train_norm, WINDOW_SIZE, STRIDE), batch_size=128, shuffle=True)
    
    # Build Model
    model = build_model(in_channels=NUM_FEAT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize Center
    c = compute_center(train_loader, model, DEVICE)
    
    # Training Loop
    best_loss = np.inf
    for ep in range(1, 21): # Train for 20 epochs
        loss = train_one_epoch(model, train_loader, optimizer, c, DEVICE)
        print(f"Epoch {ep:02d} | Loss: {loss:.6f}")
        
        # Save best model checkpoint
        if loss < best_loss:
            best_loss = loss
            ckpt = {"model": model.state_dict(), "c": c, "min": fmin, "max": fmax}
            torch.save(ckpt, os.path.join(run_dir, "model_best.pt"))

    # --- POST TRAINING EVALUATION ---
    print("\nLoading best model for evaluation...")
    ckpt = torch.load(os.path.join(run_dir, "model_best.pt"), weights_only = False) 
    model.load_state_dict(ckpt["model"])
    c = ckpt["c"]

    # 1. Feature Importance
    # We use the training set to understand what the model considers "normal" features
    df_imp = compute_permutation_importance(model, train_loader, c, DEVICE, feature_names=feat_names)
    df_imp.to_csv(os.path.join(run_dir, "feature_importance.csv"), index=False)
    print(f"Feature importance saved.")

    # 2. Inference on Test Sets
    # Define scenarios: (Anomaly Records, Normal Records)
    test_scenarios = {
        "collision": (coll_norm, ctrl_norm), 
        "weight": (weight_norm, ctrl_norm),
        "velocity": (vel_norm, ctrl_norm)
    }

    for name, (anom_recs, norm_recs) in test_scenarios.items():
        if len(anom_recs) == 0: continue
        
        # Merge anomalies and normal controls for testing
        full_recs = anom_recs + norm_recs
        
        # Create dataset and loader (using include_label=True for evaluation)
        ds_test = RoADWindowedDataset(full_recs, WINDOW_SIZE, 256, include_label=True)
        dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)
        
        s, y = collect_inference_results(model, dl_test, c, DEVICE)
        
        # Save raw scores and labels for plotting later
        res_path = os.path.join(run_dir, f"results_{name}.npz")
        np.savez(res_path, scores=s, labels=y)
        print(f"Saved results for {name} to {res_path}")

    print("\nDone. Run plots.py to visualize curves.")

if __name__ == "__main__":
    main()