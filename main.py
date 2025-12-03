import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from models.cnn_1d import build_model as build_cnn
from models.mlp import WindowedMLP

# CONFIGURATION
RUNS_DIR = "runs_pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20  
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WINDOW_SIZE = 256
TRAIN_STRIDE = 128
TEST_STRIDE = 256
NUM_FEATURES = 86

# UTILS FOR DATA PROCESSING

def compute_minmax(train_records, num_features=86):
    """Computes global min and max from training records for normalization."""
    arrays = [np.asarray(rec)[:, :num_features] for rec in train_records]
    big = np.concatenate(arrays, axis=0)
    feat_min = big.min(axis=0)
    feat_max = big.max(axis=0)
    return feat_min.astype(np.float32), feat_max.astype(np.float32)

def minmax_scale(arr, min_vals, max_vals, eps=1e-8):
    """Scales data to [-1, 1]."""
    arr = arr.astype(np.float32)
    num = arr - min_vals
    den = (max_vals - min_vals) + eps
    return 2.0 * (num / den) - 1.0

def normalize_records(records, feat_min, feat_max, num_features=86):
    """Applies min-max scaling to a list of records."""
    norm_recs = []
    for rec in records:
        arr = np.asarray(rec)
        feats = arr[:, :num_features]
        feats_scaled = minmax_scale(feats, feat_min, feat_max)
        
        # Preserve label if present
        if arr.shape[1] > num_features:
            label = arr[:, num_features : num_features + 1]
            arr_scaled = np.concatenate([feats_scaled, label], axis=1)
        else:
            arr_scaled = feats_scaled
        norm_recs.append(arr_scaled)
    return norm_recs

class TimeSeriesWindowDataset(Dataset):
    """
    PyTorch Dataset that creates windows from continuous records.
    Unlike the AutoGluon script which computed stats (mean/std), 
    this returns raw tensors (C, T) for Deep Learning models.
    """
    def __init__(self, records, window_size, stride, num_features=86, max_windows=None):
        self.samples = []
        self.labels = []
        
        total_windows = 0
        
        for arr in records:
            arr = np.asarray(arr)
            N = arr.shape[0]
            if N < window_size:
                continue
                
            feats_all = arr[:, :num_features]
            # Check if label exists in the last column
            if arr.shape[1] > num_features:
                labels_all = arr[:, num_features]
            else:
                labels_all = np.zeros(N, dtype=np.float32)
                
            starts = list(range(0, N - window_size + 1, stride))
            
            for start in starts:
                end = start + window_size
                window_feats = feats_all[start:end, :]
                window_labels = labels_all[start:end]
                
                # Label logic: 1 if any anomaly in window, else 0
                label = 1.0 if window_labels.max() > 0 else 0.0
                
                # Transpose for PyTorch (T, C) -> (C, T)
                window_feats_t = window_feats.transpose(1, 0).astype(np.float32)
                
                self.samples.append(window_feats_t)
                self.labels.append(label)
                
                total_windows += 1
                if max_windows and total_windows >= max_windows:
                    break
            
            if max_windows and total_windows >= max_windows:
                break
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

# METRICS CALCULATION 

def compute_metrics(y_true, scores, thresholds=None):
    """
    Computes AUC, PR-AUC, Best F1, and optionally returns a learning curve DataFrame.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    
    # BCE Loss estimation for metrics
    eps = 1e-7
    p = np.clip(scores, eps, 1 - eps)
    bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    loss_overall = float(bce.mean())
    
    auc_roc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    
    # Find Best F1
    unique_scores = np.unique(scores)
    # Downsample thresholds if too many for speed
    curr_thrs = unique_scores if len(unique_scores) < 1000 else np.quantile(unique_scores, np.linspace(0,1,1000))
    
    best_f1 = -1.0
    best_thr = 0.5
    
    for thr in curr_thrs:
        y_pred = (scores >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            
    # Metrics at 0.5
    y_pred_05 = (scores >= 0.5).astype(int)
    _, _, f1_05, _ = precision_recall_fscore_support(y_true, y_pred_05, average="binary", zero_division=0)
    acc_05 = float((y_pred_05 == y_true).mean())
    
    metrics = {
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "bce_loss_overall": loss_overall,
        "best_f1": float(best_f1),
        "best_threshold": float(best_thr),
        "f1_at_0.5": float(f1_05),
        "accuracy_at_0.5": acc_05
    }
    
    learning_curve = None
    if thresholds is not None:
        acc_list, f1_list, acc0_list, acc1_list = [], [], [], []
        total_pos = (y_true == 1).sum()
        total_neg = (y_true == 0).sum()
        
        for thr in thresholds:
            y_pred_thr = (scores >= thr).astype(int)
            acc = (y_pred_thr == y_true).mean()
            _, _, f1_sc, _ = precision_recall_fscore_support(y_true, y_pred_thr, average="binary", zero_division=0)
            
            tp = ((y_true == 1) & (y_pred_thr == 1)).sum()
            tn = ((y_true == 0) & (y_pred_thr == 0)).sum()
            
            acc0 = tn / total_neg if total_neg > 0 else 0.0
            acc1 = tp / total_pos if total_pos > 0 else 0.0
            
            acc_list.append(acc)
            f1_list.append(f1_sc)
            acc0_list.append(acc0)
            acc1_list.append(acc1)
            
        learning_curve = pd.DataFrame({
            "threshold": thresholds,
            "accuracy_overall": acc_list,
            "f1_score": f1_list,
            "accuracy_class0": acc0_list,
            "accuracy_class1": acc1_list,
        })
        
    return metrics, learning_curve

# MODEL WRAPPERS 

class ClassificationMLP(nn.Module):
    """
    Wraps the provided WindowedMLP (Autoencoder) to act as a Classifier.
    It uses the 'forward_features' method (Encoder) and adds a classification head.
    """
    def __init__(self, in_channels, window_size, hidden_dim=128, bottleneck_dim=32, dropout=0.2):
        super().__init__()
        # Initialize the base autoencoder
        self.backbone = WindowedMLP(in_channels, window_size, hidden_dim, bottleneck_dim, dropout)
        # Classification head (from latent space z to 1 logit)
        self.head = nn.Linear(bottleneck_dim, 1)
        
    def forward(self, x):
        # Extract features (z) using the encoder part of the MLP
        z = self.backbone.forward_features(x) 
        # Project to logit
        return self.head(z)

class ClassificationCNN(nn.Module):
    """
    Wraps the RoADCNN1D. The original model returns (seq_logit, frame_logits).
    We only need seq_logit for this binary classification task.
    """
    def __init__(self, in_channels=86, base_channels=64, dropout=0.1):
        super().__init__()
        self.model = build_cnn(in_channels=in_channels, base_channels=base_channels, dropout=dropout)
        
    def forward(self, x):
        seq_logit, _ = self.model(x)
        return seq_logit

# TRAINING ENGINE

def train_model(model, train_loader, test_loader, model_name, output_dir):
    print(f"\n--- Training {model_name} ---")
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    
    # Directories
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f}")
        
    # Final Evaluation on Test Set
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_scores.extend(probs)
            all_labels.extend(y_batch.numpy())
            
    # Calculate Metrics
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics, lc_df = compute_metrics(all_labels, all_scores, thresholds)
    
    print(f"  -> Test ROC AUC: {metrics['auc_roc']:.4f}")
    print(f"  -> Best F1:      {metrics['best_f1']:.4f}")
    
    # Save outputs
    # 1. Metrics JSON
    with open(os.path.join(model_dir, "metrics_train_test.json"), "w") as f:
        # Note: We simulate the 'train' key to match plotting script structure, 
        # though we only calculated detailed metrics for test here.
        json.dump({"test": metrics}, f, indent=2)
        
    # 2. Learning Curve CSV
    if lc_df is not None:
        lc_df.to_csv(os.path.join(model_dir, "learning_curve_test.csv"), index=False)
        
    # 3. Save Model
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    
    return metrics

# MAIN

def main():
    # 1. Setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"run_{timestamp}")
    models_dir = os.path.join(run_dir, "Models") 
    os.makedirs(models_dir, exist_ok=True)
    print(f"Output Directory: {run_dir}")
    
    # 2. Data Loading (Using RoADDataset from your environment)
    from RoADDataset import Dataset as RoADDataset
    ds = RoADDataset(normalize=False)
    
    # Split Records
    base_train = list(ds.sets["training"])
    control_recs = list(ds.sets.get("control", []))
    collision_recs = list(ds.sets.get("collision", []))
    weight_recs = list(ds.sets.get("weight", []))
    velocity_recs = list(ds.sets.get("velocity", []))
    
    train_group = base_train + control_recs
    
    # 3. Normalization (Fit on Train, Transform All)
    print("Computing Normalization Stats...")
    feat_min, feat_max = compute_minmax(train_group, NUM_FEATURES)
    
    norm_train = normalize_records(train_group, feat_min, feat_max, NUM_FEATURES)
    norm_collision = normalize_records(collision_recs, feat_min, feat_max, NUM_FEATURES)
    norm_weight = normalize_records(weight_recs, feat_min, feat_max, NUM_FEATURES)
    norm_velocity = normalize_records(velocity_recs, feat_min, feat_max, NUM_FEATURES)
    
    scenario_data = {
        "normal": norm_train,
        "collision": norm_collision,
        "weight": norm_weight,
        "velocity": norm_velocity
    }
    
    # 4. Train/Test Split & Windowing
    train_windows = []
    test_windows = []
    
    print("Creating Windows...")
    for name, records in scenario_data.items():
        if not records: continue
        
        # Split records 70/30
        if len(records) > 1:
            recs_tr, recs_te = train_test_split(records, test_size=0.3, random_state=42)
        else:
            recs_tr, recs_te = records, []
            
        # Build Datasets
        # Note: We merge lists here, will create TensorDataset later
        ds_tr = TimeSeriesWindowDataset(recs_tr, WINDOW_SIZE, TRAIN_STRIDE, NUM_FEATURES, max_windows=4000)
        ds_te = TimeSeriesWindowDataset(recs_te, WINDOW_SIZE, TEST_STRIDE, NUM_FEATURES, max_windows=None)
        
        train_windows.extend(ds_tr)
        test_windows.extend(ds_te)
        
    print(f"Train Windows: {len(train_windows)}")
    print(f"Test Windows:  {len(test_windows)}")
    
    # Create DataLoaders
    # We need to manually collate the list of (tensor, label) tuples
    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y
        
    train_loader = DataLoader(train_windows, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_windows, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 5. Define Models
    models_to_train = [
        ("MLP_Classifier", ClassificationMLP(in_channels=NUM_FEATURES, window_size=WINDOW_SIZE)),
        ("CNN_1D", ClassificationCNN(in_channels=NUM_FEATURES))
    ]
    
    comparison_rows = []
    
    # 6. Training Loop
    for model_name, model in models_to_train:
        metrics = train_model(model, train_loader, test_loader, model_name, models_dir)
        
        comparison_rows.append({
            "model": model_name,
            "bce_loss_overall": metrics["bce_loss_overall"],
            "accuracy_at_0.5": metrics["accuracy_at_0.5"],
            "f1_at_0.5": metrics["f1_at_0.5"],
            "best_f1": metrics["best_f1"],
            "roc_auc": metrics["auc_roc"]
        })
        
    # 7. Save Comparative Results
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(run_dir, "models_comparison_test.csv"), index = False)
    print("Done.")

if __name__ == "__main__":
    main()