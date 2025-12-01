import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# Import your dataset
from RoADDataset import Dataset as RoADDataset

# --- Configuration ---
WINDOW_SIZE = 256
TRAIN_STRIDE = 128
TEST_STRIDE = 256
NUM_FEATURES = 86
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions (Data Processing) ---

def compute_minmax(train_records, num_features=86):
    arrays = [np.asarray(rec)[:, :num_features] for rec in train_records]
    big = np.concatenate(arrays, axis=0)
    feat_min = big.min(axis=0).astype(np.float32)
    feat_max = big.max(axis=0).astype(np.float32)
    return feat_min, feat_max

def normalize_records(records, feat_min, feat_max, num_features=86):
    norm_recs = []
    eps = 1e-8
    den = (feat_max - feat_min) + eps
    for rec in records:
        arr = np.asarray(rec)
        feats = arr[:, :num_features].astype(np.float32)
        feats_scaled = 2.0 * ((feats - feat_min) / den) - 1.0
        
        if arr.shape[1] > num_features:
            label = arr[:, num_features : num_features + 1]
            arr_scaled = np.concatenate([feats_scaled, label], axis=1)
        else:
            arr_scaled = feats_scaled
        norm_recs.append(arr_scaled)
    return norm_recs

def create_sequences(records, window_size, stride, num_features, label_col_idx=86, max_windows=None):
    X_list = []
    y_list = []
    total_windows = 0
    for arr in records:
        N = arr.shape[0]
        if N < window_size: continue
        
        feats = arr[:, :num_features]
        labels = arr[:, label_col_idx] if arr.shape[1] > num_features else np.zeros(N)
        
        starts = range(0, N - window_size + 1, stride)
        for start in starts:
            end = start + window_size
            seq = feats[start:end, :]
            label = 1 if labels[start:end].max() > 0 else 0
            
            X_list.append(seq)
            y_list.append(label)
            total_windows += 1
            if max_windows and total_windows >= max_windows: break
        if max_windows and total_windows >= max_windows: break
            
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)

def compute_metrics_and_curves(y_true, scores, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    
    eps = 1e-7
    p = np.clip(scores, eps, 1 - eps)
    bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    
    best_f1 = 0
    best_thr = 0.5
    search_thrs = np.linspace(0, 1, 100)
    for thr in search_thrs:
        pred = (scores >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            
    pred_05 = (scores >= 0.5).astype(int)
    _, _, f1_05, _ = precision_recall_fscore_support(y_true, pred_05, average='binary', zero_division=0)
    acc_05 = (pred_05 == y_true).mean()
    
    metrics = {
        "auc_roc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "bce_loss_overall": float(bce.mean()),
        "bce_loss_class0": float(bce[y_true == 0].mean()) if (y_true==0).any() else 0.0,
        "bce_loss_class1": float(bce[y_true == 1].mean()) if (y_true==1).any() else 0.0,
        "best_f1": float(best_f1),
        "test_threshold": float(best_thr),
        "f1_at_0.5": float(f1_05),
        "accuracy_at_0.5": float(acc_05)
    }
    
    lc_df = None
    if thresholds is not None:
        data = []
        total_neg = (y_true == 0).sum()
        total_pos = (y_true == 1).sum()
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            acc = (pred == y_true).mean()
            _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
            tn = ((y_true == 0) & (pred == 0)).sum()
            tp = ((y_true == 1) & (pred == 1)).sum()
            acc0 = tn / total_neg if total_neg > 0 else 0
            acc1 = tp / total_pos if total_pos > 0 else 0
            data.append({
                "threshold": thr,
                "accuracy_overall": acc,
                "f1_score": f1,
                "accuracy_class0": acc0,
                "accuracy_class1": acc1
            })
        lc_df = pd.DataFrame(data)
        
    return metrics, lc_df

# --- Permutation Importance Function ---

def compute_permutation_importance(model, dataset_class, X_test, y_test, baseline_score, feature_names, device, batch_size=64):
    print("\nComputing Permutation Importance (this may take a while)...")
    model.eval()
    importances = []
    
    n_features = X_test.shape[2] 
    X_test_np = np.array(X_test, copy=True)
    
    def get_score(X_in):
        # We must use the correct Dataset class (CNN vs MLP) to format input correctly
        ds = dataset_class(X_in, y_test) 
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for inputs, _ in dl:
                inputs = inputs.to(device)
                out = model(inputs)
                all_scores.extend(out.cpu().numpy())
        return roc_auc_score(y_test, all_scores)

    for i in range(n_features):
        original_col = X_test_np[:, :, i].copy()
        np.random.shuffle(X_test_np[:, :, i]) 
        
        perm_score = get_score(X_test_np)
        imp = baseline_score - perm_score 
        importances.append(imp)
        
        X_test_np[:, :, i] = original_col 
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{n_features} features...")

    if feature_names and len(feature_names) == n_features:
        names = feature_names
    else:
        names = [f"f{k}" for k in range(n_features)]
        
    fi_df = pd.DataFrame({
        "feature": names,
        "importance": importances
    })
    fi_df = fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return fi_df

# --- Datasets ---

class CNNDataset(Dataset):
    def __init__(self, X, y):
        # CNN expects (Batch, Features, Time) -> Permute
        self.X = torch.FloatTensor(X).permute(0, 2, 1) 
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLPDataset(Dataset):
    def __init__(self, X, y):
        # MLP expects (Batch, Window, Features) -> The model's Flatten layer will handle 2D->1D
        # We just pass it as FloatTensor
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- Models ---

class Simple1DCNN(nn.Module):
    def __init__(self, num_features, window_size):
        super(Simple1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze()

class SimpleMLP(nn.Module):
    def __init__(self, num_features, window_size):
        super(SimpleMLP, self).__init__()
        input_dim = num_features * window_size
        self.net = nn.Sequential(
            nn.Flatten(), # Flattens (Batch, Window, Features) -> (Batch, Window*Features)
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze()

# --- Generic Training Function ---

def train_and_evaluate_model(model_name, model, dataset_cls, X_train, y_train, X_test, y_test, run_dir, feature_names):
    """
    Generic function to train a model, evaluate it, and save all artifacts.
    """
    print(f"\n{'='*40}")
    print(f"Starting pipeline for: {model_name}")
    print(f"{'='*40}")

    model_save_dir = os.path.join(run_dir, "Models", model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # 1. Dataloaders
    train_ds = dataset_cls(X_train, y_train)
    test_ds = dataset_cls(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("Training...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Optional: Print every few epochs to reduce clutter
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # 4. Evaluation
    print("Evaluating...")
    model.eval()
    
    # Train Scores
    train_scores, train_true = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            out = model(inputs.to(DEVICE))
            train_scores.extend(out.cpu().numpy())
            train_true.extend(labels.numpy())
            
    # Test Scores
    test_scores, test_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            out = model(inputs.to(DEVICE))
            test_scores.extend(out.cpu().numpy())
            test_true.extend(labels.numpy())
            
    # Metrics
    metrics_train, _ = compute_metrics_and_curves(train_true, train_scores)
    metrics_test, lc_test = compute_metrics_and_curves(test_true, test_scores, np.linspace(0.01, 0.99, 99))
    
    # 5. Feature Importance
    baseline_auc = metrics_test["auc_roc"]
    fi_df = compute_permutation_importance(
        model=model, 
        dataset_class=dataset_cls, # Pass the class so FI function knows how to wrap data
        X_test=X_test, 
        y_test=y_test, 
        baseline_score=baseline_auc, 
        feature_names=feature_names, 
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    
    # 6. Saving
    fi_path = os.path.join(model_save_dir, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"  Saved feature importance to {fi_path}")

    full_metrics = {"train": metrics_train, "test": metrics_test}
    with open(os.path.join(model_save_dir, "metrics_train_test.json"), "w") as f:
        json.dump(full_metrics, f, indent=2)
        
    if lc_test is not None:
        lc_test.to_csv(os.path.join(model_save_dir, "learning_curve_test.csv"), index=False)
        
    torch.save(model.state_dict(), os.path.join(model_save_dir, "model.pth"))
    
    # Return metrics for comparison CSV
    return {
        "model": model_name,
        "bce_loss_overall": metrics_test["bce_loss_overall"],
        "accuracy_at_0.5": metrics_test["accuracy_at_0.5"],
        "f1_at_0.5": metrics_test["f1_at_0.5"],
        "best_f1": metrics_test["best_f1"],
        "roc_auc": metrics_test["auc_roc"]
    }

# --- Main Execution ---

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs_combined", f"run_{timestamp}") # Changed dir name
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving outputs to: {run_dir}")

    # 1. LOAD DATA
    ds = RoADDataset(normalize=False)
    
    if hasattr(ds, 'feature_names') and len(ds.feature_names) == 86:
        feature_names = ds.feature_names
        print("Using feature names from Dataset class.")
    else:
        print("Using manual list of feature names.")
        feature_names = [
            "robot_action_id", "apparent_power", "current", "frequency", "phase_angle",
            "power", "power_factor", "reactive_power", "voltage", 
            "j1_acc_x", "j1_acc_y", "j1_acc_z", "j1_gyro_x", "j1_gyro_y", "j1_gyro_z",
            "j1_qw", "j1_qx", "j1_qy", "j1_qz", "j1_temp",
            "j2_acc_x", "j2_acc_y", "j2_acc_z", "j2_gyro_x", "j2_gyro_y", "j2_gyro_z",
            "j2_qw", "j2_qx", "j2_qy", "j2_qz", "j2_temp",
            "j3_acc_x", "j3_acc_y", "j3_acc_z", "j3_gyro_x", "j3_gyro_y", "j3_gyro_z",
            "j3_qw", "j3_qx", "j3_qy", "j3_qz", "j3_temp",
            "j4_acc_x", "j4_acc_y", "j4_acc_z", "j4_gyro_x", "j4_gyro_y", "j4_gyro_z",
            "j4_qw", "j4_qx", "j4_qy", "j4_qz", "j4_temp",
            "j5_acc_x", "j5_acc_y", "j5_acc_z", "j5_gyro_x", "j5_gyro_y", "j5_gyro_z",
            "j5_qw", "j5_qx", "j5_qy", "j5_qz", "j5_temp",
            "j6_acc_x", "j6_acc_y", "j6_acc_z", "j6_gyro_x", "j6_gyro_y", "j6_gyro_z",
            "j6_qw", "j6_qx", "j6_qy", "j6_qz", "j6_temp",
            "j7_acc_x", "j7_acc_y", "j7_acc_z", "j7_gyro_x", "j7_gyro_y", "j7_gyro_z",
            "j7_qw", "j7_qx", "j7_qy", "j7_qz", "j7_temp"
        ]

    # 2. PREPARE RAW DATA
    base_train = list(ds.sets["training"])
    control = list(ds.sets.get("control", []))
    train_group = base_train + control
    
    others = {
        "collision": list(ds.sets.get("collision", [])),
        "weight": list(ds.sets.get("weight", [])),
        "velocity": list(ds.sets.get("velocity", []))
    }

    print("Computing normalization stats on training data...")
    if len(train_group) == 0: raise ValueError("Training group is empty!")
    feat_min, feat_max = compute_minmax(train_group, NUM_FEATURES)
    
    train_norm = normalize_records(train_group, feat_min, feat_max, NUM_FEATURES)
    others_norm = {k: normalize_records(v, feat_min, feat_max, NUM_FEATURES) for k, v in others.items()}

    # 3. CREATE WINDOWS (Shared for both models)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    # Normal Data Split
    if len(train_norm) > 1:
        tr_recs, te_recs = train_test_split(train_norm, test_size=0.3, random_state=42)
    else:
        tr_recs, te_recs = train_norm, []
        
    xt, yt = create_sequences(tr_recs, WINDOW_SIZE, TRAIN_STRIDE, NUM_FEATURES, max_windows=4000)
    xe, ye = create_sequences(te_recs, WINDOW_SIZE, TEST_STRIDE, NUM_FEATURES)
    if len(xt)>0: X_train_list.append(xt); y_train_list.append(yt)
    if len(xe)>0: X_test_list.append(xe); y_test_list.append(ye)

    # Anomaly Data Split
    for name, recs in others_norm.items():
        if not recs: continue
        if len(recs) > 1:
            tr_recs, te_recs = train_test_split(recs, test_size=0.3, random_state=42)
        else:
            print(f"Warning: '{name}' has only {len(recs)} record(s). Using all for training.")
            tr_recs = recs; te_recs = []
        
        xt, yt = create_sequences(tr_recs, WINDOW_SIZE, TRAIN_STRIDE, NUM_FEATURES, max_windows=4000)
        xe, ye = create_sequences(te_recs, WINDOW_SIZE, TEST_STRIDE, NUM_FEATURES)
        if len(xt)>0: X_train_list.append(xt); y_train_list.append(yt)
        if len(xe)>0: X_test_list.append(xe); y_test_list.append(ye)
            
    if len(X_train_list) == 0: raise ValueError("No training data!")
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    if len(X_test_list) > 0:
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        print("Warning: No test data! Metrics will be only on Train.")
        X_test = X_train; y_test = y_train
    
    print(f"\nTotal Train Windows: {X_train.shape[0]}")
    print(f"Total Test Windows : {X_test.shape[0]}")
    
    # 4. RUN MODELS
    comparison_rows = []
    
    # --- Model A: CNN-1D ---
    cnn_model = Simple1DCNN(NUM_FEATURES, WINDOW_SIZE)
    res_cnn = train_and_evaluate_model(
        "CNN_1D", cnn_model, CNNDataset, 
        X_train, y_train, X_test, y_test, 
        run_dir, feature_names
    )
    comparison_rows.append(res_cnn)
    
    # --- Model B: MLP ---
    mlp_model = SimpleMLP(NUM_FEATURES, WINDOW_SIZE)
    res_mlp = train_and_evaluate_model(
        "MLP", mlp_model, MLPDataset, 
        X_train, y_train, X_test, y_test, 
        run_dir, feature_names
    )
    comparison_rows.append(res_mlp)

    # 5. SAVE COMPARISON
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(run_dir, "models_comparison_test.csv")
    comp_df.to_csv(comp_path, index=False)
    
    print(f"\nPipeline finished. Comparison saved to {comp_path}")

if __name__ == "__main__":
    main()