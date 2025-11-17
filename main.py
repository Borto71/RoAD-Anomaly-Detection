from RoADDataset import Dataset as RoADDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from models.cnn_1d import RoADCNN1D
from models.cnn_1d import build_model

class RoADWindowedDataset(Dataset):
    def __init__(self, records, window_size, stride, num_features = 86, include_label = False):
        self.records = records
        self.window_size = window_size
        self.stride = stride
        self.num_features = num_features
        self.include_label = include_label
    
        self.index_map = [] # list of (record_index, start_index) tuples

        for rec_idx, arr in enumerate(records): # arr is ndarray of shape (N, C)
            N = arr.shape[0]
            for start in range(0, N - window_size + 1, stride):
                self.index_map.append((rec_idx, start))

    def __len__(self): 
        return len(self.index_map)
    
    def __getitem__(self, idx): # returns a window
        rec_idx, start = self.index_map[idx]
        arr = self.records[rec_idx]

        window = arr[start:start + self.window_size, :self.num_features]
        window = window.T
        x_tensor = torch.from_numpy(window.astype(np.float32))

        if not self.include_label:
            return x_tensor

        window_labels = arr[start:start + self.window_size, self.num_features]  # (T,)
        label_value = 1.0 if window_labels.max() > 0 else 0.0
        y_tensor = torch.tensor(label_value, dtype = torch.float32)
        return x_tensor, y_tensor


def compute_minmax(train_records, num_features = 86):
    # transform all in one big array 
    arrays = [np.asarray(rec)[:, :num_features] for rec in train_records]
    big = np.concatenate(arrays, axis = 0)

    feat_min = big.min(axis = 0) # feature-wise min
    feat_max = big.max(axis = 0) # feature-wise max

    return feat_min.astype(np.float32), feat_max.astype(np.float32) # not including label 87 

def minmax_scale(arr, min_vals, max_vals, eps = 1e-8): # scale to [-1, 1]
    arr = arr.astype(np.float32)
    num = arr - min_vals
    den = (max_vals - min_vals) + eps
    return 2.0 * (num / den) - 1.0 # return scaled array

def normalize_records(records, feat_min, feat_max, num_features = 86):
    """
    """
    norm_recs = []
    for rec in records:
        arr = np.asarray(rec)
        feats = arr[:, :num_features]                     # (N, 86)
        feats_scaled = minmax_scale(feats, feat_min, feat_max)  # (N, 86)

        if arr.shape[1] > num_features:
            label = arr[:, num_features:num_features + 1]   # (N, 1)
            arr_scaled = np.concatenate([feats_scaled, label], axis = 1)
        else:
            arr_scaled = feats_scaled

        norm_recs.append(arr_scaled)
    return norm_recs

def compute_center(train_loader, model, device):
    """
    Calculate the center c in the space of the embedding, 
    c is calculate on all the window of the training set
    """
    model.eval()
    n_samples = 0
    c_sum = None

    with torch.no_grad():
        for batch in train_loader:
            # batch = x, because in the training we don't have the "anomaly" label
            x = batch.to(device)  # shape: (B, C, T)

            # embedding in windows 
            z = model.forward_features(x)  # shape: (B, F)

            # storing the sum
            if c_sum is None:
                c_sum = z.sum(dim = 0)
            else:
                c_sum += z.sum(dim = 0)

            n_samples += z.shape[0]

    # global even
    c = c_sum / n_samples
    # we secure a clean tensor
    c = c.detach().to(device)
    return c


def train_one_epoch(model, train_loader, optimizer, c, device):
    """
    Execute one epoch training Deep SVDD, 
    mimimize the distance of the embedding from the center c.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    n_samples = 0

    for x in train_loader:
        # x: (B, C, T)
        x = x.to(device)

        optimizer.zero_grad()

        # 1) embedding of the window
        z = model.forward_features(x)  # (B, F)

        # 2) loss = even of the quadratic distance of c
        # c: (F,)
        dist_sq = (z - c)**2          # (B, F)
        loss = dist_sq.sum(dim = 1).mean()  # scalar

        # 3) backprop
        loss.backward()
        optimizer.step()

        # intermediate log
        batch_size = x.shape[0]
        running_loss += loss.item()
        n_batches += 1
        n_samples += batch_size

    avg_loss = running_loss / max(1, n_batches)
    return avg_loss, n_batches, n_samples

from sklearn.metrics import roc_auc_score, average_precision_score

def collect_scores_and_labels(model, loader, c, device):
    """
   Pass all the batch of the loader in the model, elaborate:
   - anomaly score = quadratic distance from c
   - label of the window 
   Return two numpy arrays: scores, labels
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # here the loader test give (x, y)
            x, y = batch
            x = x.to(device)

            # embedding
            z = model.forward_features(x)  # (B, F)

            # score = quadratic distance from the center
            dist_sq = (z - c)**2          # (B, F)
            scores = dist_sq.sum(dim = 1)   # (B,)

            all_scores.append(scores.cpu().numpy())
            all_labels.append(y.numpy())

    if len(all_scores) == 0:
        return None, None

    scores = np.concatenate(all_scores, axis = 0)
    labels = np.concatenate(all_labels, axis = 0)
    return scores, labels



def main():
    ds = RoADDataset(normalize = False)

    print("Subsets available:", ds.sets.keys()) # show the available subset for training and validation

    # 1) Train subset, used only for training
    train_records = ds.sets["training"]

    # 2) Test set, subset of collision, weight, velocity and control
    collision_records = ds.sets.get("collision", [])
    weight_records    = ds.sets.get("weight",    [])
    velocity_records  = ds.sets.get("velocity",  [])
    control_records   = ds.sets.get("control",   [])

    print(f"Number of training records:  {len(train_records)}")
    print(f"Number of collision records: {len(collision_records)}")
    print(f"Number of weight records:    {len(weight_records)}")
    print(f"Number of velocity records:  {len(velocity_records)}")
    print(f"Number of control records:   {len(control_records)}")

    first = train_records[0]
    arr = np.asarray(first)
    print("Shape of first training record:", arr.shape) # (N, C)
    print("Feature number", arr.shape[1])

        # 1) Compute global min/max from TRAIN ONLY
    print("Computing global min/max over the training set...")
    feat_min, feat_max = compute_minmax(train_records, num_features = 86)

    # 2) Normalize all subsets using train stats
    train_norm    = normalize_records(train_records,    feat_min, feat_max, num_features = 86)
    collision_norm = normalize_records(collision_records, feat_min, feat_max, num_features = 86)
    weight_norm    = normalize_records(weight_records,    feat_min, feat_max, num_features = 86)
    velocity_norm  = normalize_records(velocity_records,  feat_min, feat_max, num_features = 86)
    control_norm   = normalize_records(control_records,   feat_min, feat_max, num_features = 86)

    # 3) Show some stats
    print("Global feature-wise min (first 5 features):", feat_min[:5])
    print("Global feature-wise max (first 5 features):", feat_max[:5])

        # 4) Create windowed datasets

    # TRAIN: unsupervised, no label
    train_dataset = RoADWindowedDataset(
        train_norm,
        window_size = 256,
        stride = 128,
        num_features = 86,
        include_label = False,
    )

    # TEST SCENARIOS: here we have the label
    collision_dataset = RoADWindowedDataset(
        collision_norm,
        window_size = 256,
        stride = 128,
        num_features = 86,
        include_label = True,
    )

    weight_dataset = RoADWindowedDataset(
        weight_norm,
        window_size = 256,
        stride = 128,
        num_features = 86,
        include_label = True,
    )

    velocity_dataset = RoADWindowedDataset(
        velocity_norm,
        window_size = 256,
        stride = 128,
        num_features = 86,
        include_label = True,
    )

    control_dataset = RoADWindowedDataset(
        control_norm,
        window_size = 256,
        stride = 128,
        num_features = 86,
        include_label = True,
    )

    print("Total train windows:", len(train_dataset))
    print("Total collision windows:", len(collision_dataset))
    print("Total weight windows:", len(weight_dataset))
    print("Total velocity windows:", len(velocity_dataset))
    print("Total control windows:", len(control_dataset))

    # Check on a sample
    x_train = train_dataset[0]
    print("Sample train window shape:", x_train.shape)  # (C, T)

    if len(collision_dataset) > 0:
        x_col, y_col = collision_dataset[0]
        print("Sample collision window shape:", x_col.shape)
        print("Sample collision window label:", y_col.item())

    # DataLoader Hyperparameters
    batch_size_train = 128
    batch_size_eval = 256
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    # Train Dataloaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True, drop_last = False, num_workers = num_workers, pin_memory = pin_memory)

    # Test DataLoaders
    collision_loader = DataLoader(collision_dataset, batch_size = batch_size_eval, shuffle = False, drop_last = False, num_workers = num_workers, pin_memory = pin_memory)

    weight_loader = DataLoader(collision_dataset, batch_size = batch_size_eval, shuffle = False, drop_last = False, num_workers = num_workers, pin_memory = pin_memory)

    velocity_loader = DataLoader(collision_dataset, batch_size = batch_size_eval, shuffle = False, drop_last = False, num_workers = num_workers, pin_memory = pin_memory)

    control_loader = DataLoader(collision_dataset, batch_size = batch_size_eval, shuffle = False, drop_last = False, num_workers = num_workers, pin_memory = pin_memory)

    # Dictionary for dataloaders
    test_loaders = { 
        "collision": collision_loader,
        "weight": weight_loader,
        "velocity": velocity_loader,
        "control": control_loader,
    }

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # building the model
    model = build_model(in_channels = 86, base_channels = 64, dropout = 0.1, return_frame_logits = False).to(device)

    # Optimizer for Deep SVDD
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-6)

    # Cumpute center c in feature space
    print("Computing center c from training data embeddings...")
    c = compute_center(train_loader, model, device)
    print("Center c shape", c.shape)

    # Training Loop Deep SVDD
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        avg_loss, n_batches, n_samples = train_one_epoch(model, train_loader, optimizer, c, device)
        print(
            f"Epoch {epoch:03d}/{num_epochs} "
            f"- loss: {avg_loss:.6f} "
            f"- batches: {n_batches} "
            f"- samples: {n_samples}"
        )


    # Evaluation scenario: COLLISION
    print("\nEvaluation scenario: COLLISION")

    coll_scores, coll_labels = collect_scores_and_labels(model, collision_loader, c, device)

    if coll_scores is not None:
        auc_roc_coll = roc_auc_score(coll_labels, coll_scores)
        auc_pr_coll = average_precision_score(coll_labels, coll_scores)

        print(f"Collision - ROC-AUC: {auc_roc_coll:.4f}")
        print(f"Collision - PR-AUC : {auc_pr_coll:.4f}")
    else:
        print("No collision data available for evaluation.")


    # Evaluation scenario: WEIGHT vs CONTROL
    print("\nEvaluating scenario: WEIGHT vs CONTROL")

    weight_scores, weight_labels = collect_scores_and_labels(
        model, weight_loader, c, device
    )
    control_scores, control_labels = collect_scores_and_labels(
        model, control_loader, c, device
    )

    if weight_scores is not None and control_scores is not None:
        y_weight  = np.ones_like(weight_scores)   # all anomalies
        y_control = np.zeros_like(control_scores) # all normal

        scores_wc = np.concatenate([weight_scores, control_scores], axis = 0)
        labels_wc = np.concatenate([y_weight, y_control], axis = 0)

        auc_roc_weight = roc_auc_score(labels_wc, scores_wc)
        auc_pr_weight  = average_precision_score(labels_wc, scores_wc)

        print(f"Weight vs Control - ROC-AUC: {auc_roc_weight:.4f}")
        print(f"Weight vs Control - PR-AUC : {auc_pr_weight:.4f}")
    else:
        print("Not enough data for WEIGHT vs CONTROL evaluation.")


    # Evaluation scenario: VELOCITY vs CONTROL
        print("\nEvaluating scenario: VELOCITY vs CONTROL")

    velocity_scores, velocity_labels = collect_scores_and_labels(
        model, velocity_loader, c, device
    )

    if velocity_scores is not None and control_scores is not None:
        y_velocity = np.ones_like(velocity_scores)
        y_control  = np.zeros_like(control_scores)

        scores_vc = np.concatenate([velocity_scores, control_scores], axis=0)
        labels_vc = np.concatenate([y_velocity,     y_control],      axis=0)

        auc_roc_velocity = roc_auc_score(labels_vc, scores_vc)
        auc_pr_velocity  = average_precision_score(labels_vc, scores_vc)

        print(f"Velocity vs Control - ROC-AUC: {auc_roc_velocity:.4f}")
        print(f"Velocity vs Control - PR-AUC : {auc_pr_velocity:.4f}")
    else:
        print("Not enough data for VELOCITY vs CONTROL evaluation.")    

    print("Normal scores mean:", coll_scores[coll_labels == 0].mean())
    print("Anomal scores mean:", coll_scores[coll_labels == 1].mean())


if __name__ == "__main__":
    main()