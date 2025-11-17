from RoADDataset import Dataset as RoADDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from models.cnn_1d import RoADCNN1D

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


if __name__ == "__main__":
    main()