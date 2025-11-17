from RoADDataset import Dataset as RoADDataset
from torch.utils.data import Dataset
import numpy as np
import torch


# from models.cnn_1d import CNN1D

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

    print("Subsets available:", ds.sets.keys())

    train_records = ds.sets["training"]
    # assumo che esistano anche questi; se i nomi differiscono, li aggiusti:
    val_records   = ds.sets.get("validation", [])
    test_records  = ds.sets.get("test", [])

    print(f"Number of training records: {len(train_records)}")
    print(f"Number of validation records: {len(val_records)}")
    print(f"Number of test records: {len(test_records)}")

    first = train_records[0]
    arr = np.asarray(first)
    print("Shape of first training record:", arr.shape) # (N, C)
    print("Feature number", arr.shape[1])

    # 1) Compute global min/max from TRAIN ONLY
    print("Computing global min/max over the training set...")
    feat_min, feat_max = compute_minmax(train_records, num_features = 86)

    # 2) Normalize all subsets using train stats
    train_norm = normalize_records(train_records, feat_min, feat_max, num_features=86)
    val_norm   = normalize_records(val_records,   feat_min, feat_max, num_features=86)
    test_norm  = normalize_records(test_records,  feat_min, feat_max, num_features=86)

    # 3) Show some stats
    print("Global feature-wise min (first 5 features):", feat_min[:5])
    print("Global feature-wise max (first 5 features):", feat_max[:5])

    # 4) Create windowed datasets (per ora solo train/val)
    train_dataset = RoADWindowedDataset(
        train_norm,
        window_size = 256,
        stride = 128,
        num_features = 86,
        include_label = False,   # unsupervised, no label in training
    )

    val_dataset = RoADWindowedDataset(
        val_norm,
        window_size = 256,
        stride = 128,
        num_features = 86,
        include_label = True,    # on the val dataset we need the label
    )

    print("Total train windows:", len(train_dataset))
    print("Total val windows:", len(val_dataset))

    # Check on the sample
    x_train = train_dataset[0]              # only features
    x_val, y_val = val_dataset[0]          # features + label windowed

    print("Sample train window shape:", x_train.shape)  # (C, T)
    print("Sample val window shape:", x_val.shape)      # (C, T)
    print("Sample val window label:", y_val.item())



if __name__ == "__main__":
    main()