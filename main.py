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
    
        self.index_map = []

        for rec_idx, arr in enumerate(records):
            N = arr.shape[0]
            for start in range(0, N - window_size + 1, stride):
                self.index_map.append((rec_idx, start))

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        rec_idx, start = self.index_map[idx]
        arr = self.records[rec_idx]

        window = arr[start:start + self.window_size, :self.num_features]
        window = window.T

        return torch.from_numpy(window)



def compute_minmax(train_records, num_features = 86):
    # transform all in one big array
    arrays = [np.asarray(rec)[:, :num_features] for rec in train_records]
    big = np.concatenate(arrays, axis = 0)

    feat_min = big.min(axis = 0)
    feat_max = big.max(axis = 0)

    return feat_min.astype(np.float32), feat_max.astype(np.float32) # not including label 87 

def minmax_scale(arr, min_vals, max_vals, eps = 1e-8):
    arr = arr.astype(np.float32)
    num = arr - min_vals
    den = (max_vals - min_vals) + eps
    return 2.0 * (num / den) - 1.0


def main():
    ds = RoADDataset(normalize = False)

    print("Subsets available:", ds.sets.keys())

    train_records = ds.sets["training"]

    print(f"Number of training records: {len(train_records)}")

    first = train_records[0]
    arr = np.asarray(first)

    print("Shape of first training record:", arr.shape) # (N, C)
    print("Feature number", arr.shape[1])

    print("Computing global min/max over the training set...")
    feat_min, feat_max = compute_minmax(train_records, num_features = 86)
    train_norm = [
    minmax_scale(np.asarray(rec)[:, :86], feat_min, feat_max)
    for rec in train_records
    ]

    print("Global feature-wise min (first 5 features):", feat_min[:5])
    print("Global feature-wise max (first 5 features):", feat_max[:5])

    dataset = RoADWindowedDataset(train_norm, window_size = 256, stride = 128)
    print("Total windows:", len(dataset))

    sample = dataset[0]
    print("Sample shape:", sample.shape)

if __name__ == "__main__":
    main()