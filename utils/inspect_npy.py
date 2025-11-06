import numpy as np
import os
import sys
import pandas as pd

def inspect_npy(path):
    if not os.path.exists(path):
        print(f"The file '{path}' doesn't exist.")
        return
    
    try:
        if path.endswith(".npz"):
            data = np.load(path)
            print(f"\nFile NPZ with {len(data.files)} array:")
            for name in data.files:
                arr = data[name]
                print(f"\nArray '{name}': shape={arr.shape}, dtype={arr.dtype}")
                _preview_array(arr)
        elif path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            print(f"\nFile NPY: shape={arr.shape}, dtype={arr.dtype}")
            _preview_array(arr)
        else:
            print("the file format isn't .npy o .npz")
    except Exception as e:
        print(f"Error on downloading the file: {e}")

def _preview_array(arr):
    print("Preview of the data:")
    if arr.dtype.names:  # Structured array
        df = pd.DataFrame(arr)
        print(df.head())
    elif arr.ndim == 1:
        print(arr[:10])
    elif arr.ndim == 2:
        df = pd.DataFrame(arr)
        print(df.head())
    else:
        print("Array with dimension > 2, showing the first section:")
        print(arr.take(indices = range(min(3, arr.shape[0])), axis = 0))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Using: python inspect_npy.py <path_al_file.npy>")
    else:
        inspect_npy(sys.argv[1])
