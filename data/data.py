import os, numpy as np
from RoADDataset import Dataset

dataset = Dataset() 

out = "dataset"
os.makedirs(out, exist_ok = True)

for split, recs in dataset.sets.items():
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok = True)

    for i, arr in enumerate(recs):
        np.save(os.path.join(d, f"rec_{i:04d}.npy"), arr)

print("Exported in:", out, "the files are already normalized and stored as numpy arrays.")
