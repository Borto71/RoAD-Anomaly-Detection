import os
import numpy as np
import pandas as pd
from RoADDataset import Dataset

output_dir = "dataset"
os.makedirs(output_dir, exist_ok = True)


dataset = Dataset(normalize = True)  


for subset_name, recordings in dataset.sets.items():
    subset_dir = os.path.join(output_dir, subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    print(f"Extraction from the subset: {subset_name} ({len(recordings)} registration)")


    for i, rec in enumerate(recordings):
        
        filename = f"{subset_name}_{i:04d}.csv"
        filepath = os.path.join(subset_dir, filename)

        
        df = pd.DataFrame(rec)
        df.to_csv(filepath, index = False, header = False)

print("\nExtraction completed in the folder './dataset'")
