from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

try:
    from RoADDataset import Dataset
except ImportError:
    sys.stderr.write(
        " RoADDataset not found. Install:\n"
        "    pip install git+https://gitlab.com/AlessioMascolini/roaddataset/\n"
    )
    sys.exit(1)

# Intestazioni delle 87 feature (dal README del repo)
HEADERS = [
    # global
    "robot_action_id", "apparent_power", "current", "frequency", "phase_angle",
    "power", "power_factor", "reactive_power", "voltage",
    # joint 1
    "j1_acc_x","j1_acc_y","j1_acc_z","j1_gyro_x","j1_gyro_y","j1_gyro_z",
    "j1_qw","j1_qx","j1_qy","j1_qz","j1_temp",
    # joint 2
    "j2_acc_x","j2_acc_y","j2_acc_z","j2_gyro_x","j2_gyro_y","j2_gyro_z",
    "j2_qw","j2_qx","j2_qy","j2_qz","j2_temp",
    # joint 3
    "j3_acc_x","j3_acc_y","j3_acc_z","j3_gyro_x","j3_gyro_y","j3_gyro_z",
    "j3_qw","j3_qx","j3_qy","j3_qz","j3_temp",
    # joint 4
    "j4_acc_x","j4_acc_y","j4_acc_z","j4_gyro_x","j4_gyro_y","j4_gyro_z",
    "j4_qw","j4_qx","j4_qy","j4_qz","j4_temp",
    # joint 5
    "j5_acc_x","j5_acc_y","j5_acc_z","j5_gyro_x","j5_gyro_y","j5_gyro_z",
    "j5_qw","j5_qx","j5_qy","j5_qz","j5_temp",
    # joint 6
    "j6_acc_x","j6_acc_y","j6_acc_z","j6_gyro_x","j6_gyro_y","j6_gyro_z",
    "j6_qw","j6_qx","j6_qy","j6_qz","j6_temp",
    # joint 7
    "j7_acc_x","j7_acc_y","j7_acc_z","j7_gyro_x","j7_gyro_y","j7_gyro_z",
    "j7_qw","j7_qx","j7_qy","j7_qz","j7_temp",
    # anomaly label
    "anomaly"
]

def main(out_dir: Path = Path("dataset")):
    out_dir.mkdir(parents = True, exist_ok = True)

    # Load the dataset with normalization
    ds = Dataset(normalize = True)  # using the internal normalization
    subsets = ds.sets.keys()

    meta = {
        "normalized": True,
        "subsets": {},
        "note": "Channel 87 (anomaly) isn't always included as last column."
    }

    for subset in subsets:
        records = ds.sets[subset]  # list of recordings
        subset_dir = out_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        exported = []
        for i, rec in enumerate(records):
            # Each element can be ndarray, list, tuple, or custom object.
            # A few checks to convert to ndarray
            if isinstance(rec, np.ndarray):
                arr = rec
            elif hasattr(rec, "shape"):
                arr = np.asarray(rec)
            elif isinstance(rec, (list, tuple)):
                # take the first element
                arr = np.asarray(rec[0])
            else:
                raise TypeError(f"Registration type: {subset}[{i}]: {type(rec)}")

            
            n_feat = arr.shape[1]
            headers = HEADERS[:n_feat]

            df = pd.DataFrame(arr, columns=headers)
            out_csv = subset_dir / f"rec_{i:04d}.csv"
            df.to_csv(out_csv, index=False)
            exported.append(str(out_csv))

        meta["subsets"][subset] = {
            "count": len(exported),
            "files": exported
        }

    # Save a summary of metadata
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Data extraction completed in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
