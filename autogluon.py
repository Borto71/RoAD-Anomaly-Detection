from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularDataset, TabularPredictor
import torch


def load_metadata(meta_path: Path) -> dict:
    """Load the metadata.json file produced by data.py."""
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta


def to_path(metadata_root: Path, relative_path: str | Path) -> Path:
    """
    Convert a path from metadata.json into a real filesystem path.

    The metadata paths are typically like:
        'dataset/collision/rec_0000.csv'

    Our actual folder structure is:
        'data/dataset/collision/rec_0000.csv'

    So we prepend 'data/' (or the parent of metadata_root) if needed.
    """
    p = Path(relative_path)

    # If the path is already absolute, just return it.
    if p.is_absolute():
        return p

    return metadata_root.parent.parent / p


def load_subset_df(meta: dict, subset_name: str, meta_path: Path) -> pd.DataFrame:
    """
    Load all CSV files for a given subset (e.g. 'collision', 'control')
    and concatenate them into a single DataFrame.
    """
    if subset_name not in meta["subsets"]:
        raise KeyError(f"Subset '{subset_name}' not found in metadata.json")

    file_list = meta["subsets"][subset_name]["files"]

    dfs = []
    for fpath in file_list:
        full_path = to_path(meta_path, fpath)
        if not full_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {full_path}")
        df = pd.read_csv(full_path)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No CSV files found for subset '{subset_name}'")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def prepare_dataset(meta_path: Path) -> pd.DataFrame:
    """
    Load collision and control subsets and build a single DataFrame with:
        - all features from the CSVs
        - a binary label column: 1 if anomaly > 0, 0 otherwise
    """
    print(f"Loading metadata from: {meta_path}")
    meta = load_metadata(meta_path)

    print("Loading 'collision' subset...")
    df_collision = load_subset_df(meta, "collision", meta_path)
    print(f"Collision shape: {df_collision.shape}")

    print("Loading 'control' subset...")
    df_control = load_subset_df(meta, "control", meta_path)
    print(f"Control shape:   {df_control.shape}")

    # Concatenate both subsets into a single DataFrame
    df_all = pd.concat([df_collision, df_control], ignore_index=True)

    # Check that 'anomaly' column exists
    if "anomaly" not in df_all.columns:
        raise KeyError("Column 'anomaly' not found in the loaded data.")

    # Create binary label: 1 if anomaly > 0, else 0
    df_all["label"] = (df_all["anomaly"] > 0).astype(int)

    print("Label distribution (0 = normal, 1 = anomalous):")
    print(df_all["label"].value_counts())

    return df_all


def train_autogluon(df_all: pd.DataFrame, output_path: str = "AutogluonModels/road_baseline"):
    """
    Split the data into train/test, train an AutoGluon Tabular model,
    evaluate performance, and print leaderboard.
    We spingiamo al massimo l'uso della GPU:
      - controllo che la GPU sia disponibile
      - restringo i modelli a quelli GPU-friendly
      - passo num_gpus=1 a fit e ai singoli modelli
    """
    label = "label"

    # --- Check GPU availability early ---
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nessuna GPU CUDA rilevata da PyTorch. "
            "Verifica driver, CUDA e versione di PyTorch prima di eseguire questo script."
        )
    else:
        print(f"GPU disponibile: {torch.cuda.get_device_name(0)}")
        print(f"Numero di GPU visibili: {torch.cuda.device_count()}")

    # Split into train and test.
    # Note: this split is done at row level, not at recording level.
    print("Splitting data into train and test...")
    train_df, test_df = train_test_split(
        df_all,
        test_size=0.2,
        stratify=df_all[label],
        random_state=42,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Test  shape: {test_df.shape}")

    # Convert pandas DataFrame to AutoGluon TabularDataset
    train_data = TabularDataset(train_df)
    test_data = TabularDataset(test_df)

    # Hyperparameters: limitiamo i modelli a quelli che possono usare la GPU
    # (NN_TORCH usa PyTorch; CAT e XGB possono sfruttare la GPU se installati con supporto GPU)
    gpu_hyperparameters = {
        "NN_TORCH": {
            "ag_args_fit": {
                "num_gpus": 1,
            }
        },
        "CAT": {
            "ag_args_fit": {
                "num_gpus": 1,
            }
        },
        "XGB": {
            "ag_args_fit": {
                "num_gpus": 1,
            }
        },
    }

    # Define and train the AutoGluon TabularPredictor
    print("Training AutoGluon TabularPredictor (GPU-focused)...")
    predictor = TabularPredictor(
        label=label,
        problem_type="binary",      # 0/1 labels
        eval_metric="roc_auc",      # AUC is more informative than accuracy for anomaly detection
        path=output_path,
    ).fit(
        train_data=train_data,
        # rimuovo presets="best_quality" che aggiunge molti modelli CPU-only
        num_gpus=1,                 # indica che vogliamo usare 1 GPU
        hyperparameters=gpu_hyperparameters,
    )

    # Evaluate on the test set
    print("Evaluating predictor on test set...")
    perf = predictor.evaluate(test_data)
    print("Performance metrics:")
    for k, v in perf.items():
        print(f"  {k}: {v}")

    # Show leaderboard of models
    print("\nLeaderboard:")
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard)

    return predictor, leaderboard


def main():
    # Path to metadata.json produced by data.py
    meta_path = Path("data/dataset/metadata.json")

    # Prepare the dataset (collision + control with binary label)
    df_all = prepare_dataset(meta_path)

    # Train AutoGluon and print results
    train_autogluon(df_all, output_path="AutogluonModels/road_collision_vs_control")


if __name__ == "__main__":
    main()
