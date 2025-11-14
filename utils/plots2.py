import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_and_save(df, x, y, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_multi_and_save(df, x, ys, labels, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for y, label in zip(ys, labels):
        plt.plot(df[x], df[y], marker="o", label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_all_plots(
    log_path: str = "../artifacts_framelevel/training_log.csv",
    out_dir: str = "../artifacts_framelevel/plots",
    confusion_matrix_values=None,
):
    """
    Generates all training plots from training_log.csv and optionally a confusion matrix.

    Parameters
    ----------
    log_path : str
        Path to the CSV log file.
    out_dir : str
        Directory where plots will be saved.
    confusion_matrix_values : np.ndarray | None
        If provided, saves a confusion matrix plot.
        If None, tries to read val_tn/val_fp/val_fn/val_tp from the log file.
    """

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    if not Path(log_path).exists():
        raise FileNotFoundError(f"Could not find log file: {log_path}")

    df = pd.read_csv(log_path)

    # 1. Loss (Train vs Val)
    plot_multi_and_save(
        df,
        "epoch",
        ["train_loss", "val_loss"],
        ["Train Loss", "Validation Loss"],
        "Training vs Validation Loss",
        "Loss",
        out_dir / "loss.png",
    )

    # 2. AUPRC
    plot_and_save(
        df,
        "epoch",
        "val_auprc",
        "Validation AUPRC",
        "AUPRC",
        out_dir / "auprc.png",
    )

    # 3. F1 Score (threshold 0.5 and Best F1)
    plot_multi_and_save(
        df,
        "epoch",
        ["val_f1@0.5", "val_best_f1"],
        ["F1@0.5", "Best F1"],
        "F1 Score",
        "F1",
        out_dir / "f1_scores.png",
    )

    # 4. Accuracy for classe (0 e 1)
    plot_multi_and_save(
        df,
        "epoch",
        ["val_acc_class0", "val_acc_class1"],
        ["Class 0 Accuracy", "Class 1 Accuracy"],
        "Accuracy per Classe",
        "Accuracy",
        out_dir / "class_accuracy.png",
    )

    # 5. Precision - Recall - Accuracy
    plot_multi_and_save(
        df,
        "epoch",
        ["val_prec@0.5", "val_rec@0.5", "val_acc@0.5"],
        ["Precision@0.5", "Recall@0.5", "Accuracy@0.5"],
        "Precision - Recall - Accuracy",
        "Score",
        out_dir / "prec_rec_acc.png",
    )

    # 6. Learning Rate
    plot_and_save(
        df,
        "epoch",
        "lr",
        "Learning Rate",
        "LR",
        out_dir / "learning_rate.png",
    )

    # 7. Confusion Matrix
    if confusion_matrix_values is None:
        required_cols = {"val_tn", "val_fp", "val_fn", "val_tp"}
        if required_cols.issubset(df.columns):
            last = df.iloc[-1]
            cm = np.array(
                [
                    [int(last["val_tn"]), int(last["val_fp"])],
                    [int(last["val_fn"]), int(last["val_tp"])],
                ]
            )
            confusion_matrix_values = cm

    if confusion_matrix_values is not None:
        plot_confusion_matrix(
            np.array(confusion_matrix_values),
            out_dir / "confusion_matrix.png",
            title="Confusion Matrix (Full Validation)",
        )

    print(f"Plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    generate_all_plots()
