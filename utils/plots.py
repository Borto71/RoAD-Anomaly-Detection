import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path):
    path.mkdir(parents = True, exist_ok = True)


def plot_and_save(df, x, y, title, ylabel, save_path):
    plt.figure(figsize = (8, 5))
    plt.plot(df[x], df[y], marker = "o")
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
        plt.plot(df[x], df[y], marker = "o", label = label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, save_path, title = "Confusion Matrix"):
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_all_plots(
    log_path: str = "../artifacts/training_log.csv",
    out_dir: str = "../artifacts/plots",
    confusion_matrix_values = None,
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

    # Loss Plot (Train vs Val)
    plot_multi_and_save(
        df,
        "epoch",
        ["train_loss", "val_loss"],
        ["Train Loss", "Validation Loss"],
        "Training vs Validation Loss",
        "Loss",
        out_dir / "loss.png",
    )

    # 2. Learning Curve - Accuracy (Train vs Val)
    if {"train_acc@0.5", "val_acc@0.5"}.issubset(df.columns):
        plot_multi_and_save(
            df,
            "epoch",
            ["train_acc@0.5", "val_acc@0.5"],
            ["Train Accuracy@0.5", "Validation Accuracy@0.5"],
            "Training vs Validation Accuracy",
            "Accuracy",
            out_dir / "accuracy.png",
        )

    # 3. Validation AUPRC
    if "val_auprc" in df.columns:
        plot_and_save(
            df,
            "epoch",
            "val_auprc",
            "Validation AUPRC",
            "AUPRC",
            out_dir / "val_auprc.png",
        )

    # 4. F1 Score (Train F1@0.5, Val F1@0.5, Val Best F1)
    f1_cols = ["train_f1@0.5", "val_f1@0.5", "val_best_f1"]
    if set(f1_cols).issubset(df.columns):
        plot_multi_and_save(
            df,
            "epoch",
            f1_cols,
            ["Train F1@0.5", "Val F1@0.5", "Val Best F1"],
            "F1 Score",
            "F1",
            out_dir / "f1_scores.png",
        )

    # 5. Accuracy per Classe (Train + Val)
    acc_class_cols = [
        "train_acc_class0",
        "train_acc_class1",
        "val_acc_class0",
        "val_acc_class1",
    ]
    if set(acc_class_cols).issubset(df.columns):
        plot_multi_and_save(
            df,
            "epoch",
            acc_class_cols,
            [
                "Train Class 0 Accuracy",
                "Train Class 1 Accuracy",
                "Val Class 0 Accuracy",
                "Val Class 1 Accuracy",
            ],
            "Accuracy per Class",
            "Accuracy",
            out_dir / "class_accuracy_train_val.png",
        )

    # 6. Loss per Classe (Train + Val)
    loss_class_cols = [
        "train_loss_class0",
        "train_loss_class1",
        "val_loss_class0",
        "val_loss_class1",
    ]
    if set(loss_class_cols).issubset(df.columns):
        plot_multi_and_save(
            df,
            "epoch",
            loss_class_cols,
            [
                "Train Class 0 Loss",
                "Train Class 1 Loss",
                "Val Class 0 Loss",
                "Val Class 1 Loss",
            ],
            "Loss per Class",
            "Loss",
            out_dir / "class_loss_train_val.png",
        )

    # 7. Precision - Recall - F1 (Train)
    prec_rec_f1_train_cols = [
        "train_prec@0.5",
        "train_rec@0.5",
        "train_f1@0.5",
    ]
    if set(prec_rec_f1_train_cols).issubset(df.columns):
        plot_multi_and_save(
            df,
            "epoch",
            prec_rec_f1_train_cols,
            ["Train Precision@0.5", "Train Recall@0.5", "Train F1@0.5"],
            "Precision - Recall - F1 (Train)",
            "Score",
            out_dir / "prec_rec_f1_train.png",
        )

    # 8. Precision - Recall - F1 (Validation)
    prec_rec_f1_val_cols = [
        "val_prec@0.5",
        "val_rec@0.5",
        "val_f1@0.5",
    ]
    if set(prec_rec_f1_val_cols).issubset(df.columns):
        plot_multi_and_save(
            df,
            "epoch",
            prec_rec_f1_val_cols,
            ["Val Precision@0.5", "Val Recall@0.5", "Val F1@0.5"],
            "Precision - Recall - F1 (Validation)",
            "Score",
            out_dir / "prec_rec_f1_val.png",
        )

    # 9. Learning Rate
    if "lr" in df.columns:
        plot_and_save(
            df,
            "epoch",
            "lr",
            "Learning Rate",
            "LR",
            out_dir / "learning_rate.png",
        )

    # 10. Confusion Matrix
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
