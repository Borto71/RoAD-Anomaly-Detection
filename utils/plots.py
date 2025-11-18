import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(script_dir)                
    return project_root

def get_latest_run(runs_root = None):
    if runs_root is None:
        project_root = get_project_root()
        runs_root = os.path.join(project_root, "runs")

    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"Runs directory '{runs_root}' not found")

    runs = [
        d for d in os.listdir(runs_root)
        if os.path.isdir(os.path.join(runs_root, d)) and d.startswith("run_")
    ]
    if not runs:
        raise FileNotFoundError(f"No run_* directories found inside '{runs_root}'.")

    runs.sort()
    latest = runs[-1]
    return os.path.join(runs_root, latest)


def load_metrics(run_dir):
    """
    Load the runs JSON
    """

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics,json not found in '{run_dir}'")
    
    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data

def plot_train_loss(data, run_dir):
    """
    Plot the curve of the training loss and save it as PNG
    """
    losses = data["train"]["loss_per_epoch"]
    epochs = np.arange(1, len(losses) + 1)

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)

    out_path = os.path.join(run_dir, "train_loss.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss plot to '{out_path}'")



def plot_auc_bars(data, run_dir):
    """
    Plot a bar chart of the ROC-AUC per scenario
    """
    scenarios = ["collision", "weight_vs_control", "velocity_vs_control"]
    labels = ["Collision", "Weight vs Control" , "Velocity vs Control"]

    aucs = []
    for s in scenarios:
        metrics = data.get(s, None)
        if metrics is None or metrics.get("roc_auc") is None:
            aucs.append(0.0)
        else:
            aucs.append(metrics["roc_auc"])
    
    x = np.arange(len(scenarios))

    plt.figure()
    plt.bar(x, aucs)
    plt.xticks(x, labels, rotation = 20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC per scenario")
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
    
    for i, v in enumerate(aucs):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

    out_path = os.path.join(run_dir, "roc_auc_bars.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC-AUC bar plot to {out_path}")


def plot_f1_bars(data, run_dir):
    """
    Plot a bar chart for the f1 score
    """
    scenarios = ["collision", "weight_vs_control", "velocity_vs_control"]
    labels = ["Collision", "Weight vs Control", "Velocity vs Control"]

    f1s = []
    for s in scenarios:
        metrics = data.get(s, None)
        if metrics is None or metrics.get("f1") is None:
            f1s.append(0.0)
        else:
            f1s.append(metrics["f1"])

    x = np.arange(len(scenarios))

    plt.figure()
    plt.bar(x, f1s)
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1-score")
    plt.title("F1-score per scenario (best threshold)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(f1s):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

    out_path = os.path.join(run_dir, "f1_bars.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved F1 bar plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot RoAD anomaly detection results.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to a specific run directory (e.g., runs/run_20250207_154355). "
             "If not provided, the latest run in 'runs/' will be used.",
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_dir = get_latest_run()

    print(f"Using run directory: {run_dir}")

    data = load_metrics(run_dir)

    # Plots
    plot_train_loss(data, run_dir)
    plot_auc_bars(data, run_dir)
    plot_f1_bars(data, run_dir)


if __name__ == "__main__":
    main()



    
