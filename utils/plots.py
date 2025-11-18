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


def plot_class_loss(data, run_dir):
    scenarios = ["collision", "weight_vs_control", "velocity_vs_control"]
    scen_labels = ["Collision", "Weight vs Control", "Velocity vs Control"]

    normal_losses = []
    anom_losses = []

    for s in scenarios:
        metrics = data.get(s, None)
        if metrics is None:
            normal_losses.append(0.0)
            anom_losses.append(0.0)
        else:
            ln = metrics.get("mean_loss_normal", None)
            la = metrics.get("mean_loss_anom", None)
            normal_losses.append(ln if ln is not None else 0.0)
            anom_losses.append(la if la is not None else 0.0)

    x = np.arange(len(scenarios))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, normal_losses, width, label="Normal")
    plt.bar(x + width/2, anom_losses,   width, label="Anomal")

    plt.xticks(x, scen_labels, rotation=20)
    plt.ylabel("Mean loss (score)")
    plt.title("Mean loss per class and scenario")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()


    out_path = os.path.join(run_dir, "class_loss_bars.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class loss bar plot to {out_path}")

def plot_class_accuracy(data, run_dir):
    scenarios = ["collision", "weight_vs_control", "velocity_vs_control"]
    scen_labels = ["Collision", "Weight vs Control", "Velocity vs Control"]

    acc_normals = []
    acc_anoms = []

    for s in scenarios:
        metrics = data.get(s, None)
        if metrics is None:
            acc_normals.append(0.0)
            acc_anoms.append(0.0)
        else:
            an = metrics.get("acc_normal", None)
            aa = metrics.get("acc_anom", None)
            acc_normals.append(an if an is not None else 0.0)
            acc_anoms.append(aa if aa is not None else 0.0)

    x = np.arange(len(scenarios))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, acc_normals, width, label="Normal")
    plt.bar(x + width/2, acc_anoms,   width, label="Anomal")

    plt.xticks(x, scen_labels, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Per-class accuracy per scenario (best threshold)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()

    for i, (n, a) in enumerate(zip(acc_normals, acc_anoms)):
        plt.text(i - width/2, n + 0.02, f"{n:.2f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + width/2, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(run_dir, "class_accuracy_bars.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class accuracy bar plot to {out_path}")


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

    data = load_metrics(run_dir)

    # Plots 
    plot_train_loss(data, run_dir)
    plot_auc_bars(data, run_dir)
    plot_f1_bars(data, run_dir)

    # New Plots per class
    plot_class_loss(data, run_dir)
    plot_class_accuracy(data, run_dir)



if __name__ == "__main__":
    main()



    
