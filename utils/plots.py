import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RUNS_BASE_DIR = "../runs_pytorch"
MODELS_DIR_NAME = "Models"
LC_CSV_NAME = "learning_curve_test.csv"
COMP_CSV_NAME = "models_comparison_test.csv"
METRICS_JSON_NAME = "metrics_train_test.json"

def find_latest_run_dir(base_dir):
    """Finds the most recently created run directory."""
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist.")
        return None
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        print(f"Error: No run directories found in {base_dir}")
        return None
    run_dirs.sort(reverse=True)
    return run_dirs[0]

def plot_confusion_matrix(model_name, model_dir):
    """Plots a normalized Confusion Matrix based on the Best Threshold."""
    json_path = os.path.join(model_dir, METRICS_JSON_NAME)
    if not os.path.exists(json_path):
        return

    try:
        with open(json_path, 'r') as f:
            metrics_root = json.load(f)
            # Support both structures: directly metrics or nested under "test"
            metrics = metrics_root.get("test", metrics_root)
        
        best_thr = metrics.get("best_threshold", 0.5)
        
        acc0 = None 
        acc1 = None 
        used_thr = best_thr
        
        # Reading the Learning Curve CSV to find accuracies at best threshold
        lc_path = os.path.join(model_dir, LC_CSV_NAME)
        if os.path.exists(lc_path):
            lc_df = pd.read_csv(lc_path)
            
            # Find row closest to the best threshold
            row = lc_df.iloc[(lc_df['threshold'] - best_thr).abs().argsort()[:1]]
            
            if not row.empty:
                acc0 = row['accuracy_class0'].values[0]
                acc1 = row['accuracy_class1'].values[0]
                used_thr = row['threshold'].values[0]

        if acc0 is not None and acc1 is not None:
            # Construct normalized CM: [[TN rate, FP rate], [FN rate, TP rate]]
            # Note: acc0 = TN / (TN+FP), acc1 = TP / (TP+FN)
            cm_norm = np.array([[acc0, 1 - acc0], [1 - acc1, acc1]])
            
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
                        xticklabels=["Pred Normal (0)", "Pred Anomaly (1)"],
                        yticklabels=["True Normal (0)", "True Anomaly (1)"],
                        ax=ax, annot_kws={"size": 14})
            
            ax.set_title(f'Normalized CM (Best Thr={used_thr:.3f})\n{model_name}', fontsize=14)
            plt.tight_layout()
            
            cm_path = os.path.join(model_dir, "confusion_matrix_norm.png")
            plt.savefig(cm_path)
            plt.close(fig)
            print(f"     Saved confusion matrix to {cm_path}")
            
    except Exception as e:
        print(f"Error plotting CM for {model_name}: {e}")

def plot_learning_curves(model_name, model_dir):
    """Plots F1, Accuracy Class 0, and Accuracy Class 1 vs Threshold."""
    lc_path = os.path.join(model_dir, LC_CSV_NAME)
    
    if not os.path.exists(lc_path):
        return

    try:
        lc_df = pd.read_csv(lc_path)
    except Exception as e:
        print(f"Error reading learning curve CSV for {model_name}: {e}")
        return

    print(f"  -> Plotting metrics for {model_name}")

    required_cols = ['f1_score', 'accuracy_class0', 'accuracy_class1']
    if all(col in lc_df.columns for col in required_cols):
        # We drop the Feature Importance subplot since custom PyTorch models 
        # don't output it by default in the generic training loop.
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Metrics vs Threshold - {model_name}', fontsize=16)
        
        plot_configs = [
            (0, 'f1_score', 'F1 Score', 'tab:green'),
            (1, 'accuracy_class0', 'Acc Normal (Class 0)', 'tab:blue'),
            (2, 'accuracy_class1', 'Acc Anomaly (Class 1)', 'tab:orange')
        ]
        
        for idx, col, title, color in plot_configs:
            ax = axs[idx]
            ax.plot(lc_df['threshold'], lc_df[col], color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
            
            # Highlight max
            max_idx = lc_df[col].idxmax()
            max_val = lc_df[col].max()
            max_thr = lc_df['threshold'].iloc[max_idx]
            ax.scatter(max_thr, max_val, color='black', zorder=5, s=30)
            ax.text(max_thr, max_val, f' {max_val:.3f}', fontsize=9, va='bottom')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(model_dir, "detailed_metrics_vs_threshold.png")
        plt.savefig(plot_path)
        plt.close(fig)
    else:
        print(f"     Skipping LC plot: missing columns in {LC_CSV_NAME}")

def plot_model_comparison(comp_df, run_dir):
    """Plots bar charts comparing all trained models."""
    model_names = comp_df['model'].tolist()
    metrics = [
        ('roc_auc', "ROC AUC", "mediumpurple"),
        ('best_f1', "Best F1 Score", "teal"),
        ('accuracy_at_0.5', "Accuracy (Thr 0.5)", "cornflowerblue")
    ]
    
    for col, title, color in metrics:
        if col in comp_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(model_names, comp_df[col], color=color)
            ax.set_title(f'Comparative {title}')
            ax.set_ylabel(title)
            # Add value labels
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, 
                        f"{yval:.4f}", ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.savefig(os.path.join(run_dir, f"comparison_{col}.png"))
            plt.close(fig)
            print(f"Saved comparison plot for {title}")

def main():
    print("Starting plotting script (PyTorch version)...")
    run_dir = find_latest_run_dir(RUNS_BASE_DIR)
    if run_dir is None: return
    print(f"\nProcessing latest run directory: {run_dir}")

    models_path = os.path.join(run_dir, MODELS_DIR_NAME)
    if not os.path.exists(models_path): 
        print(f"Models directory not found: {models_path}")
        return

    print("\n--- Generating individual model plots ---")
    model_dirs = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
    for model_name in model_dirs:
        model_dir = os.path.join(models_path, model_name)
        if os.path.exists(os.path.join(model_dir, LC_CSV_NAME)):
            plot_learning_curves(model_name, model_dir)
            plot_confusion_matrix(model_name, model_dir)

    comp_csv_path = os.path.join(run_dir, COMP_CSV_NAME)
    if os.path.exists(comp_csv_path):
        try:
            comp_df = pd.read_csv(comp_csv_path)
            print("\n--- Generating comparative plots ---")
            plot_model_comparison(comp_df, run_dir)
        except Exception as e:
            print(f"Error reading comparison CSV: {e}")

    print("\nPlotting script finished successfully.")

if __name__ == "__main__":
    main()