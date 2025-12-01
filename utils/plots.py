import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Directory settings
RUNS_BASE_DIR = "../runs_combined"
MODELS_PARENT_DIR = "Models"
LC_CSV_NAME = "learning_curve_test.csv"
COMP_CSV_NAME = "models_comparison_test.csv"
METRICS_JSON_NAME = "metrics_train_test.json"
FI_CSV_NAME = "feature_importance.csv"

def find_latest_run_dir(base_dir):
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        print(f"Error: No run directories found in {base_dir}")
        return None
    run_dirs.sort(reverse=True)
    return run_dirs[0]

def plot_confusion_matrix(model_name, model_dir):
    json_path = os.path.join(model_dir, METRICS_JSON_NAME)
    if not os.path.exists(json_path): return

    try:
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        best_thr = metrics.get("test", {}).get("test_threshold", 0.5)
        
        acc0 = None; acc1 = None
        lc_path = os.path.join(model_dir, LC_CSV_NAME)
        used_thr = best_thr
        
        if os.path.exists(lc_path):
            lc_df = pd.read_csv(lc_path)
            row = lc_df.iloc[(lc_df['threshold'] - best_thr).abs().argsort()[:1]]
            if not row.empty:
                acc0 = row['accuracy_class0'].values[0]
                acc1 = row['accuracy_class1'].values[0]
                used_thr = row['threshold'].values[0]

        if acc0 is not None and acc1 is not None:
            cm_norm = np.array([[acc0, 1 - acc0], [1 - acc1, acc1]])
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
                        xticklabels=["Pred Normal (0)", "Pred Anomaly (1)"],
                        yticklabels=["True Normal (0)", "True Anomaly (1)"],
                        ax=ax, annot_kws={"size": 14})
            ax.set_title(f'Normalized CM (Best Thr={used_thr:.3f})\n{model_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "confusion_matrix_norm.png"))
            plt.close(fig)
            print(f"     Saved confusion matrix.")
            
    except Exception as e:
        print(f"Error plotting CM for {model_name}: {e}")

def plot_learning_curves(model_name, model_dir):
    lc_path = os.path.join(model_dir, LC_CSV_NAME)
    fi_path = os.path.join(model_dir, FI_CSV_NAME)
    
    if not os.path.exists(lc_path): return

    try:
        lc_df = pd.read_csv(lc_path)
    except Exception as e:
        print(f"Error reading learning curve CSV: {e}"); return

    print(f"  -> Plotting metrics for {model_name}")

    required_cols = ['f1_score', 'accuracy_class0', 'accuracy_class1']
    has_extended_metrics = all(col in lc_df.columns for col in required_cols)

    if has_extended_metrics:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Metrics & Permutation Importance - {model_name}', fontsize=16)
        
        line_plots_config = [
            (0, 0, 'f1_score', 'F1 Score', 'tab:green'),
            (1, 0, 'accuracy_class0', 'Accuracy Class 0 (Normal)', 'tab:blue'),
            (1, 1, 'accuracy_class1', 'Accuracy Class 1 (Anomaly)', 'tab:orange')
        ]
        
        for r, c, col, title, color in line_plots_config:
            ax = axs[r, c]
            if col in lc_df.columns:
                ax.plot(lc_df['threshold'], lc_df[col], color=color, linewidth=2)
                ax.set_title(title)
                ax.set_xlabel('Threshold')
                ax.set_ylabel('Score')
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_ylim(-0.05, 1.05)
                
                max_idx = lc_df[col].idxmax()
                max_val = lc_df[col].max()
                max_thr = lc_df['threshold'].iloc[max_idx]
                ax.scatter(max_thr, max_val, color='black', zorder=5, s=30)
                ax.text(max_thr, max_val, f' Max: {max_val:.3f}', fontsize=9, verticalalignment='bottom')

        # Feature Importance Plot (Permutation)
        ax_fi = axs[0, 1]
        
        if os.path.exists(fi_path):
            try:
                fi_df = pd.read_csv(fi_path)
                # Ensure we have the right columns
                if 'feature' in fi_df.columns and 'importance' in fi_df.columns:
                    fi_top = fi_df.sort_values(by='importance', ascending=False).head(10)
                    fi_top = fi_top.iloc[::-1] # Invert for plotting
                    
                    y_pos = np.arange(len(fi_top))
                    
                    ax_fi.barh(y_pos, fi_top['importance'], color='tab:purple', alpha=0.8)
                    ax_fi.set_yticks(y_pos)
                    ax_fi.set_yticklabels(fi_top['feature'], fontsize=10)
                    
                    ax_fi.set_title('Top 10 Feature Importance (Permutation)')
                    ax_fi.set_xlabel('AUC Drop when permuted')
                    ax_fi.grid(axis='x', linestyle='--', alpha=0.5)
                else:
                    ax_fi.text(0.5, 0.5, 'CSV format invalid', ha='center')
            except Exception as e:
                print(f"Error plotting feature importance: {e}")
                ax_fi.text(0.5, 0.5, 'Error loading FI data', ha='center')
        else:
            ax_fi.text(0.5, 0.5, 'Feature Importance CSV missing', ha='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(model_dir, "detailed_metrics_vs_threshold.png")
        plt.savefig(plot_path)
        plt.close(fig)

def plot_model_comparison(comp_df, run_dir):
    model_names = comp_df['model'].tolist()
    metrics = [
        ('roc_auc', "ROC AUC", "mediumpurple"),
        ('best_f1', "Best F1 Score", "teal"),
        ('accuracy_at_0.5', "Accuracy (Thr 0.5)", "cornflowerblue")
    ]
    
    for col, title, color in metrics:
        if col in comp_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(model_names, comp_df[col], color=color, width=0.4)
            ax.set_title(f'Comparative {title}')
            ax.set_ylabel(title)
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.1)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"comparison_{col}.png"))
            plt.close(fig)

def main():
    print("Starting plotting script for CNN...")
    run_dir = find_latest_run_dir(RUNS_BASE_DIR)
    if run_dir is None: return
    print(f"\nProcessing latest run directory: {run_dir}")

    models_path = os.path.join(run_dir, MODELS_PARENT_DIR)
    if not os.path.exists(models_path): return

    if os.path.exists(models_path):
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
            plot_model_comparison(comp_df, run_dir)
        except Exception as e:
            print(f"Error reading comparison CSV: {e}")

    print("\nPlotting script finished successfully.")

if __name__ == "__main__":
    main()