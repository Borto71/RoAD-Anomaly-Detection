import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

# Directory settings
RUNS_BASE_DIR = "../runs_cnn"

def find_latest_run_dir(base_dir):
    """
    Finds the most recent experiment folder based on the timestamp.
    """
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs: return None
    run_dirs.sort(reverse=True)
    return run_dirs[0]

def plot_metrics_vs_threshold(scores, labels, scenario_name, output_dir):
    """
    Plots F1 Score, Accuracy Class 0 (Normal), and Accuracy Class 1 (Anomaly)
    as a function of the anomaly score threshold.
    """
    print(f"  Generating Threshold Curves for {scenario_name}...")
    
    # Generate a range of thresholds between min and max scores
    t_min, t_max = scores.min(), scores.max()
    thresholds = np.linspace(t_min, t_max, 200) # 200 steps for a smooth curve
    
    f1_list = []
    acc0_list = []
    acc1_list = []
    
    for thr in thresholds:
        # Predicted: 1 if score >= threshold, else 0
        y_pred = (scores >= thr).astype(int)
        
        # Calculate Confusion Matrix elements manually
        tp = ((labels == 1) & (y_pred == 1)).sum()
        tn = ((labels == 0) & (y_pred == 0)).sum()
        fp = ((labels == 0) & (y_pred == 1)).sum()
        fn = ((labels == 1) & (y_pred == 0)).sum()
        
        # Specificity (Accuracy on Normals)
        acc0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
        # Recall / Sensitivity (Accuracy on Anomalies)
        acc1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        
        # F1 Score calculation
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = acc1
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        f1_list.append(f1)
        acc0_list.append(acc0)
        acc1_list.append(acc1)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, f1_list, label='F1 Score', color='green', linewidth=2)
    ax.plot(thresholds, acc0_list, label='Accuracy Class 0 (Normal)', color='blue', linestyle='--')
    ax.plot(thresholds, acc1_list, label='Accuracy Class 1 (Anomaly)', color='orange', linestyle='--')
    
    # Highlight the Best F1 Score
    best_idx = np.argmax(f1_list)
    best_thr = thresholds[best_idx]
    best_f1 = f1_list[best_idx]
    
    ax.scatter(best_thr, best_f1, color='black', zorder=5)
    ax.text(best_thr, best_f1 + 0.02, f"Best F1: {best_f1:.2f}\nThr: {best_thr:.2f}", ha='center')
    
    ax.set_title(f'Metrics vs Threshold - {scenario_name.capitalize()}')
    ax.set_xlabel('Anomaly Score Threshold (Distance^2)')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"curves_threshold_{scenario_name}.png"))
    plt.close()

def plot_roc_curve(scores, labels, scenario_name, output_dir):
    """
    Plots the standard Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Acc Class 0)')
    ax.set_ylabel('True Positive Rate (Acc Class 1)')
    ax.set_title(f'ROC Curve - {scenario_name.capitalize()}')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_curve_{scenario_name}.png"))
    plt.close()

def plot_feature_importance(run_dir):
    """
    Reads the CSV generated during training and plots the top 20 features.
    """
    csv_path = os.path.join(run_dir, "feature_importance.csv")
    if not os.path.exists(csv_path):
        print("Feature importance CSV not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Select top 20 and sort ascending for the horizontal bar chart
    top_df = df.head(20).sort_values(by="importance", ascending=True) 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df['feature'], top_df['importance'], color='teal')
    
    ax.set_xlabel('Permutation Importance (Increase in Loss)')
    ax.set_title('Top 20 Most Important Features (Deep SVDD)')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "feature_importance.png"))
    plt.close()
    print("  Saved feature_importance.png")

def main():
    # Find the latest run automatically
    run_dir = find_latest_run_dir(RUNS_BASE_DIR)
    if not run_dir:
        print("No run directory found. Run main.py first.")
        return
        
    print(f"Processing plots for: {run_dir}")
    
    # 1. Feature Importance
    plot_feature_importance(run_dir)
    
    # 2. Process Results for each scenario
    result_files = glob.glob(os.path.join(run_dir, "results_*.npz"))
    
    for r_file in result_files:
        scenario = os.path.basename(r_file).replace("results_", "").replace(".npz", "")
        
        data = np.load(r_file)
        scores = data["scores"]
        labels = data["labels"]
        
        # Only plot if we have scores and at least two classes (0 and 1)
        if len(scores) > 0 and len(np.unique(labels)) > 1:
            plot_metrics_vs_threshold(scores, labels, scenario, run_dir)
            plot_roc_curve(scores, labels, scenario, run_dir)
        else:
            print(f"Skipping {scenario}: not enough data or only one class present.")
            
    print("\nAll plots generated successfully.")

if __name__ == "__main__":
    main()