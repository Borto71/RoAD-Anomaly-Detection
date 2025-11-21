# plts_autogluon.py

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Directory where the autogluon script saves its runs
RUNS_BASE_DIR = "../runs_autogluon"
# Name of the directory containing the AutoGluon models
AG_MODELS_DIR_NAME = "AutogluonModels"
# Name of the learning curve CSV file
LC_CSV_NAME = "learning_curve_test.csv"
# Name of the comparison CSV file
COMP_CSV_NAME = "models_comparison_test.csv"

def find_latest_run_dir(base_dir):
    """
    Finds the latest run directory (based on creation timestamp) inside the base directory.
    """
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        print(f"Error: No run directories found in {base_dir}")
        return None
    
    # Sort directories by name (which includes the timestamp YYYYMMDD_HHMMSS)
    run_dirs.sort(reverse=True)
    return run_dirs[0]

def plot_learning_curves(ag_path, model_name, model_dir):
    """
    Generates and saves the Learning Curve plot (Accuracy and Loss vs. Threshold) 
    for a specific model from its CSV file.
    """
    lc_path = os.path.join(model_dir, LC_CSV_NAME)
    
    if not os.path.exists(lc_path):
        print(f"Warning: Learning curve CSV not found for model {model_name} at {lc_path}")
        return

    try:
        lc_df = pd.read_csv(lc_path)
    except Exception as e:
        print(f"Error reading learning curve CSV for {model_name}: {e}")
        return

    print(f"  -> Plotting learning curve for {model_name}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Accuracy on the primary y-axis (left)
    color_acc = 'tab:blue'
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Accuracy', color=color_acc)
    ax1.plot(lc_df['threshold'], lc_df['accuracy'], color=color_acc, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create a secondary y-axis for Loss (right)
    ax2 = ax1.twinx()  
    color_loss = 'tab:red'
    ax2.set_ylabel('BCE Loss', color=color_loss)  
    # Loss is constant across thresholds, but we plot it for completeness
    ax2.plot(lc_df['threshold'], lc_df['loss'], color=color_loss, linestyle='--', label='BCE Loss')
    ax2.tick_params(axis='y', labelcolor=color_loss)
    
    fig.suptitle(f'Test Set: Accuracy and Loss vs. Threshold for {model_name}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    # Save plot
    plot_path = os.path.join(model_dir, "learning_curve_test.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  -> Saved learning curve plot to {plot_path}")

def plot_model_comparison(comp_df, run_dir):
    """
    Generates and saves comparative bar plots for Class 0 and Class 1 Accuracy (at threshold 0.5).
    """
    model_names = comp_df['model'].tolist()
    
    # --- Accuracy Class 0 Comparison (Normal Samples) ---
    acc_class0 = comp_df['accuracy_class0_at_0.5'].tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, acc_class0, color='skyblue')
    
    ax.set_ylabel('Accuracy (Class 0: Normal)')
    ax.set_xlabel('Model')
    ax.set_title('Comparative Accuracy on Normal Samples (Class 0) at Threshold 0.5')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom')
        
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path_0 = os.path.join(run_dir, "comparison_accuracy_class0.png")
    plt.savefig(plot_path_0)
    plt.close(fig)
    print(f"Saved comparison plot for Class 0 Accuracy to {plot_path_0}")

    # --- Accuracy Class 1 Comparison (Anomaly Samples) ---
    acc_class1 = comp_df['accuracy_class1_at_0.5'].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, acc_class1, color='lightcoral')

    ax.set_ylabel('Accuracy (Class 1: Anomaly)')
    ax.set_xlabel('Model')
    ax.set_title('Comparative Accuracy on Anomaly Samples (Class 1) at Threshold 0.5')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom')
        
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path_1 = os.path.join(run_dir, "comparison_accuracy_class1.png")
    plt.savefig(plot_path_1)
    plt.close(fig)
    print(f"Saved comparison plot for Class 1 Accuracy to {plot_path_1}")


def main():
    """
    Main function to orchestrate the plotting process.
    """
    print("Starting plotting script...")

    # 1. Find the latest run directory
    run_dir = find_latest_run_dir(RUNS_BASE_DIR)
    if run_dir is None:
        return
    print(f"\nProcessing latest run directory: {run_dir}")

    ag_path = os.path.join(run_dir, AG_MODELS_DIR_NAME)
    if not os.path.exists(ag_path):
        print(f"Error: AutoGluon models directory not found at {ag_path}")
        return

    # 2. Plot learning curves for each individual model
    print("\n--- Generating individual model plots (Learning Curves) ---")
    
    # Find all model directories inside AutogluonModels
    model_dirs = [d for d in os.listdir(ag_path) if os.path.isdir(os.path.join(ag_path, d))]
    
    for model_name in model_dirs:
        model_dir = os.path.join(ag_path, model_name)
        plot_learning_curves(ag_path, model_name, model_dir)

    # 3. Load the comparison CSV
    comp_csv_path = os.path.join(run_dir, COMP_CSV_NAME)
    if not os.path.exists(comp_csv_path):
        print(f"Error: Comparison CSV not found at {comp_csv_path}")
        return

    try:
        comp_df = pd.read_csv(comp_csv_path)
    except Exception as e:
        print(f"Error reading comparison CSV: {e}")
        return
    
    # 4. Generate comparative plots
    print("\n--- Generating comparative plots (Class Accuracies) ---")
    plot_model_comparison(comp_df, run_dir)

    print("\nPlotting script finished successfully.")


if __name__ == "__main__":
    main()