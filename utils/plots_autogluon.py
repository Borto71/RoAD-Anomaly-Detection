import os
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

def plot_learning_curves(model_name, model_dir):
    """
    Generates and saves the Learning Curve plot.
    If extended metrics (F1, Acc Class 0, Acc Class 1) are present, plots a 2x2 grid.
    Otherwise, falls back to the standard Accuracy/Loss plot.
    """
    lc_path = os.path.join(model_dir, LC_CSV_NAME)
    
    if not os.path.exists(lc_path):
        return

    try:
        lc_df = pd.read_csv(lc_path)
    except Exception as e:
        print(f"Error reading learning curve CSV for {model_name}: {e}")
        return

    print(f"  -> Plotting metrics for {model_name}")

    # Colonne necessarie per i grafici estesi
    required_cols = ['f1_score', 'accuracy_class0', 'accuracy_class1']
    has_extended_metrics = all(col in lc_df.columns for col in required_cols)

    if has_extended_metrics:
        # --- MODALITÀ ESTESA (4 Grafici) ---
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Metrics vs Threshold - {model_name}', fontsize=16)
        
        # Definizione dei 4 subplot
        metrics_config = [
            (0, 0, 'f1_score', 'F1 Score', 'tab:green'),
            (0, 1, 'loss', 'BCE Loss', 'tab:red'),
            (1, 0, 'accuracy_class0', 'Accuracy Class 0 (Normal)', 'tab:blue'),
            (1, 1, 'accuracy_class1', 'Accuracy Class 1 (Anomaly)', 'tab:orange')
        ]
        
        for r, c, col, title, color in metrics_config:
            ax = axs[r, c]
            if col in lc_df.columns:
                ax.plot(lc_df['threshold'], lc_df[col], color=color, linewidth=2)
                ax.set_title(title)
                ax.set_xlabel('Threshold')
                ax.set_ylabel('Score')
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # Se non è la Loss, fissa asse Y tra 0 e 1 per leggibilità
                if col != 'loss':
                    ax.set_ylim(-0.05, 1.05)
                    # Segna il punto massimo
                    max_idx = lc_df[col].idxmax()
                    max_val = lc_df[col].max()
                    max_thr = lc_df['threshold'].iloc[max_idx]
                    ax.scatter(max_thr, max_val, color='black', zorder=5, s=30)
                    ax.text(max_thr, max_val, f' Max: {max_val:.3f}', fontsize=9, verticalalignment='bottom')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Salva come file distinto per metriche dettagliate
        plot_path = os.path.join(model_dir, "detailed_metrics_vs_threshold.png")
        plt.savefig(plot_path)
        plt.close(fig)
        
    else:
        # --- MODALITÀ STANDARD (Solo Accuracy e Loss) ---
        # (Questo è il tuo codice originale mantenuto per retro-compatibilità)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color_acc = 'tab:blue'
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Accuracy', color=color_acc)
        ax1.plot(lc_df['threshold'], lc_df['accuracy'], color=color_acc, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = ax1.twinx()  
        color_loss = 'tab:red'
        ax2.set_ylabel('BCE Loss', color=color_loss)  
        
        if 'loss' in lc_df.columns:
            ax2.plot(lc_df['threshold'], lc_df['loss'], color=color_loss, linestyle='--', label='BCE Loss')
        ax2.tick_params(axis='y', labelcolor=color_loss)
        
        fig.suptitle(f'Test Set: Accuracy and Loss vs. Threshold for {model_name}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

        plot_path = os.path.join(model_dir, "learning_curve_test.png")
        plt.savefig(plot_path)
        plt.close(fig)


def plot_bar_metric(model_names, values, metric_name, color, run_dir, filename):
    """
    Helper function to plot a simple bar chart for a given metric.
    """
    if not values:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, values, color=color)
    
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Model')
    ax.set_title(f'Comparative {metric_name}')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)
        
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(run_dir, filename)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved comparison plot for {metric_name} to {filename}")

def plot_model_comparison(comp_df, run_dir):
    """
    Generates and saves comparative bar plots based on available columns in the CSV.
    """
    model_names = comp_df['model'].tolist()
    
    # Check which columns are available and plot accordingly
    
    # 1. ROC AUC (Most important for Anomaly Detection)
    if 'roc_auc' in comp_df.columns:
        plot_bar_metric(
            model_names, 
            comp_df['roc_auc'].tolist(), 
            "ROC AUC", 
            "mediumpurple", 
            run_dir, 
            "comparison_roc_auc.png"
        )
    
    # 2. Best F1 Score
    if 'best_f1' in comp_df.columns:
        plot_bar_metric(
            model_names, 
            comp_df['best_f1'].tolist(), 
            "Best F1 Score", 
            "teal", 
            run_dir, 
            "comparison_best_f1.png"
        )

    # 3. Overall Accuracy at 0.5
    if 'accuracy_at_0.5' in comp_df.columns:
        plot_bar_metric(
            model_names, 
            comp_df['accuracy_at_0.5'].tolist(), 
            "Accuracy (Threshold 0.5)", 
            "cornflowerblue", 
            run_dir, 
            "comparison_accuracy_overall.png"
        )

    # 4. Class 0 Accuracy (Legacy support)
    if 'accuracy_class0_at_0.5' in comp_df.columns:
        plot_bar_metric(
            model_names, 
            comp_df['accuracy_class0_at_0.5'].tolist(), 
            "Accuracy Class 0 (Normal)", 
            "skyblue", 
            run_dir, 
            "comparison_accuracy_class0.png"
        )

    # 5. Class 1 Accuracy (Legacy support)
    if 'accuracy_class1_at_0.5' in comp_df.columns:
        plot_bar_metric(
            model_names, 
            comp_df['accuracy_class1_at_0.5'].tolist(), 
            "Accuracy Class 1 (Anomaly)", 
            "lightcoral", 
            run_dir, 
            "comparison_accuracy_class1.png"
        )

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
    
    if os.path.exists(ag_path):
        model_dirs = [d for d in os.listdir(ag_path) if os.path.isdir(os.path.join(ag_path, d))]
        for model_name in model_dirs:
            model_dir = os.path.join(ag_path, model_name)
            # Only try to plot if the CSV exists, to avoid warnings for non-model folders
            if os.path.exists(os.path.join(model_dir, LC_CSV_NAME)):
                plot_learning_curves(model_name, model_dir)

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
    print("\n--- Generating comparative plots ---")
    plot_model_comparison(comp_df, run_dir)

    print("\nPlotting script finished successfully.")


if __name__ == "__main__":
    main()