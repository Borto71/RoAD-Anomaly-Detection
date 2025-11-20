import os
import json
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_run(runs_base_dir="../runs_autogluon"):
    """Trova l'ultima run in base all'ordinamento alfabetico dei nomi directory."""
    if not os.path.isdir(runs_base_dir):
        raise FileNotFoundError(f"Base directory '{runs_base_dir}' not found.")

    run_dirs = [
        os.path.join(runs_base_dir, d)
        for d in os.listdir(runs_base_dir)
        if os.path.isdir(os.path.join(runs_base_dir, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in '{runs_base_dir}'.")

    run_dirs_sorted = sorted(run_dirs)
    return run_dirs_sorted[-1]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_get(dct, *keys, default=None):
    tmp = dct
    for k in keys:
        if not isinstance(tmp, dict) or k not in tmp:
            return default
        tmp = tmp[k]
    return tmp


def plot_bar(x_labels, values, title, ylabel, save_path, rotation=45):
    """Utility per fare un barplot 1D e salvarlo."""
    plt.figure()
    idx = np.arange(len(x_labels))
    plt.bar(idx, values)
    plt.xticks(idx, x_labels, rotation=rotation, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_grouped_bar(
    groups,
    series_dict,
    title,
    ylabel,
    save_path,
    rotation=0,
):
    """
    groups: lista di categorie (es. ["class0", "class1"])
    series_dict: dict {nome_serie: [valori per ciascun gruppo]}
                 es. {"train": [v_c0_train, v_c1_train], "test": [v_c0_test, v_c1_test]}
    """
    plt.figure()

    n_groups = len(groups)
    n_series = len(series_dict)
    index = np.arange(n_groups)
    bar_width = 0.8 / max(n_series, 1)

    for i, (series_name, values) in enumerate(series_dict.items()):
        offset = (i - (n_series - 1) / 2.0) * bar_width
        plt.bar(index + offset, values, width=bar_width, label=series_name)

    plt.xticks(index, groups, rotation=rotation)
    plt.title(title)
    plt.ylabel(ylabel)
    if n_series > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_learning_curve(df_lc, metric_col, title, ylabel, save_path):
    """
    df_lc: DataFrame con almeno colonne ["threshold", metric_col]
    metric_col: "accuracy" o "loss"
    """
    plt.figure()
    plt.plot(df_lc["threshold"].values, df_lc[metric_col].values)
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_leaderboard_overview(run_dir, autogluon_models_dir, leaderboard_path):
    """
    Plot globali che confrontano i modelli:
      - AUC-ROC test
      - BCE loss class1 test
      - F1 test
    usando metrics_train_test.json di ciascun modello.
    """
    leaderboard = pd.read_csv(leaderboard_path)
    model_names = leaderboard["model"].tolist()

    # raccogliamo metriche test per ogni modello
    auc_list = []
    f1_list = []
    loss_c0_list = []
    loss_c1_list = []
    models_available = []

    for model_name in model_names:
        model_dir = os.path.join(autogluon_models_dir, model_name)
        metrics_path = os.path.join(model_dir, "metrics_train_test.json")

        if not os.path.isfile(metrics_path):
            # se manca il file, saltiamo
            print(f"[WARNING] metrics_train_test.json not found for model '{model_name}', skipping.")
            continue

        metrics = load_json(metrics_path)
        test_metrics = metrics.get("test", {})

        auc_roc = safe_get(test_metrics, "auc_roc", default=None)
        f1 = safe_get(test_metrics, "best_f1", default=None)
        loss_c0 = safe_get(test_metrics, "bce_loss_class0", default=None)
        loss_c1 = safe_get(test_metrics, "bce_loss_class1", default=None)

        # se auc_roc manca, saltiamo interamente questo modello
        if auc_roc is None:
            print(f"[WARNING] 'auc_roc' missing in test metrics for model '{model_name}', skipping.")
            continue

        models_available.append(model_name)
        auc_list.append(auc_roc)
        f1_list.append(f1 if f1 is not None else 0.0)
        loss_c0_list.append(loss_c0 if loss_c0 is not None else 0.0)
        loss_c1_list.append(loss_c1 if loss_c1 is not None else 0.0)

    if not models_available:
        print("[WARNING] No models with test metrics found. Skipping leaderboard plots.")
        return

    # AUC-ROC test per modello
    plot_bar(
        x_labels=models_available,
        values=auc_list,
        title="AUC-ROC (test) per modello",
        ylabel="AUC-ROC (test)",
        save_path=os.path.join(run_dir, "leaderboard_auc_roc_test.png"),
        rotation=45,
    )

    # F1 (best) test per modello
    plot_bar(
        x_labels=models_available,
        values=f1_list,
        title="Best F1 (test) per modello",
        ylabel="F1 (test)",
        save_path=os.path.join(run_dir, "leaderboard_best_f1_test.png"),
        rotation=45,
    )

    # BCE loss per classe (test) per modello -> due grafici separati
    plot_bar(
        x_labels=models_available,
        values=loss_c0_list,
        title="BCE loss (classe 0, test) per modello",
        ylabel="BCE loss class 0 (test)",
        save_path=os.path.join(run_dir, "leaderboard_bce_loss_class0_test.png"),
        rotation=45,
    )

    plot_bar(
        x_labels=models_available,
        values=loss_c1_list,
        title="BCE loss (classe 1, test) per modello",
        ylabel="BCE loss class 1 (test)",
        save_path=os.path.join(run_dir, "leaderboard_bce_loss_class1_test.png"),
        rotation=45,
    )

    print("Saved leaderboard-level plots into:", run_dir)


def plot_per_model_metrics(autogluon_models_dir, leaderboard_path):
    """
    Per ogni modello in leaderboard:
      - legge metrics_train_test.json e learning_curve_test.csv
      - salva:
        - loss_per_class_train_test.png
        - auc_roc_train_test.png
        - f1_precision_recall_accuracy_test.png
        - learning_curve_accuracy.png
        - learning_curve_loss.png
    """
    leaderboard = pd.read_csv(leaderboard_path)
    model_names = leaderboard["model"].tolist()

    for model_name in model_names:
        print(f"\n[Model plots] {model_name}")
        model_dir = os.path.join(autogluon_models_dir, model_name)
        if not os.path.isdir(model_dir):
            print(f"  [WARNING] Model dir '{model_dir}' not found, skipping.")
            continue

        metrics_path = os.path.join(model_dir, "metrics_train_test.json")
        if not os.path.isfile(metrics_path):
            print(f"  [WARNING] metrics_train_test.json not found, skipping model.")
            continue

        # --- Metriche train/test ---
        metrics = load_json(metrics_path)
        train_metrics = metrics.get("train", {})
        test_metrics = metrics.get("test", {})

        # BCE loss per classe
        loss_c0_train = safe_get(train_metrics, "bce_loss_class0", default=None)
        loss_c1_train = safe_get(train_metrics, "bce_loss_class1", default=None)
        loss_c0_test = safe_get(test_metrics, "bce_loss_class0", default=None)
        loss_c1_test = safe_get(test_metrics, "bce_loss_class1", default=None)

        # sostituiamo None con 0.0 per poter plottare
        loss_c0_train = 0.0 if loss_c0_train is None else loss_c0_train
        loss_c1_train = 0.0 if loss_c1_train is None else loss_c1_train
        loss_c0_test = 0.0 if loss_c0_test is None else loss_c0_test
        loss_c1_test = 0.0 if loss_c1_test is None else loss_c1_test

        # grafico BCE loss per classe (train vs test)
        loss_groups = ["class0", "class1"]
        series = {
            "train": [loss_c0_train, loss_c1_train],
            "test": [loss_c0_test, loss_c1_test],
        }
        plot_grouped_bar(
            groups=loss_groups,
            series_dict=series,
            title=f"BCE loss per classe - {model_name}",
            ylabel="BCE loss",
            save_path=os.path.join(model_dir, "loss_per_class_train_test.png"),
            rotation=0,
        )

        # AUC-ROC train vs test
        auc_train = safe_get(train_metrics, "auc_roc", default=None)
        auc_test = safe_get(test_metrics, "auc_roc", default=None)
        auc_train = 0.0 if auc_train is None else auc_train
        auc_test = 0.0 if auc_test is None else auc_test

        plot_grouped_bar(
            groups=["AUC-ROC"],
            series_dict={"train": [auc_train], "test": [auc_test]},
            title=f"AUC-ROC train vs test - {model_name}",
            ylabel="AUC-ROC",
            save_path=os.path.join(model_dir, "auc_roc_train_test.png"),
            rotation=0,
        )

        # F1 / precision / recall / accuracy (solo test, più interessante)
        f1_test = safe_get(test_metrics, "best_f1", default=None)
        prec_test = safe_get(test_metrics, "best_precision", default=None)
        rec_test = safe_get(test_metrics, "best_recall", default=None)
        acc_05_test = safe_get(test_metrics, "accuracy_at_0.5", default=None)

        f1_test = 0.0 if f1_test is None else f1_test
        prec_test = 0.0 if prec_test is None else prec_test
        rec_test = 0.0 if rec_test is None else rec_test
        acc_05_test = 0.0 if acc_05_test is None else acc_05_test

        metrics_names = ["F1 (best)", "Precision (best)", "Recall (best)", "Accuracy (thr=0.5)"]
        metrics_values = [f1_test, prec_test, rec_test, acc_05_test]

        plot_bar(
            x_labels=metrics_names,
            values=metrics_values,
            title=f"Metriche test (best thr / thr=0.5) - {model_name}",
            ylabel="Score",
            save_path=os.path.join(model_dir, "f1_precision_recall_accuracy_test.png"),
            rotation=30,
        )

        # --- Learning curve (accuracy e loss vs threshold, test) ---
        lc_path = os.path.join(model_dir, "learning_curve_test.csv")
        if os.path.isfile(lc_path):
            df_lc = pd.read_csv(lc_path)

            if "threshold" in df_lc.columns and "accuracy" in df_lc.columns:
                plot_learning_curve(
                    df_lc,
                    metric_col="accuracy",
                    title=f"Learning curve (accuracy vs threshold) - {model_name}",
                    ylabel="Accuracy",
                    save_path=os.path.join(model_dir, "learning_curve_accuracy.png"),
                )

            if "threshold" in df_lc.columns and "loss" in df_lc.columns:
                plot_learning_curve(
                    df_lc,
                    metric_col="loss",
                    title=f"Learning curve (loss vs threshold) - {model_name}",
                    ylabel="Loss",
                    save_path=os.path.join(model_dir, "learning_curve_loss.png"),
                )
        else:
            print(f"  [INFO] learning_curve_test.csv not found for {model_name}, skipping learning curve plots.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot utility for AutoGluon runs (metrics and leaderboard)."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run directory (e.g. runs_autogluon/run_20251120_123456). "
             "If not provided, the latest run in 'runs_autogluon' will be used.",
    )

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_run("../runs_autogluon")
        print(f"No --run-dir specified. Using latest run: {run_dir}")
    else:
        run_dir = args.run_dir
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory '{run_dir}' not found.")

    leaderboard_path = os.path.join(run_dir, "leaderboard.csv")
    overall_metrics_path = os.path.join(run_dir, "metrics_overall.json")
    autogluon_models_dir = os.path.join(run_dir, "AutogluonModels")

    if not os.path.isfile(leaderboard_path):
        raise FileNotFoundError(f"leaderboard.csv not found in '{run_dir}'.")

    if not os.path.isdir(autogluon_models_dir):
        raise FileNotFoundError(f"AutogluonModels directory not found in '{run_dir}'.")

    print(f"Using run directory: {run_dir}")

    # (opzionale) leggiamo metrics_overall giusto per info
    if os.path.isfile(overall_metrics_path):
        overall = load_json(overall_metrics_path)
        print("\n[Overall metrics summary]")
        metrics_best = overall.get("metrics_test_best_model", {})
        for k, v in metrics_best.items():
            print(f"  {k}: {v}")
    else:
        print("[INFO] metrics_overall.json not found, proceeding without overall summary.")

    # Plot per ogni modello
    plot_per_model_metrics(autogluon_models_dir, leaderboard_path)

    # Plot globali basati su metrics_train_test.json dei modelli
    plot_leaderboard_overview(run_dir, autogluon_models_dir, leaderboard_path)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
