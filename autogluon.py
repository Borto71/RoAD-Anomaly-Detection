import os
import time
import json

import numpy as np
import pandas as pd

from RoADDataset import Dataset as RoADDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from autogluon.tabular import TabularPredictor


def compute_minmax(train_records, num_features = 86):
    arrays = [np.asarray(rec)[:, :num_features] for rec in train_records]
    big = np.concatenate(arrays, axis = 0) 

    feat_min = big.min(axis = 0)
    feat_max = big.max(axis = 0)

    return feat_min.astype(np.float32), feat_max.astype(np.float32)


def minmax_scale(arr, min_vals, max_vals, eps = 1e-8):
    arr = arr.astype(np.float32)
    num = arr - min_vals
    den = (max_vals - min_vals) + eps
    return 2.0 * (num / den) - 1.0


def normalize_records(records, feat_min, feat_max, num_features = 86):
    """
    Normalize only the feature columns (first num_features columns) using
    min-max scaling to [-1, 1].
    """
    norm_recs = []
    for rec in records:
        arr = np.asarray(rec)
        feats = arr[:, :num_features]
        feats_scaled = minmax_scale(feats, feat_min, feat_max)

        if arr.shape[1] > num_features:
            label = arr[:, num_features:num_features + 1]
            arr_scaled = np.concatenate([feats_scaled, label], axis=1)
        else:
            arr_scaled = feats_scaled

        norm_recs.append(arr_scaled)
    return norm_recs


def compute_metrics_from_scores(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    mean_loss_normal = float(scores[labels == 0].mean()) if np.any(labels == 0) else None
    mean_loss_anom = float(scores[labels == 1].mean()) if np.any(labels == 1) else None

    unique_scores = np.unique(scores)
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = None
    best_conf = None

    for thr in unique_scores:
        y_pred = (scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, y_pred, average = "binary", zero_division = 0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_thr = thr

            tp = int(((labels == 1) & (y_pred == 1)).sum())
            tn = int(((labels == 0) & (y_pred == 0)).sum())
            fp = int(((labels == 0) & (y_pred == 1)).sum())
            fn = int(((labels == 1) & (y_pred == 0)).sum())
            best_conf = (tp, fp, tn, fn)

    if best_conf is not None:
        tp, fp, tn, fn = best_conf
        total = tp + fp + tn + fn
        acc_overall = (tp + tn) / total if total > 0 else None
        den_normal = tn + fp
        acc_normal = tn / den_normal if den_normal > 0 else None
        den_anom = tp + fn
        acc_anom = tp / den_anom if den_anom > 0 else None
    else:
        acc_overall = acc_normal = acc_anom = None

    metrics = { # dictionary of computed metrics
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mean_loss_normal": mean_loss_normal,
        "mean_loss_anom": mean_loss_anom,
        "best_threshold": float(best_thr) if best_thr is not None else None,
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "acc_overall": acc_overall,
        "acc_normal": acc_normal,
        "acc_anom": acc_anom,
    }
    return metrics


def compute_bce_losses_and_metrics(y_true, scores, thresholds_for_curve=None):
    """
    Compute BCE losses and various metrics from true labels and predicted scores.
    Optionally compute learning curve data for given thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    # clip for numerical stability
    eps = 1e-7
    p = np.clip(scores, eps, 1 - eps)

    # BCE loss
    bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    loss_overall = float(bce.mean())
    loss_class0 = float(bce[y_true == 0].mean()) if np.any(y_true == 0) else None
    loss_class1 = float(bce[y_true == 1].mean()) if np.any(y_true == 1) else None

    # AUC amd PR-AUC
    auc_roc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    # F1 and threshold optimization
    unique_scores = np.unique(scores)
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = None

    for thr in unique_scores:
        y_pred_thr = (scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_thr, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_thr = thr

    # metrics at threshold 0.5
    y_pred_05 = (scores >= 0.5).astype(int)
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(
        y_true, y_pred_05, average="binary", zero_division=0
    )
    acc_05 = float((y_pred_05 == y_true).mean())

    # confusion matrix components at 0.5
    tp = int(((y_true == 1) & (y_pred_05 == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_05 == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_05 == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_05 == 0)).sum())

    den_class0 = tn + fp
    den_class1 = tp + fn

    acc_class0_05 = float(tn / den_class0) if den_class0 > 0 else None
    acc_class1_05 = float(tp / den_class1) if den_class1 > 0 else None

    metrics = {
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),

        "bce_loss_overall": loss_overall,
        "bce_loss_class0": loss_class0,
        "bce_loss_class1": loss_class1,

        "best_f1": float(best_f1),
        "best_precision": float(best_prec),
        "best_recall": float(best_rec),
        "best_threshold": float(best_thr) if best_thr is not None else None,

        "f1_at_0.5": float(f1_05),
        "precision_at_0.5": float(prec_05),
        "recall_at_0.5": float(rec_05),
        "accuracy_at_0.5": acc_05,

        "accuracy_class0_at_0.5": acc_class0_05,
        "accuracy_class1_at_0.5": acc_class1_05,
    }

    # learning curve data
    learning_curve = None
    if thresholds_for_curve is not None:
        thresholds_for_curve = np.asarray(thresholds_for_curve)
        acc_list = []
        loss_list = []
        for thr in thresholds_for_curve:
            y_pred_thr = (scores >= thr).astype(int)
            acc = float((y_pred_thr == y_true).mean())
            # BCE loss at this threshold
            acc_list.append(acc)
            loss_list.append(loss_overall)

        learning_curve = pd.DataFrame(
            {
                "threshold": thresholds_for_curve,
                "accuracy": acc_list,
                "loss": loss_list,
            }
        )

    return metrics, learning_curve

def build_window_dataframe_stats(
    records,
    window_size = 256,
    stride = 128,
    num_features = 86,
    label_col_name = "is_anomaly",
    scenario_name = None,
    max_windows = None,
    random_state = 42,
):
    """
    Build a DataFrame of windowed statistics from the given records.
    Each window is represented by mean, std, min, max of each feature.
    """
    rng = np.random.default_rng(random_state)

    rows = []
    labels = []
    scenarios = []

    for rec_idx, arr in enumerate(records):
        arr = np.asarray(arr)
        N = arr.shape[0]

        feats_all = arr[:, :num_features]
        if arr.shape[1] > num_features:
            labels_all = arr[:, num_features]
        else:
            labels_all = np.zeros(N, dtype=np.float32)

        starts = list(range(0, N - window_size + 1, stride))
        if max_windows is not None and len(starts) > max_windows:
            starts = list(rng.choice(starts, size = max_windows, replace = False))

        for start in starts:
            end = start + window_size
            window_feats = feats_all[start:end, :]
            window_labels = labels_all[start:end]

            window_label = 1 if window_labels.max() > 0 else 0

            mean = window_feats.mean(axis = 0)
            std = window_feats.std(axis = 0)
            wmin = window_feats.min(axis = 0)
            wmax = window_feats.max(axis = 0)

            feat_vec = np.concatenate([mean, std, wmin, wmax], axis = 0)

            rows.append(feat_vec)
            labels.append(window_label)
            scenarios.append(scenario_name if scenario_name is not None else "unknown")

            if max_windows is not None and len(rows) >= max_windows:
                break

    if not rows:
        return pd.DataFrame()

    X = np.stack(rows, axis = 0)
    feature_names = []
    for stat in ["mean", "std", "min", "max"]:
        feature_names += [f"{stat}_f{i}" for i in range(num_features)]

    df = pd.DataFrame(X, columns=feature_names)
    df[label_col_name] = labels
    df["scenario"] = scenarios

    return df

def main():
    # Run dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs_autogluon", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nSaving outputs to: {run_dir}")

    # Load dataset
    ds = RoADDataset(normalize = False)
    print("Subsets available:", ds.sets.keys())

    base_train_records = ds.sets["training"]
    control_records = ds.sets.get("control", [])

    train_records = list(base_train_records) + list(control_records)

    collision_records = ds.sets.get("collision", [])
    weight_records = ds.sets.get("weight", [])
    velocity_records = ds.sets.get("velocity", [])

    print(f"Number of training records:  {len(train_records)}")
    print(f"Number of collision records: {len(collision_records)}")
    print(f"Number of weight records:    {len(weight_records)}")
    print(f"Number of velocity records:  {len(velocity_records)}")
    print(f"Number of control records:   {len(control_records)}")

    window_size = 256
    stride = 128
    num_features = 86
    label_col = "is_anomaly"

    # For subsampling windows per scenario
    max_windows_per_scenario = 5000

    # 1) min/max only on train
    print("Computing global min/max over the training set...")
    feat_min, feat_max = compute_minmax(train_records, num_features = num_features)

    # 2) Normalize all records
    train_norm = normalize_records(train_records, feat_min, feat_max, num_features = num_features)
    collision_norm = normalize_records(collision_records, feat_min, feat_max, num_features = num_features)
    weight_norm = normalize_records(weight_records, feat_min, feat_max, num_features = num_features)
    velocity_norm = normalize_records(velocity_records, feat_min, feat_max, num_features = num_features)
    control_norm = normalize_records(control_records, feat_min, feat_max, num_features = num_features)

    # 3) DataFrame with windowed stats + subsampling
    print("Building windowed DataFrames with stats and subsampling...")

    df_train = build_window_dataframe_stats(
        train_norm,
        window_size = window_size,
        stride = stride,
        num_features = num_features,
        label_col_name = label_col,
        scenario_name = "train",
        max_windows = max_windows_per_scenario,
        random_state = 0,
    )

    df_collision = build_window_dataframe_stats(
        collision_norm,
        window_size = window_size,
        stride = stride,
        num_features = num_features,
        label_col_name = label_col,
        scenario_name = "collision",
        max_windows = max_windows_per_scenario,
        random_state = 1,
    )

    df_weight = build_window_dataframe_stats(
        weight_norm,
        window_size = window_size,
        stride = stride,
        num_features = num_features,
        label_col_name = label_col,
        scenario_name = "weight",
        max_windows = max_windows_per_scenario,
        random_state = 2,
    )

    df_velocity = build_window_dataframe_stats(
        velocity_norm,
        window_size = window_size,
        stride = stride,
        num_features = num_features,
        label_col_name = label_col,
        scenario_name = "velocity",
        max_windows = max_windows_per_scenario,
        random_state = 3,
    )

    df_control = build_window_dataframe_stats(
        control_norm,
        window_size = window_size,
        stride = stride,
        num_features = num_features,
        label_col_name = label_col,
        scenario_name = "control",
        max_windows = max_windows_per_scenario,
        random_state = 4,
    )

    print("Windows per subset (dopo subsampling):")
    print("  train     :", len(df_train))
    print("  collision :", len(df_collision))
    print("  weight    :", len(df_weight))
    print("  velocity  :", len(df_velocity))
    print("  control   :", len(df_control))

    # 4) Concatenation for all data
    df_all = pd.concat(
        [df_train, df_collision, df_weight, df_velocity, df_control],
        axis = 0,
        ignore_index = True,
    )

    print("Total windows:", len(df_all))
    print("Class balance (0=normal, 1=anom):")
    print(df_all[label_col].value_counts())

    # 5) train/test split stratificato
    train_df, test_df = train_test_split(
        df_all,
        test_size = 0.3,
        random_state = 42,
        stratify = df_all[label_col],
    )

    print("Train windows:", len(train_df))
    print("Test windows :", len(test_df))

    # 6) AutoGluon Tabular
    ag_path = os.path.join(run_dir, "AutogluonModels")

    predictor = TabularPredictor(
        label = label_col,
        problem_type = "binary",
        eval_metric = "roc_auc",
        path = ag_path,
    ).fit(
        train_data = train_df,
        presets = "medium_quality_faster_train",
        time_limit = None,
    )

    # 7) leaderboard 
    leaderboard = predictor.leaderboard(test_df, silent = True)
    leaderboard_csv_path = os.path.join(run_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_csv_path, index = False)
    print(f"Saved AutoGluon leaderboard to {leaderboard_csv_path}")

    # 8) metrics on test (best model)
    y_true_test = test_df[label_col].values
    proba_best = predictor.predict_proba(test_df)  # best model
    if isinstance(proba_best, pd.DataFrame):
        if 1 in proba_best.columns:
            scores_best = proba_best[1].values
        else:
            scores_best = proba_best.iloc[:, 1].values
    else:
        scores_best = np.asarray(proba_best)

    metrics_global = compute_metrics_from_scores(scores_best, y_true_test)
    print("\nEvaluation on test split (best model):")
    for k, v in metrics_global.items():
        print(f"  {k}: {v}")

    # 9) Saving the overall metrics
    results = {
        "config": {
            "window_size": window_size,
            "stride": stride,
            "num_features": num_features,
            "label_col": label_col,
            "max_windows_per_scenario": max_windows_per_scenario,
        },
        "dataset_sizes": {
            "num_windows_total": int(len(df_all)),
            "num_windows_train": int(len(train_df)),
            "num_windows_test": int(len(test_df)),
        },
        "class_balance": df_all[label_col].value_counts().to_dict(),
        "metrics_test_best_model": metrics_global,
        "autogluon_path": ag_path,
    }

    metrics_path = os.path.join(run_dir, "metrics_overall.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved overall metrics to {metrics_path}")

    # 10) Metrics for each model + learning curves
    print("\nComputing per-model metrics and learning curves...")

    # thresholds for learning curve
    thresholds = np.linspace(0.01, 0.99, 99)

    model_names = leaderboard["model"].tolist()

    # lists to build comparison CSV
    comparison_rows = []

    for model_name in model_names:
        print(f"  -> Model: {model_name}")

        model_dir = os.path.join(ag_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Train split
        y_train = train_df[label_col].values
        proba_train = predictor.predict_proba(train_df, model=model_name)

        if isinstance(proba_train, pd.DataFrame):
            if 1 in proba_train.columns:
                scores_train = proba_train[1].values
            else:
                scores_train = proba_train.iloc[:, 1].values
        else:
            scores_train = np.asarray(proba_train)

        metrics_train, _ = compute_bce_losses_and_metrics(
            y_true=y_train,
            scores=scores_train,
            thresholds_for_curve=None,  # learning curve only on test
        )

        # Test split
        y_test = test_df[label_col].values
        proba_test = predictor.predict_proba(test_df, model=model_name)

        if isinstance(proba_test, pd.DataFrame):
            if 1 in proba_test.columns:
                scores_test = proba_test[1].values
            else:
                scores_test = proba_test.iloc[:, 1].values
        else:
            scores_test = np.asarray(proba_test)

        metrics_test, lc_test = compute_bce_losses_and_metrics(
            y_true=y_test,
            scores=scores_test,
            thresholds_for_curve=thresholds,
        )

        # save metrics JSON
        per_model_metrics = {
            "train": metrics_train,
            "test": metrics_test,
        }

        metrics_model_path = os.path.join(model_dir, "metrics_train_test.json")
        with open(metrics_model_path, "w") as f:
            json.dump(per_model_metrics, f, indent=2)

        # save learning curve CSV (test)
        if lc_test is not None:
            lc_path = os.path.join(model_dir, "learning_curve_test.csv")
            lc_test.to_csv(lc_path, index=False)

        # row for comparison CSV 
        comparison_rows.append(
            {
                "model": model_name,

                # Loss
                "bce_loss_overall": metrics_test["bce_loss_overall"],
                "bce_loss_class0": metrics_test["bce_loss_class0"],
                "bce_loss_class1": metrics_test["bce_loss_class1"],

                # Accuracy 
                "accuracy_class0_at_0.5": metrics_test["accuracy_class0_at_0.5"],
                "accuracy_class1_at_0.5": metrics_test["accuracy_class1_at_0.5"],

                # Accuracy 
                "accuracy_at_0.5": metrics_test["accuracy_at_0.5"],

                # Precision
                "precision_at_0.5": metrics_test["precision_at_0.5"],
                "recall_at_0.5": metrics_test["recall_at_0.5"],
                "f1_at_0.5": metrics_test["f1_at_0.5"],

                # best threshold metrics
                "best_threshold": metrics_test["best_threshold"],
                "best_f1": metrics_test["best_f1"],
                "best_precision": metrics_test["best_precision"],
                "best_recall": metrics_test["best_recall"],
            }
        )

    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = os.path.join(run_dir, "models_comparison_test.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved per-model comparison metrics to {comparison_csv_path}")

    print("Done computing per-model metrics and learning curves.")


if __name__ == "__main__":
    main()
