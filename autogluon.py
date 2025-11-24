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

# Compute global min/max for feature scaling
def compute_minmax(train_records, num_features=86):
    arrays = [np.asarray(rec)[:, :num_features] for rec in train_records]
    big = np.concatenate(arrays, axis=0)

    feat_min = big.min(axis=0)
    feat_max = big.max(axis=0)

    return feat_min.astype(np.float32), feat_max.astype(np.float32)

# Min-Max scaling to [-1, 1]
def minmax_scale(arr, min_vals, max_vals, eps=1e-8):
    arr = arr.astype(np.float32)
    num = arr - min_vals
    den = (max_vals - min_vals) + eps
    return 2.0 * (num / den) - 1.0

# Normalize records
def normalize_records(records, feat_min, feat_max, num_features=86):
    norm_recs = []
    for rec in records:
        arr = np.asarray(rec)
        feats = arr[:, :num_features]
        feats_scaled = minmax_scale(feats, feat_min, feat_max)

        if arr.shape[1] > num_features:
            label = arr[:, num_features : num_features + 1]
            arr_scaled = np.concatenate([feats_scaled, label], axis=1)
        else:
            arr_scaled = feats_scaled

        norm_recs.append(arr_scaled)
    return norm_recs

# Compute metrics from scores and true labels
def compute_metrics_from_scores(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    mean_loss_normal = (
        float(scores[labels == 0].mean()) if np.any(labels == 0) else None
    )
    mean_loss_anom = (
        float(scores[labels == 1].mean()) if np.any(labels == 1) else None
    )

    unique_scores = np.unique(scores)
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = None
    best_conf = None

    for thr in unique_scores:
        y_pred = (scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, y_pred, average="binary", zero_division=0
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

    metrics = {
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

# Compute BCE losses and metrics, with optional learning curve
def compute_bce_losses_and_metrics(y_true, scores, thresholds_for_curve=None):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    eps = 1e-7
    p = np.clip(scores, eps, 1 - eps)

    bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    loss_overall = float(bce.mean())
    loss_class0 = float(bce[y_true == 0].mean()) if np.any(y_true == 0) else None
    loss_class1 = float(bce[y_true == 1].mean()) if np.any(y_true == 1) else None

    auc_roc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    unique_scores = np.unique(scores)
    best_f1 = -1.0
    best_thr = None
    
    curr_thrs = unique_scores if len(unique_scores) < 1000 else np.quantile(unique_scores, np.linspace(0,1,1000))
    for thr in curr_thrs:
        y_pred_thr = (scores >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred_thr, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    y_pred_05 = (scores >= 0.5).astype(int)
    _, _, f1_05, _ = precision_recall_fscore_support(y_true, y_pred_05, average="binary", zero_division=0)
    acc_05 = float((y_pred_05 == y_true).mean())
    
    metrics = {
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "bce_loss_overall": loss_overall,
        "bce_loss_class0": loss_class0,
        "bce_loss_class1": loss_class1,
        "best_f1": float(best_f1),
        "best_threshold": float(best_thr) if best_thr is not None else None,
        "f1_at_0.5": float(f1_05),
        "accuracy_at_0.5": acc_05,
    }

    learning_curve = None
    if thresholds_for_curve is not None:
        thresholds_for_curve = np.asarray(thresholds_for_curve)
        acc_list = []
        f1_list = []
        acc0_list = []
        acc1_list = []
        
        total_pos = (y_true == 1).sum()
        total_neg = (y_true == 0).sum()

        for thr in thresholds_for_curve:
            y_pred_thr = (scores >= thr).astype(int)
            acc = (y_pred_thr == y_true).mean()
            _, _, f1_sc, _ = precision_recall_fscore_support(y_true, y_pred_thr, average="binary", zero_division=0)
            
            tp_t = ((y_true == 1) & (y_pred_thr == 1)).sum()
            tn_t = ((y_true == 0) & (y_pred_thr == 0)).sum()
            
            acc0 = tn_t / total_neg if total_neg > 0 else 0.0
            acc1 = tp_t / total_pos if total_pos > 0 else 0.0
            
            acc_list.append(acc)
            f1_list.append(f1_sc)
            acc0_list.append(acc0)
            acc1_list.append(acc1)

        learning_curve = pd.DataFrame({
            "threshold": thresholds_for_curve,
            "accuracy_overall": acc_list,
            "f1_score": f1_list,
            "accuracy_class0": acc0_list,
            "accuracy_class1": acc1_list,
        })

    return metrics, learning_curve

# Build windowed DataFrame with statistics
def build_window_dataframe_stats(
    records,
    window_size=256,
    stride=128,
    num_features=86,
    label_col_name="is_anomaly",
    scenario_name=None,
    max_windows=None,
    random_state=42,
    feature_names_list=None  
):
    """
    Build a DataFrame of windowed statistics from the given records.
    Uses feature_names_list to create meaningful column names.
    """
    rng = np.random.default_rng(random_state)

    rows = []
    labels = []
    scenarios = []
    
    for rec_idx, arr in enumerate(records):
        arr = np.asarray(arr)
        N = arr.shape[0]
        
        if N < window_size:
            continue

        feats_all = arr[:, :num_features]
        if arr.shape[1] > num_features:
            labels_all = arr[:, num_features]
        else:
            labels_all = np.zeros(N, dtype=np.float32)

        starts = list(range(0, N - window_size + 1, stride))
        
        if max_windows is not None and len(starts) > max_windows:
            pass 
            
        for start in starts:
            end = start + window_size
            window_feats = feats_all[start:end, :]
            window_labels = labels_all[start:end]

            window_label = 1 if window_labels.max() > 0 else 0

            mean = window_feats.mean(axis=0)
            std = window_feats.std(axis=0)
            wmin = window_feats.min(axis=0)
            wmax = window_feats.max(axis=0)

            # Concatenate all statistics into a single feature vector
            feat_vec = np.concatenate([mean, std, wmin, wmax], axis=0)

            rows.append(feat_vec)
            labels.append(window_label)
            scenarios.append(scenario_name if scenario_name is not None else "unknown")

            if max_windows is not None and len(rows) >= max_windows:
                break
        
        if max_windows is not None and len(rows) >= max_windows:
            break

    if not rows:
        return pd.DataFrame()

    X = np.stack(rows, axis=0)
    
    if feature_names_list is None or len(feature_names_list) != num_features:
        feature_names_list = [f"f{i}" for i in range(num_features)]
        
    final_col_names = []

    for stat in ["mean", "std", "min", "max"]:
        final_col_names += [f"{stat}_{fname}" for fname in feature_names_list]

    df = pd.DataFrame(X, columns=final_col_names)
    df[label_col_name] = labels
    df["scenario"] = scenarios

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def main():
    # Run dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs_autogluon", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nSaving outputs to: {run_dir}")

    # Load dataset
    ds = RoADDataset(normalize=False)
    print("Subsets available:", ds.sets.keys())
    
    # Feature names
    if hasattr(ds, 'feature_names') and len(ds.feature_names) == 86:
        original_features = ds.feature_names
        print("Using feature names provided by Dataset.")
    else:
        print("Dataset feature names not found. Using manual list.")

        # original feature names list, taken from dataset documentation
        original_features = [
            "robot_action_id", "apparent_power", "current", "frequency", "phase_angle",
            "power", "power_factor", "reactive_power", "voltage", 
            "j1_acc_x", "j1_acc_y", "j1_acc_z", "j1_gyro_x", "j1_gyro_y", "j1_gyro_z",
            "j1_qw", "j1_qx", "j1_qy", "j1_qz", "j1_temp",
            "j2_acc_x", "j2_acc_y", "j2_acc_z", "j2_gyro_x", "j2_gyro_y", "j2_gyro_z",
            "j2_qw", "j2_qx", "j2_qy", "j2_qz", "j2_temp",
            "j3_acc_x", "j3_acc_y", "j3_acc_z", "j3_gyro_x", "j3_gyro_y", "j3_gyro_z",
            "j3_qw", "j3_qx", "j3_qy", "j3_qz", "j3_temp",
            "j4_acc_x", "j4_acc_y", "j4_acc_z", "j4_gyro_x", "j4_gyro_y", "j4_gyro_z",
            "j4_qw", "j4_qx", "j4_qy", "j4_qz", "j4_temp",
            "j5_acc_x", "j5_acc_y", "j5_acc_z", "j5_gyro_x", "j5_gyro_y", "j5_gyro_z",
            "j5_qw", "j5_qx", "j5_qy", "j5_qz", "j5_temp",
            "j6_acc_x", "j6_acc_y", "j6_acc_z", "j6_gyro_x", "j6_gyro_y", "j6_gyro_z",
            "j6_qw", "j6_qx", "j6_qy", "j6_qz", "j6_temp",
            "j7_acc_x", "j7_acc_y", "j7_acc_z", "j7_gyro_x", "j7_gyro_y", "j7_gyro_z",
            "j7_qw", "j7_qx", "j7_qy", "j7_qz", "j7_temp"
        ]
        
        # Immediate check, for safety
        if len(original_features) != 86:
            raise ValueError(f"ERRORE: La lista contiene {len(original_features)} nomi, ma ne servono 86!")

    # Organize raw records by scenario
    base_train_records = list(ds.sets["training"])
    control_records = list(ds.sets.get("control", []))
    collision_records = list(ds.sets.get("collision", []))
    weight_records = list(ds.sets.get("weight", []))
    velocity_records = list(ds.sets.get("velocity", []))

    train_group_records = base_train_records + control_records

    print(f"Number of training/normal records: {len(train_group_records)}")
    print(f"Number of collision records:       {len(collision_records)}")

    window_size = 256
    stride = 128
    num_features = 86
    label_col = "is_anomaly"
    
    max_windows_train_per_scenario = 4000
    max_windows_test_per_scenario = None

    # 1) Compute min/max
    print("Computing global min/max over the training set...")
    feat_min, feat_max = compute_minmax(train_group_records, num_features=num_features)

    # 2) Normalize
    train_norm = normalize_records(train_group_records, feat_min, feat_max, num_features=num_features)
    collision_norm = normalize_records(collision_records, feat_min, feat_max, num_features=num_features)
    weight_norm = normalize_records(weight_records, feat_min, feat_max, num_features=num_features)
    velocity_norm = normalize_records(velocity_records, feat_min, feat_max, num_features=num_features)

    scenario_data = {
        "normal": train_norm,
        "collision": collision_norm,
        "weight": weight_norm,
        "velocity": velocity_norm
    }

    print("\n--- STARTING DATA SPLIT ---")
    
    train_dfs = []
    test_dfs = []

    # 3) Split records -> Build Windows
    for name, records in scenario_data.items():
        if len(records) == 0:
            continue
            
        if len(records) > 1:
            recs_train, recs_test = train_test_split(records, test_size=0.3, random_state=42)
        else:
            recs_train = records
            recs_test = []

        
        df_tr = build_window_dataframe_stats(
            recs_train,
            window_size=window_size,
            stride=stride,
            num_features=num_features,
            label_col_name=label_col,
            scenario_name=name,
            max_windows=max_windows_train_per_scenario,
            random_state=10 + len(train_dfs),
            feature_names_list=original_features, 
        )
        train_dfs.append(df_tr)

        df_te = build_window_dataframe_stats(
            recs_test,
            window_size=window_size,
            stride=stride,
            num_features=num_features,
            label_col_name=label_col,
            scenario_name=name,
            max_windows=max_windows_test_per_scenario, 
            random_state=20 + len(test_dfs),
            feature_names_list=original_features, 
        )
        test_dfs.append(df_te)

    train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
    test_df = pd.concat(test_dfs, axis=0, ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal Dataset Sizes (Window Level):")
    print(f"  Train Windows: {len(train_df)}")
    print(f"  Test Windows : {len(test_df)}")

    print("\n[SECURITY] Dropping 'scenario' column to prevent Data Leakage...")
    train_data_clean = train_df.drop(columns=["scenario"], errors='ignore')
    test_data_clean = test_df.drop(columns=["scenario"], errors='ignore')
    
    ag_path = os.path.join(run_dir, "AutogluonModels")
    predictor = TabularPredictor(
        label=label_col,
        problem_type="binary",
        eval_metric="roc_auc",
        path=ag_path,
    ).fit(
        train_data=train_data_clean, 
        presets="medium_quality_faster_train",
        time_limit=None,
    )

    leaderboard = predictor.leaderboard(test_data_clean, silent=True)
    leaderboard_csv_path = os.path.join(run_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_csv_path, index=False)

    y_true_test = test_df[label_col].values
    proba_best = predictor.predict_proba(test_data_clean)
    if isinstance(proba_best, pd.DataFrame):
        scores_best = proba_best.iloc[:, 1].values
    else:
        scores_best = np.asarray(proba_best)

    metrics_global = compute_metrics_from_scores(scores_best, y_true_test)
    
    results = {
        "config": {
            "window_size": window_size,
            "stride": stride,
            "num_features": num_features,
            "label_col": label_col,
        },
        "metrics_test_best_model": metrics_global,
        "autogluon_path": ag_path,
    }

    metrics_path = os.path.join(run_dir, "metrics_overall.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # 10) Metrics for each model + FEATURE IMPORTANCE (con nomi veri)
    print("\nComputing per-model metrics and feature importance...")
    thresholds = np.linspace(0.01, 0.99, 99)
    model_names = leaderboard["model"].tolist()
    comparison_rows = []

    for model_name in model_names:
        model_dir = os.path.join(ag_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Processing model: {model_name}")

        proba_train = predictor.predict_proba(train_data_clean, model=model_name)
        scores_train = proba_train.iloc[:, 1].values

        metrics_train, _ = compute_bce_losses_and_metrics(
            y_true=train_df[label_col].values,
            scores=scores_train,
        )

        proba_test = predictor.predict_proba(test_data_clean, model=model_name)
        scores_test = proba_test.iloc[:, 1].values

        metrics_test, lc_test = compute_bce_losses_and_metrics(
            y_true=y_true_test,
            scores=scores_test,
            thresholds_for_curve=thresholds,
        )

        per_model_metrics = {"train": metrics_train, "test": metrics_test}
        with open(os.path.join(model_dir, "metrics_train_test.json"), "w") as f:
            json.dump(per_model_metrics, f, indent=2)

        if lc_test is not None:
            lc_test.to_csv(os.path.join(model_dir, "learning_curve_test.csv"), index=False)

        # Feature importance 
        fi_df = predictor.feature_importance(test_data_clean, model=model_name)
        fi_path = os.path.join(model_dir, "feature_importance.csv")
        fi_df.to_csv(fi_path)
        print(f"     Saved feature importance to {fi_path}")

        comparison_rows.append({
            "model": model_name,
            "bce_loss_overall": metrics_test["bce_loss_overall"],
            "accuracy_at_0.5": metrics_test["accuracy_at_0.5"],
            "f1_at_0.5": metrics_test["f1_at_0.5"],
            "best_f1": metrics_test["best_f1"],
            "roc_auc": metrics_test["auc_roc"]
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(run_dir, "models_comparison_test.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()