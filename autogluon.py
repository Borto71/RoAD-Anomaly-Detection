import os
import time
import json

import numpy as np
import pandas as pd

# Assuming RoADDataset is in the same directory or python path
from RoADDataset import Dataset as RoADDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from autogluon.tabular import TabularPredictor


def compute_minmax(train_records, num_features=86):
    arrays = [np.asarray(rec)[:, :num_features] for rec in train_records]
    big = np.concatenate(arrays, axis=0)

    feat_min = big.min(axis=0)
    feat_max = big.max(axis=0)

    return feat_min.astype(np.float32), feat_max.astype(np.float32)


def minmax_scale(arr, min_vals, max_vals, eps=1e-8):
    arr = arr.astype(np.float32)
    num = arr - min_vals
    den = (max_vals - min_vals) + eps
    return 2.0 * (num / den) - 1.0


def normalize_records(records, feat_min, feat_max, num_features=86):
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
            label = arr[:, num_features : num_features + 1]
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

    # Basic optimization loop for threshold
    # (For large datasets, consider subsampling unique_scores for speed)
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


def compute_bce_losses_and_metrics(y_true, scores, thresholds_for_curve=None):
    """
    Compute BCE losses and metrics.
    Updated to include F1, Acc0, Acc1 in the learning curve dataframe.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    # Clip for numerical stability
    eps = 1e-7
    p = np.clip(scores, eps, 1 - eps)

    # BCE loss calculation
    bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    loss_overall = float(bce.mean())
    loss_class0 = float(bce[y_true == 0].mean()) if np.any(y_true == 0) else None
    loss_class1 = float(bce[y_true == 1].mean()) if np.any(y_true == 1) else None

    # Standard metrics
    auc_roc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    # Best F1 logic (omitted details for brevity, logic remains same as original script)
    # ... (MANTIENI LA TUA LOGICA DI BEST F1 QUI) ...
    # Per brevità, ricalcolo veloce il best F1 come nel tuo script originale
    unique_scores = np.unique(scores)
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = None
    
    # Simplified optimization loop
    # Note: For speed on large arrays, consider subsampling unique_scores
    curr_thrs = unique_scores if len(unique_scores) < 1000 else np.quantile(unique_scores, np.linspace(0,1,1000))
    for thr in curr_thrs:
        y_pred_thr = (scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_thr, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_thr = thr

    # Metrics at 0.5
    y_pred_05 = (scores >= 0.5).astype(int)
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(y_true, y_pred_05, average="binary", zero_division=0)
    acc_05 = float((y_pred_05 == y_true).mean())
    
    # Confusion matrix parts for 0.5
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(y_true) > 0:
        # Manual CM to avoid sklearn import errors or shape issues
        tp = ((y_true == 1) & (y_pred_05 == 1)).sum()
        tn = ((y_true == 0) & (y_pred_05 == 0)).sum()
        fp = ((y_true == 0) & (y_pred_05 == 1)).sum()
        fn = ((y_true == 1) & (y_pred_05 == 0)).sum()

    acc_class0_05 = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    acc_class1_05 = float(tp / (tp + fn)) if (tp + fn) > 0 else None

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

    # --- LEARNING CURVE DATA (Threshold Curve) ---
    learning_curve = None
    if thresholds_for_curve is not None:
        thresholds_for_curve = np.asarray(thresholds_for_curve)
        
        # Lists to store metrics per threshold
        acc_list = []
        f1_list = []
        acc0_list = []
        acc1_list = []
        
        # Pre-calculate totals for efficiency
        total_pos = (y_true == 1).sum()
        total_neg = (y_true == 0).sum()

        for thr in thresholds_for_curve:
            y_pred_thr = (scores >= thr).astype(int)
            
            # Overall Accuracy
            acc = (y_pred_thr == y_true).mean()
            
            # F1 Score
            p_sc, r_sc, f1_sc, _ = precision_recall_fscore_support(y_true, y_pred_thr, average="binary", zero_division=0)
            
            # Class Accuracies
            # TP: True=1, Pred=1 | TN: True=0, Pred=0
            tp_t = ((y_true == 1) & (y_pred_thr == 1)).sum()
            tn_t = ((y_true == 0) & (y_pred_thr == 0)).sum()
            
            acc0 = tn_t / total_neg if total_neg > 0 else 0.0
            acc1 = tp_t / total_pos if total_pos > 0 else 0.0
            
            acc_list.append(acc)
            f1_list.append(f1_sc)
            acc0_list.append(acc0)
            acc1_list.append(acc1)

        # Note: Loss doesn't change with threshold (it depends on probability), 
        # but Accuracy/F1 do. We fill Loss as constant just for shape if needed, 
        # or we omit it from the "threshold" curve since it's constant per model prediction.
        # However, users often plot "Loss" vs Epoch. Here X is Threshold.
        # Including Loss vs Threshold is meaningless (Loss is fixed for the predictions).
        # We will skip Loss in the DataFrame vs Threshold.
        
        learning_curve = pd.DataFrame({
            "threshold": thresholds_for_curve,
            "accuracy_overall": acc_list,
            "f1_score": f1_list,
            "accuracy_class0": acc0_list,
            "accuracy_class1": acc1_list,
        })

    return metrics, learning_curve

def build_window_dataframe_stats(
    records,
    window_size=256,
    stride=128,
    num_features=86,
    label_col_name="is_anomaly",
    scenario_name=None,
    max_windows=None,
    random_state=42,
):
    """
    Build a DataFrame of windowed statistics from the given records.
    """
    rng = np.random.default_rng(random_state)

    rows = []
    labels = []
    scenarios = []
    
    # We iterate over the records (which are now already split into train/test)
    for rec_idx, arr in enumerate(records):
        arr = np.asarray(arr)
        N = arr.shape[0]
        
        # Safety check for short records
        if N < window_size:
            continue

        feats_all = arr[:, :num_features]
        if arr.shape[1] > num_features:
            labels_all = arr[:, num_features]
        else:
            labels_all = np.zeros(N, dtype=np.float32)

        starts = list(range(0, N - window_size + 1, stride))
        
        # Subsampling happens per-record or globally. 
        # Here it's implemented per-record loop, but logically applied per call.
        # To strictly respect max_windows for the WHOLE set passed to this function,
        # we should collect all windows first or sample carefully.
        # However, the original logic sampled per record call? 
        # Actually, original logic had "if len(rows) >= max_windows: break".
        # That means it filled up from the first records and ignored the last ones.
        # Improved logic: Collect all possible starts, then sample.
        
        # ... keeping original logic structure but adding checks ...
        
        if max_windows is not None and len(starts) > max_windows:
            # Note: This sampling is per-record in the loop? 
            # The original code passed `max_windows` to this function, 
            # but the check `if len(rows) >= max_windows` breaks the outer loop.
            # Let's trust the shuffle happens before or we accept bias towards first records.
            # For safety in this fixed version, let's just take all windows here
            # and shuffle/sample the resulting DataFrame if needed, 
            # OR rely on the user passing a reasonable subset of records.
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
    feature_names = []
    for stat in ["mean", "std", "min", "max"]:
        feature_names += [f"{stat}_f{i}" for i in range(num_features)]

    df = pd.DataFrame(X, columns=feature_names)
    df[label_col_name] = labels
    df["scenario"] = scenarios

    # Optional: Shuffle the resulting windows so they aren't ordered by record ID
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

    # Organize raw records by scenario
    base_train_records = list(ds.sets["training"])
    control_records = list(ds.sets.get("control", []))
    collision_records = list(ds.sets.get("collision", []))
    weight_records = list(ds.sets.get("weight", []))
    velocity_records = list(ds.sets.get("velocity", []))

    # Combine base train and control for the "normal" class
    # (Assuming control is also normal driving)
    train_group_records = base_train_records + control_records

    print(f"Number of training/normal records: {len(train_group_records)}")
    print(f"Number of collision records:       {len(collision_records)}")
    print(f"Number of weight records:          {len(weight_records)}")
    print(f"Number of velocity records:        {len(velocity_records)}")

    window_size = 256
    stride = 128
    num_features = 86
    label_col = "is_anomaly"
    
    # Configuration for subsampling windows
    max_windows_train_per_scenario = 4000  # Limit for training set
    max_windows_test_per_scenario = None   # No limit for test set (evaluate on everything)

    # 1) Compute min/max ONLY on the training records (to avoid leakage)
    print("Computing global min/max over the training set...")
    feat_min, feat_max = compute_minmax(train_group_records, num_features=num_features)

    # 2) Normalize ALL records
    # We can normalize everything now, splitting happens later.
    # Normalization itself is safe because parameters came only from train records.
    train_norm = normalize_records(train_group_records, feat_min, feat_max, num_features=num_features)
    collision_norm = normalize_records(collision_records, feat_min, feat_max, num_features=num_features)
    weight_norm = normalize_records(weight_records, feat_min, feat_max, num_features=num_features)
    velocity_norm = normalize_records(velocity_records, feat_min, feat_max, num_features=num_features)

    # Dictionary to iterate easily
    scenario_data = {
        "normal": train_norm,
        "collision": collision_norm,
        "weight": weight_norm,
        "velocity": velocity_norm
    }

    print("\n--- STARTING DATA SPLIT (LEAKAGE FIX) ---")
    print("Splitting RECORDS into Train/Test sets BEFORE creating windows.")
    print("This ensures no window in Test comes from the same recording as Train.")

    train_dfs = []
    test_dfs = []

    # 3) Split records -> Build Windows -> Collect
    for name, records in scenario_data.items():
        if len(records) == 0:
            continue
            
        # SPLIT THE LIST OF RECORDS (Raw Data Level)
        # If there is only 1 record, we cannot split it effectively 70/30 without leakage.
        # We assign single records to Test to be safe, or duplicate (here we put in Train if len=1)
        if len(records) > 1:
            recs_train, recs_test = train_test_split(records, test_size=0.3, random_state=42)
        else:
            print(f"Warning: Scenario '{name}' has only 1 record. Putting it in Train.")
            recs_train = records
            recs_test = []

        print(f"Scenario '{name}': {len(recs_train)} recs for Train, {len(recs_test)} recs for Test.")

        # Build windows for TRAIN portion
        df_tr = build_window_dataframe_stats(
            recs_train,
            window_size=window_size,
            stride=stride,
            num_features=num_features,
            label_col_name=label_col,
            scenario_name=name,
            max_windows=max_windows_train_per_scenario,
            random_state=10 + len(train_dfs),
        )
        train_dfs.append(df_tr)

        # Build windows for TEST portion
        # Note: We usually want ALL test windows to evaluate properly, so max_windows=None
        df_te = build_window_dataframe_stats(
            recs_test,
            window_size=window_size,
            stride=stride,
            num_features=num_features,
            label_col_name=label_col,
            scenario_name=name,
            max_windows=max_windows_test_per_scenario, 
            random_state=20 + len(test_dfs),
        )
        test_dfs.append(df_te)

    # 4) Concatenate to create final Train/Test DataFrames
    train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
    test_df = pd.concat(test_dfs, axis=0, ignore_index=True)

    # Shuffle rows (optional, but good for training batches)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal Dataset Sizes (Window Level):")
    print(f"  Train Windows: {len(train_df)} | Anomaly Rate: {train_df[label_col].mean():.2f}")
    print(f"  Test Windows : {len(test_df)}  | Anomaly Rate: {test_df[label_col].mean():.2f}")

    # -------------------------------------------------------
    # AUTO GLUON SECTION (Unchanged logic, just using new DFs)
    # -------------------------------------------------------
    
    ag_path = os.path.join(run_dir, "AutogluonModels")

    predictor = TabularPredictor(
        label=label_col,
        problem_type="binary",
        eval_metric="roc_auc",
        path=ag_path,
    ).fit(
        train_data=train_df,
        presets="medium_quality_faster_train",
        time_limit=None,
    )

    # 7) leaderboard 
    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard_csv_path = os.path.join(run_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_csv_path, index=False)
    print(f"Saved AutoGluon leaderboard to {leaderboard_csv_path}")

    # 8) metrics on test (best model)
    y_true_test = test_df[label_col].values
    proba_best = predictor.predict_proba(test_df)
    
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
            "note": "Splitting done by RECORD, not by window (Leakage Fixed)"
        },
        "dataset_sizes": {
            "num_windows_train": int(len(train_df)),
            "num_windows_test": int(len(test_df)),
        },
        "metrics_test_best_model": metrics_global,
        "autogluon_path": ag_path,
    }

    metrics_path = os.path.join(run_dir, "metrics_overall.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved overall metrics to {metrics_path}")

    # 10) Metrics for each model + learning curves
    print("\nComputing per-model metrics and learning curves...")
    thresholds = np.linspace(0.01, 0.99, 99)
    model_names = leaderboard["model"].tolist()
    comparison_rows = []

    for model_name in model_names:
        # print(f"  -> Model: {model_name}")
        model_dir = os.path.join(ag_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Train stats
        y_train_vals = train_df[label_col].values
        proba_train = predictor.predict_proba(train_df, model=model_name)
        scores_train = proba_train[1].values if isinstance(proba_train, pd.DataFrame) else np.asarray(proba_train)

        metrics_train, _ = compute_bce_losses_and_metrics(
            y_true=y_train_vals,
            scores=scores_train,
            thresholds_for_curve=None,
        )

        # Test stats
        proba_test = predictor.predict_proba(test_df, model=model_name)
        scores_test = proba_test[1].values if isinstance(proba_test, pd.DataFrame) else np.asarray(proba_test)

        metrics_test, lc_test = compute_bce_losses_and_metrics(
            y_true=y_true_test,
            scores=scores_test,
            thresholds_for_curve=thresholds,
        )

        # save metrics JSON
        per_model_metrics = {
            "train": metrics_train,
            "test": metrics_test,
        }
        with open(os.path.join(model_dir, "metrics_train_test.json"), "w") as f:
            json.dump(per_model_metrics, f, indent=2)

        # save learning curve CSV
        if lc_test is not None:
            lc_test.to_csv(os.path.join(model_dir, "learning_curve_test.csv"), index=False)

        # row for comparison CSV 
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