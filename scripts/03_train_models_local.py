"""
03_train_models_local.py
---------------------------
Part A Step 2, TA-clarified version (compare 4 techniques per case, keep
ONE winning technique, apply it to all 4 targets in that case) -- run
LOCALLY so training and later model-loading (Part B) happen in the exact
same Python/scikit-learn environment. This sidesteps the joblib
cross-version pickle fragility that broke loading the Colab-trained models
on this machine (HistGradientBoostingRegressor's internal format changed
between sklearn versions enough to break entirely; MLP-based models merely
warned, which is luck, not a guarantee).

Usage:
    python3 03_train_models_local.py
    (trains+saves all 4 cases; takes a few minutes total)

Outputs:
    data/model_comparison_all_cases.csv   -- 64 rows, full diagnostic
    data/case_level_winners.csv           -- 16 rows, the 4x4 deliverable table
    data/test_metrics_all_cases.csv       -- 16 rows, final technique's test metrics
    models/trained_v2/<CASE>_<target>_<technique>.joblib  -- 16 final models
    plots/<CASE>_<target>_pred_vs_spice.png -- 16 predicted-vs-SPICE plots
"""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from common import PROJECT_ROOT, RANDOM_SEED
from ml_utils import TARGET_COLUMNS, build_candidate_models, build_feature_matrix, from_symlog, relative_error, to_symlog

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]

# Accuracy-improvement Option 1: train on the combined (original 10k +
# targeted 3k augmentation) dataset when it exists, so the train/val/test
# split and the model both see the previously-sparse regions properly.
# Falls back to the plain dataset if no combined file is present yet, so
# this script still works before augmentation has been run.
USE_COMBINED_DATASET = True


def split_data(df):
    train_df, temp_df = train_test_split(df, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(temp_df, train_size=rel_val, random_state=RANDOM_SEED)
    return train_df, val_df, test_df


def evaluate(model, X, y_true_raw, y_true_symlog):
    y_pred_symlog = model.predict(X)
    y_pred_raw = from_symlog(y_pred_symlog)
    r2 = r2_score(y_true_symlog, y_pred_symlog)
    rel_err = relative_error(y_true_raw, y_pred_raw)
    return dict(r2=r2, median_rel_err=np.median(rel_err), p95_rel_err=np.percentile(rel_err, 95)), y_pred_raw


def main():
    model_dir = PROJECT_ROOT / "models" / "trained_v2"
    plot_dir = PROJECT_ROOT / "plots"
    model_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    all_comparison_rows, case_level_rows, final_test_rows = [], [], []

    for case_name in CASES:
        t0 = time.time()
        combined_path = PROJECT_ROOT / "data" / f"{case_name}_combined_dataset.csv"
        plain_path = PROJECT_ROOT / "data" / f"{case_name}_dataset.csv"
        data_path = combined_path if (USE_COMBINED_DATASET and combined_path.exists()) else plain_path
        df = pd.read_csv(data_path)
        print(f"[{case_name}] training on {data_path.name} ({len(df)} rows)")
        train_df, val_df, test_df = split_data(df)
        X_train = build_feature_matrix(train_df)
        X_val = build_feature_matrix(val_df)
        X_test = build_feature_matrix(test_df)

        fitted = {}
        for target in TARGET_COLUMNS:
            y_train_raw, y_val_raw = train_df[target].values, val_df[target].values
            y_train_symlog, y_val_symlog = to_symlog(y_train_raw), to_symlog(y_val_raw)

            for model_name, model in build_candidate_models(RANDOM_SEED).items():
                model.fit(X_train, y_train_symlog)
                metrics, _ = evaluate(model, X_val, y_val_raw, y_val_symlog)
                all_comparison_rows.append(dict(case=case_name, target=target, model=model_name, **metrics))
                print(f"[{case_name}][{target}] {model_name:22s} R2={metrics['r2']:.4f}  "
                      f"median_rel_err={metrics['median_rel_err']:.3e}")
                fitted[(target, model_name)] = model

        case_comp = pd.DataFrame([r for r in all_comparison_rows if r["case"] == case_name])
        avg_r2 = case_comp.groupby("model")["r2"].mean().sort_values(ascending=False)
        winner = avg_r2.index[0]
        for model_name, r2v in avg_r2.items():
            case_level_rows.append(dict(case=case_name, model=model_name, avg_val_r2=r2v,
                                         is_winner=(model_name == winner)))
        print(f"[{case_name}] CASE-LEVEL WINNER: {winner} (avg val R2={avg_r2.iloc[0]:.4f})")

        for target in TARGET_COLUMNS:
            best_model = fitted[(target, winner)]
            y_test_raw = test_df[target].values
            y_test_symlog = to_symlog(y_test_raw)
            test_metrics, y_test_pred = evaluate(best_model, X_test, y_test_raw, y_test_symlog)
            final_test_rows.append(dict(case=case_name, target=target, chosen_model=winner, **test_metrics))
            joblib.dump(best_model, model_dir / f"{case_name}_{target}_{winner}.joblib")

            fig, ax = plt.subplots(figsize=(5, 5))
            lims = [min(y_test_raw.min(), y_test_pred.min()), max(y_test_raw.max(), y_test_pred.max())]
            ax.plot(lims, lims, "k--", linewidth=1, label="ideal")
            ax.scatter(y_test_raw, y_test_pred, s=6, alpha=0.4)
            ax.set_xscale("symlog", linthresh=1e-13)
            ax.set_yscale("symlog", linthresh=1e-13)
            ax.set_xlabel(f"SPICE {target} (A)")
            ax.set_ylabel(f"ML-predicted {target} (A)")
            ax.set_title(f"{case_name} -- {target} (test set, {winner})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{case_name}_{target}_pred_vs_spice.png", dpi=150)
            plt.close(fig)

        print(f"[{case_name}] done in {time.time()-t0:.1f}s\n")

    comp_df = pd.DataFrame(all_comparison_rows)
    case_level_df = pd.DataFrame(case_level_rows)
    test_df_out = pd.DataFrame(final_test_rows)

    data_dir = PROJECT_ROOT / "data"
    comp_df.to_csv(data_dir / "model_comparison_all_cases.csv", index=False)
    case_level_df.to_csv(data_dir / "case_level_winners.csv", index=False)
    test_df_out.to_csv(data_dir / "test_metrics_all_cases.csv", index=False)

    print("\n=== Case-level summary (Deliverable 4) ===")
    print(case_level_df.round(4).to_string(index=False))
    print("\n=== Final test metrics (one technique per case) ===")
    print(test_df_out.round(4).to_string(index=False))
    print(f"\nModels saved to {model_dir}/")


if __name__ == "__main__":
    main()
