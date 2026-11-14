"""
train_final.py
-----------------
Part A step 2, final version. Uniform 10,000-sample datasets only (TA ruling),
physics-based features (physics_features.py), 70/15/15 split with RANDOM_SEED.

For each case (NMOS_ON, NMOS_OFF, PMOS_ON, PMOS_OFF):
  1. Train all six candidate techniques (ml_utils.build_candidate_models) for
     each of the four currents (id, ig, is, ib) on the training split.
  2. Validation metrics per technique and current: R2 (of the log target),
     median and 95th-percentile relative error.
  3. Select ONE technique per case (TA: "compare at least 4, finalize 1"),
     on validation data only (gate/circuit SPICE data are never used):
       - ON cases: highest mean validation R2 over the four currents (the
         rule used throughout this project).
       - OFF cases: highest validation R2 of Id. The gate method uses only
         the OFF devices' drain current, so that is the output that has to be
         right; the other currents must not decide it (PMOS_OFF Ig is entirely
         below 1e-13 A, i.e. simulator noise, so its R2 is meaningless).
     The Id error in the OFF-device operating region (|Vds| > 0.8 VDD) is
     also reported, as supporting evidence on extrapolation to the rails.
  4. Report the chosen technique's test-set metrics and save a
     predicted-vs-SPICE plot per current.
  5. For the two OFF cases, the drain-current model (the one that sets gate
     leakage) is the average of N_SEEDS fits of the chosen technique with
     different seeds, removing fit-to-fit luck.

Usage:   python3 train_final.py                 (all four cases, ~10-20 min)
         python3 train_final.py NMOS_OFF        (one case; tables are re-aggregated)
Outputs: models/final/<CASE>_<current>_<technique>.joblib
         models/final/ensemble/<CASE>_id_<technique>_m<k>.joblib  (OFF cases)
         data/model_comparison_final.csv, data/case_level_winners_final.csv,
         data/test_metrics_final.csv, plots/final_<CASE>_<current>.png
"""
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from common import PROJECT_ROOT, RANDOM_SEED
from ml_utils import build_candidate_models, relative_error
from physics_features import build_physics_features, inverse_target, target

CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
CURRENTS = ["id", "ig", "is", "ib"]
ENSEMBLE_SEEDS = [RANDOM_SEED, 11, 22, 33]
REGION = 0.8             # OFF-device operating region: |Vds| > REGION * VDD
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "models" / "final"
PLOTS = PROJECT_ROOT / "plots"
PER_CASE = DATA / "final_per_case"


def split(df):
    train, rest = train_test_split(df, train_size=0.70, random_state=RANDOM_SEED)
    val, test = train_test_split(rest, train_size=0.5, random_state=RANDOM_SEED)
    return train, val, test


def metrics(model, X, df, col):
    y = model.predict(X)
    pred = inverse_target(y, df["w"].values)
    e = relative_error(df[col].values, pred)
    return dict(r2=r2_score(target(df, col), y), median_rel_err=np.median(e), p95_rel_err=np.percentile(e, 95)), pred


def run_case(case):
    t0 = time.time()
    df = pd.read_csv(DATA / f"{case}_dataset.csv")
    train, val, test = split(df)
    Xtr, Xva, Xte = (build_physics_features(d) for d in (train, val, test))

    comp, fitted = [], {}
    for col in CURRENTS:
        for tech, model in build_candidate_models(RANDOM_SEED).items():
            model.fit(Xtr, target(train, col))
            m, pred = metrics(model, Xva, val, col)
            if col == "id":
                hi = ((val["vd"] - val["vs"]).abs() > REGION * val["vdd"]).values
                m["id_region_median_rel_err"] = np.median(relative_error(val["id"].values[hi], pred[hi]))
            comp.append(dict(case=case, current=col, model=tech, **m))
            fitted[(col, tech)] = model
            print(f"[{case}][{col}] {tech:27s} R2={m['r2']:.4f}  median={m['median_rel_err']:.4f}  p95={m['p95_rel_err']:.3f}")
    comp = pd.DataFrame(comp)
    score = comp.groupby("model")["r2"].mean().sort_values(ascending=False)
    id_r2 = comp[comp.current == "id"].set_index("model")["r2"]
    region = comp[comp.current == "id"].set_index("model")["id_region_median_rel_err"]
    if case.endswith("_OFF"):
        winner = id_r2.idxmax()
        rule = f"highest validation R2 of Id = {id_r2[winner]:.5f}"
    else:
        winner = score.index[0]
        rule = f"highest mean validation R2 = {score[winner]:.5f}"
    print(f"[{case}] CHOSEN: {winner} ({rule}; Id error at |Vds|>{REGION}VDD = {region[winner]:.3f})")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob(f"{case}_*.joblib"):
        old.unlink()
    test_rows = []
    for col in CURRENTS:
        model = fitted[(col, winner)]
        joblib.dump(model, OUT / f"{case}_{col}_{winner}.joblib")
        m, pred = metrics(model, Xte, test, col)
        test_rows.append(dict(case=case, current=col, chosen_model=winner, variant="single", **m))
        fig, ax = plt.subplots(figsize=(5, 5))
        t = test[col].values
        lims = [min(t.min(), pred.min()), max(t.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1, label="ideal")
        ax.scatter(t, pred, s=6, alpha=0.4)
        ax.set_xscale("symlog", linthresh=1e-13); ax.set_yscale("symlog", linthresh=1e-13)
        ax.set_xlabel(f"SPICE {col} (A)"); ax.set_ylabel(f"ML {col} (A)")
        ax.set_title(f"{case} {col} (test, {winner})"); ax.legend()
        fig.tight_layout(); PLOTS.mkdir(exist_ok=True); fig.savefig(PLOTS / f"final_{case}_{col}.png", dpi=150); plt.close(fig)

    if case.endswith("_OFF"):
        ens_dir = OUT / "ensemble"; ens_dir.mkdir(exist_ok=True)
        for old in ens_dir.glob(f"{case}_*.joblib"):
            old.unlink()
        members = []
        for k, seed in enumerate(ENSEMBLE_SEEDS):
            model = fitted[("id", winner)] if seed == RANDOM_SEED else build_candidate_models(seed)[winner].fit(Xtr, target(train, "id"))
            joblib.dump(model, ens_dir / f"{case}_id_{winner}_m{k}.joblib")
            members.append(model)
        y = np.mean([m.predict(Xte) for m in members], axis=0)
        e = relative_error(test["id"].values, inverse_target(y, test["w"].values))
        test_rows.append(dict(case=case, current="id", chosen_model=winner, variant=f"average of {len(members)}",
                              r2=r2_score(target(test, "id"), y), median_rel_err=np.median(e), p95_rel_err=np.percentile(e, 95)))
        print(f"[{case}] averaged Id model ({len(members)} seeds): test median error {np.median(e):.4f}")

    PER_CASE.mkdir(exist_ok=True)
    comp.to_csv(PER_CASE / f"comparison_{case}.csv", index=False)
    pd.DataFrame(test_rows).to_csv(PER_CASE / f"test_{case}.csv", index=False)
    pd.DataFrame([dict(case=case, model=k, mean_val_r2=v, id_val_r2=id_r2[k], id_region_median_rel_err=region[k],
                       is_winner=(k == winner)) for k, v in score.items()]
                 ).to_csv(PER_CASE / f"winners_{case}.csv", index=False)
    print(f"[{case}] done in {time.time() - t0:.0f}s\n")


def aggregate():
    for prefix, name in [("comparison", "model_comparison_final"), ("test", "test_metrics_final"), ("winners", "case_level_winners_final")]:
        files = sorted(PER_CASE.glob(f"{prefix}_*.csv"))
        if files:
            pd.concat([pd.read_csv(f) for f in files]).to_csv(DATA / f"{name}.csv", index=False)
    w = pd.read_csv(DATA / "case_level_winners_final.csv")
    print("=== Chosen technique per case ===")
    print(w[w.is_winner].to_string(index=False))
    t = pd.read_csv(DATA / "test_metrics_final.csv")
    print("\n=== Test metrics ===")
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    cases = sys.argv[1:] or CASES
    for c in cases:
        run_case(c)
    aggregate()
