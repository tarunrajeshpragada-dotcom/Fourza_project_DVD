"""
11_train_off_ensemble.py
---------------------------
Trains the final OFF-case Id models as an ENSEMBLE of N_MEMBERS independent
fits of each case's winning technique (read from data/case_level_winners.csv,
so it follows whatever 03_train_models_local.py selected on this machine).

Why only the OFF cases, only Id: in every static CMOS gate the OFF device
limits the leakage current, so the gate/circuit predictions use only the
NMOS_OFF and PMOS_OFF drain-current models. The other 14 models are unchanged.

Why an ensemble: at gate operating points, refitting the same technique on
resampled data moved circuit accuracy between ~87% and ~90%. Each member is
fit on a bootstrap resample (different fixed seed); predictions are averaged.
It is still ONE technique per case (bagged), in line with the TA's
"compare 4 techniques, finalize 1".

Training data per case = training split (70%, RANDOM_SEED) of
<CASE>_combined_dataset.csv (original 10k + targeted 3k) + training split
of <CASE>_gatecond_dataset.csv (minimum-width, gate-condition samples).
Held-out 15%+15% of both are used only for the printed error check.

Usage:  python3 11_train_off_ensemble.py
Output: models/ensemble/<CASE>_id_<technique>_m<k>.joblib  (k = 0..N_MEMBERS-1)
"""
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import PROJECT_ROOT, RANDOM_SEED
from ml_utils import build_candidate_models, build_feature_matrix, from_symlog, relative_error, to_symlog

N_MEMBERS = 4
MEMBER_SEEDS = [404, 505, 606, 707]
CASES = ["NMOS_OFF", "PMOS_OFF"]
DATA = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "models" / "ensemble"


def main():
    winners = pd.read_csv(DATA / "case_level_winners.csv")
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True)

    for case in CASES:
        tech = winners[(winners["case"] == case) & (winners["is_winner"])]["model"].iloc[0]
        base = pd.read_csv(DATA / f"{case}_combined_dataset.csv")
        gc = pd.read_csv(DATA / f"{case}_gatecond_dataset.csv")
        tr_b, te_b = train_test_split(base, train_size=0.70, random_state=RANDOM_SEED)
        tr_g, te_g = train_test_split(gc, train_size=0.70, random_state=RANDOM_SEED)
        train = pd.concat([tr_b, tr_g], ignore_index=True)

        members = []
        for k, seed in enumerate(MEMBER_SEEDS[:N_MEMBERS]):
            sample = train.sample(frac=1.0, replace=True, random_state=seed)
            model = build_candidate_models(seed)[tech]
            model.fit(build_feature_matrix(sample), to_symlog(sample["id"].values))
            joblib.dump(model, OUT_DIR / f"{case}_id_{tech}_m{k}.joblib")
            members.append(model)
            print(f"[{case}] member {k} ({tech}, seed {seed}) trained on {len(sample)} rows")

        for name, te in [("original held-out", te_b), ("gate-condition held-out", te_g)]:
            X = build_feature_matrix(te)
            pred = from_symlog(np.mean([m.predict(X) for m in members], axis=0))
            err = np.median(relative_error(te["id"].values, pred))
            print(f"[{case}] ensemble median_rel_err on {name}: {err:.4f}")

    print(f"Saved ensemble members to {OUT_DIR}/")


if __name__ == "__main__":
    main()
