"""
predict_inv_ml.py
-------------------
Part B Steps 4-5 for INV. For each of the 400 SPICE-reference operating
points (run_inv_spice.py), determine each transistor's own (Vd,Vg,Vs,Vb)
using the SPICE-SOLVED output voltage (taken as given, per the spec -- we
are NOT re-solving the circuit ourselves), pick the matching trained
transistor model (NMOS/PMOS x ON/OFF, based on that device's own Vg), and
predict its Id.

METHODOLOGY DECISION (flagged, not silently assumed): in any static CMOS
gate's steady state, only one of the two networks (pull-up or pull-down) is
conducting; by KCL the same current flows through the whole series path, so
there is exactly one true leakage value, and the ON-network device and the
OFF-network device are two independent estimates of it. We already found
(the earlier INV sandbox test) that our ON-case models are unreliable
exactly where the ON device sits -- near-zero Vds -- while the OFF-case
models are well-characterized there. So: the gate leakage estimate is the
OFF device's predicted |Id|; the ON device's prediction is still computed
and recorded, but only for diagnostic/error-analysis purposes, not used as
the reported number.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import NOMINAL, PROJECT_ROOT
from ml_utils import build_feature_row, from_symlog

MODEL_DIR = PROJECT_ROOT / "models" / "trained_v2"
W_GATE = 16e-9  # Wmin, matches run_inv_spice.py

_model_cache = {}

# Final setup: the two OFF-case Id models (the only ones that determine gate
# leakage, since the OFF device limits the current) are an ENSEMBLE of
# several independently fitted copies of that case's winning technique,
# trained by 11_train_off_ensemble.py into models/ensemble/. Averaging them
# removes the fit-to-fit variation measured at gate operating points (single
# fits gave 87-90% circuit accuracy; two independent 4-member ensembles gave
# 90.5% and 92.2% on the same data). If that folder is absent, the single
# model in trained_v2 is used, exactly as before.
ENSEMBLE_DIR = PROJECT_ROOT / "models" / "ensemble"


class AveragedModel:
    """Averages member predictions in the symlog (log-compressed) space."""

    def __init__(self, members):
        self.members = members

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


def load_id_model(case_name: str):
    if case_name not in _model_cache:
        members = sorted(ENSEMBLE_DIR.glob(f"{case_name}_id_*_m*.joblib")) if ENSEMBLE_DIR.exists() else []
        if members:
            _model_cache[case_name] = AveragedModel([joblib.load(f) for f in members])
        else:
            path = list(MODEL_DIR.glob(f"{case_name}_id_*.joblib"))[0]
            _model_cache[case_name] = joblib.load(path)
    return _model_cache[case_name]


def predict_transistor_id(dev: str, state: str, vdd, temp, vd, vs, vg, vb, process_params) -> float:
    """dev: 'nmos' or 'pmos'. state: 'ON' or 'OFF'. process_params: dict with
    vth0,toxe,ndep,u0,xj for THIS device. Returns the predicted drain current."""
    case_name = f"{dev.upper()}_{state}"
    model = load_id_model(case_name)
    # build_feature_row computes vgs/vds/vbs from these raw terminal voltages --
    # the EXACT same construction used at training time (ml_utils.build_feature_matrix),
    # so training and inference can never silently diverge in feature definition.
    X = build_feature_row(vdd, temp, W_GATE, vd, vs, vg, vb,
                           process_params["vth0"], process_params["toxe"], process_params["ndep"],
                           process_params["u0"], process_params["xj"])
    return from_symlog(model.predict(X))[0]


def run():
    spice_df = pd.read_csv(PROJECT_ROOT / "data" / "INV_gate_spice.csv")
    rows = []

    for row in spice_df.itertuples(index=False):
        vdd, a, vout = row.vdd, row.A, row.v_out
        nmos_params = dict(vth0=row.nmos_vth0, toxe=row.nmos_toxe, ndep=row.nmos_ndep, u0=row.nmos_u0, xj=row.nmos_xj)
        pmos_params = dict(vth0=row.pmos_vth0, toxe=row.pmos_toxe, ndep=row.pmos_ndep, u0=row.pmos_u0, xj=row.pmos_xj)

        # NMOS: drain=out, gate=A, source=0, body=0 -- matches Part A's NMOS_ON/OFF case definition exactly
        nmos_state = "ON" if a > vdd / 2 else "OFF"
        id_nmos = predict_transistor_id("nmos", nmos_state, vdd, row.temp, vd=vout, vs=0.0, vg=a, vb=0.0,
                                         process_params=nmos_params)

        # PMOS: drain=out, gate=A, source=vdd, body=vdd
        pmos_state = "OFF" if a > vdd / 2 else "ON"
        id_pmos = predict_transistor_id("pmos", pmos_state, vdd, row.temp, vd=vout, vs=vdd, vg=a, vb=vdd,
                                         process_params=pmos_params)

        # Gate leakage estimate = whichever device is OFF (see methodology note above)
        if nmos_state == "OFF":
            ml_leakage = abs(id_nmos)
            off_device, on_device_pred = "nmos", abs(id_pmos)
        else:
            ml_leakage = abs(id_pmos)
            off_device, on_device_pred = "pmos", abs(id_nmos)

        rows.append(dict(corner=row.corner, input=row.input, A=a, vdd=vdd, temp=row.temp, v_out=vout,
                          spice_leakage=row.spice_leakage, ml_leakage=ml_leakage,
                          off_device=off_device, on_device_pred=on_device_pred))

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "data" / "INV_gate_comparison.csv"
    out_df.to_csv(out_path, index=False)

    rel_err = np.abs(out_df["ml_leakage"] - out_df["spice_leakage"]) / out_df["spice_leakage"]
    bias = (out_df["ml_leakage"] - out_df["spice_leakage"]).median()
    print(f"INV gate comparison ({len(out_df)} points):")
    for inp in ["A0", "A1"]:
        sub = out_df[out_df["input"] == inp]
        rel_err_sub = np.abs(sub["ml_leakage"] - sub["spice_leakage"]) / sub["spice_leakage"]
        print(f"  input={inp}: SPICE median={sub['spice_leakage'].median():.3e} A  "
              f"ML median={sub['ml_leakage'].median():.3e} A  "
              f"median_rel_err={rel_err_sub.median():.3f}  p95_rel_err={rel_err_sub.quantile(0.95):.3f}")
    print(f"Wrote {out_path}")
    return out_df


if __name__ == "__main__":
    run()
