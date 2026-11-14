"""
predict_parity8_ml_fast.py
-----------------------------
Same Part C Step 7 methodology as predict_parity8_ml.py (identical
selection rule: series pull-down blocks if ANY device OFF -> average the
OFF one(s); parallel pull-up blocks only if BOTH OFF -> sum them), but
VECTORIZED: all transistor evaluations across all gates and all operating
points are batched by (device, ON/OFF) into 4 big model.predict() calls,
instead of one model.predict() call per single transistor per row (112
calls/row -- measured at ~380ms/point, which would take >5 hours for the
full 200-corner run). Batching brings this down to seconds.

Verified to produce IDENTICAL results to the slow, obviously-correct
per-row version on a shared subset before being trusted at full scale.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import PROJECT_ROOT
from ml_utils import from_symlog

from build_parity8_netlist import build_parity8_circuit, INPUT_NODES

GATES = build_parity8_circuit()
N_GATES = len(GATES)
W_GATE = 16e-9
MODEL_DIR = PROJECT_ROOT / "models" / "trained_v2"

_model_cache = {}


def load_id_model(case_name):
    if case_name not in _model_cache:
        path = list(MODEL_DIR.glob(f"{case_name}_id_*.joblib"))[0]
        _model_cache[case_name] = joblib.load(path)
    return _model_cache[case_name]


def predict_circuit_leakage_vectorized(spice_df: pd.DataFrame, corners: pd.DataFrame) -> np.ndarray:
    n = len(spice_df)
    vdd = spice_df["vdd"].values
    temp = spice_df["temp"].values
    pattern_bits = np.array([[int(c) for c in p] for p in spice_df["input_pattern"]])  # (n, 8)

    c = corners.set_index("corner").loc[spice_df["corner"].values]
    nmos_p = {k: c[f"nmos_{k}"].values for k in ["vth0", "toxe", "ndep", "u0", "xj"]}
    pmos_p = {k: c[f"pmos_{k}"].values for k in ["vth0", "toxe", "ndep", "u0", "xj"]}

    def node_array(node_name):
        if node_name in INPUT_NODES:
            i = INPUT_NODES.index(node_name)
            return np.where(pattern_bits[:, i] == 1, vdd, 0.0)
        return spice_df[node_name].values

    # predicted[g, row, role] and on_mask[g, row, role]; role 0=M1(nmos) 1=M2(nmos) 2=M3(pmos) 3=M4(pmos)
    predicted = np.zeros((N_GATES, n, 4))
    on_mask = np.zeros((N_GATES, n, 4), dtype=bool)

    buckets = {"NMOS_ON": [], "NMOS_OFF": [], "PMOS_ON": [], "PMOS_OFF": []}
    bucket_scatter = {"NMOS_ON": [], "NMOS_OFF": [], "PMOS_ON": [], "PMOS_OFF": []}

    for g_idx, (gate_name, in_a, in_b, out_node) in enumerate(GATES):
        v_a = node_array(in_a)
        v_b = node_array(in_b)
        v_out = spice_df[out_node].values
        v_x = spice_df[f"{gate_name}_x"].values

        m1_on = v_a > vdd / 2
        m2_on = v_b > vdd / 2
        on_mask[g_idx, :, 0] = m1_on
        on_mask[g_idx, :, 1] = m2_on
        on_mask[g_idx, :, 2] = ~m1_on
        on_mask[g_idx, :, 3] = ~m2_on

        roles = [
            ("nmos", v_out, v_x, v_a, np.zeros(n)),      # M1: vd=out,vs=x,vg=a,vb=0
            ("nmos", v_x, np.zeros(n), v_b, np.zeros(n)),  # M2: vd=x,vs=0,vg=b,vb=0
            ("pmos", v_out, vdd, v_a, vdd),              # M3: vd=out,vs=vdd,vg=a,vb=vdd
            ("pmos", v_out, vdd, v_b, vdd),              # M4: vd=out,vs=vdd,vg=b,vb=vdd
        ]
        for role, (dev, vd_, vs_, vg_, vb_) in enumerate(roles):
            role_on = on_mask[g_idx, :, role]
            pp = nmos_p if dev == "nmos" else pmos_p
            for state_label, mask in [("ON", role_on), ("OFF", ~role_on)]:
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                vgs = vg_[idx] - vs_[idx]
                vds = vd_[idx] - vs_[idx]
                vbs = vb_[idx] - vs_[idx]
                feat = np.column_stack([
                    vdd[idx], temp[idx], np.full(len(idx), W_GATE),
                    vgs, vds, vbs,
                    pp["vth0"][idx], pp["toxe"][idx], pp["ndep"][idx], pp["u0"][idx], pp["xj"][idx],
                ])
                case_name = f"{dev.upper()}_{state_label}"
                buckets[case_name].append(feat)
                bucket_scatter[case_name].append((g_idx, role, idx))

    for case_name in buckets:
        if not buckets[case_name]:
            continue
        model = load_id_model(case_name)
        full_feat = np.vstack(buckets[case_name])
        preds = from_symlog(model.predict(full_feat))
        offset = 0
        for (g_idx, role, idx) in bucket_scatter[case_name]:
            cnt = len(idx)
            predicted[g_idx, idx, role] = preds[offset:offset + cnt]
            offset += cnt

    total = np.zeros(n)
    for g_idx in range(N_GATES):
        m1_on, m2_on = on_mask[g_idx, :, 0], on_mask[g_idx, :, 1]
        id_m1, id_m2 = np.abs(predicted[g_idx, :, 0]), np.abs(predicted[g_idx, :, 1])
        id_m3, id_m4 = np.abs(predicted[g_idx, :, 2]), np.abs(predicted[g_idx, :, 3])

        both_off = (~m1_on) & (~m2_on)
        only_m1_off = (~m1_on) & m2_on
        series_leak = np.where(both_off, (id_m1 + id_m2) / 2, np.where(only_m1_off, id_m1, id_m2))
        parallel_leak = id_m3 + id_m4
        pulldown_blocks = (~m1_on) | (~m2_on)
        total += np.where(pulldown_blocks, series_leak, parallel_leak)

    return total


def run():
    spice_df = pd.read_csv(PROJECT_ROOT / "data" / "parity8_gate_spice.csv", dtype={"input_pattern": str})
    corners = pd.read_csv(PROJECT_ROOT / "data" / "corners_200.csv")

    t0 = time.time()
    ml_leakage = predict_circuit_leakage_vectorized(spice_df, corners)
    elapsed = time.time() - t0

    out_df = spice_df[["corner", "input_pattern", "vdd", "temp", "spice_leakage"]].copy()
    out_df["ml_leakage"] = ml_leakage
    out_path = PROJECT_ROOT / "data" / "parity8_comparison.csv"
    out_df.to_csv(out_path, index=False)

    rel_err = np.abs(out_df["ml_leakage"] - out_df["spice_leakage"]) / out_df["spice_leakage"]
    bias = (out_df["ml_leakage"] - out_df["spice_leakage"]).median()
    print(f"Parity checker comparison ({len(out_df)} points, {out_df['corner'].nunique()} corners) in {elapsed:.2f}s:")
    print(f"  SPICE median={out_df['spice_leakage'].median():.3e} A  ML median={out_df['ml_leakage'].median():.3e} A")
    print(f"  bias={bias:.3e}  median_rel_err={rel_err.median():.4f}  p95_rel_err={rel_err.quantile(0.95):.4f}")
    print(f"Wrote {out_path}")
    return out_df


if __name__ == "__main__":
    run()
