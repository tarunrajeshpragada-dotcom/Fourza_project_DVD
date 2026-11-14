"""
predict_and2_ml.py
---------------------
Part B Steps 4-5 for AND2. Two independent series paths share the same
Vdd_src (NAND2 stage: M1-M4; INV stage: M5-M6), so total gate leakage =
NAND2-stage leakage + INV-stage leakage (KCL: the single Vdd_src's current
splits into both branches; total = sum of both). Each stage uses exactly
the same combination rule already validated standalone (NAND2: series
pull-down / parallel pull-up; INV: single OFF device).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import PROJECT_ROOT
from predict_inv_ml import predict_transistor_id


def run():
    spice_df = pd.read_csv(PROJECT_ROOT / "data" / "AND2_gate_spice.csv", dtype={"input": str})
    rows = []

    for row in spice_df.itertuples(index=False):
        vdd, a, b, v_out, v_n1, v_x = row.vdd, row.A, row.B, row.v_out, row.v_n1, row.v_x
        nmos_params = dict(vth0=row.nmos_vth0, toxe=row.nmos_toxe, ndep=row.nmos_ndep, u0=row.nmos_u0, xj=row.nmos_xj)
        pmos_params = dict(vth0=row.pmos_vth0, toxe=row.pmos_toxe, ndep=row.pmos_ndep, u0=row.pmos_u0, xj=row.pmos_xj)

        # --- NAND2 stage (M1-M4): output=n1, internal node=x ---
        m1_state = "ON" if a > vdd / 2 else "OFF"
        m2_state = "ON" if b > vdd / 2 else "OFF"
        m3_state = "OFF" if a > vdd / 2 else "ON"
        m4_state = "OFF" if b > vdd / 2 else "ON"
        id_m1 = predict_transistor_id("nmos", m1_state, vdd, row.temp, vd=v_n1, vs=v_x, vg=a, vb=0.0, process_params=nmos_params)
        id_m2 = predict_transistor_id("nmos", m2_state, vdd, row.temp, vd=v_x, vs=0.0, vg=b, vb=0.0, process_params=nmos_params)
        id_m3 = predict_transistor_id("pmos", m3_state, vdd, row.temp, vd=v_n1, vs=vdd, vg=a, vb=vdd, process_params=pmos_params)
        id_m4 = predict_transistor_id("pmos", m4_state, vdd, row.temp, vd=v_n1, vs=vdd, vg=b, vb=vdd, process_params=pmos_params)
        pulldown_blocks = (m1_state == "OFF") or (m2_state == "OFF")
        if pulldown_blocks:
            off_vals = [abs(v) for v, s in [(id_m1, m1_state), (id_m2, m2_state)] if s == "OFF"]
            nand2_leakage = float(np.mean(off_vals))
        else:
            nand2_leakage = abs(id_m3) + abs(id_m4)

        # --- INV stage (M5-M6): input=n1 (continuous, not exactly 0/VDD), output=out ---
        m5_state = "OFF" if v_n1 > vdd / 2 else "ON"   # pmos
        m6_state = "ON" if v_n1 > vdd / 2 else "OFF"   # nmos
        id_m5 = predict_transistor_id("pmos", m5_state, vdd, row.temp, vd=v_out, vs=vdd, vg=v_n1, vb=vdd, process_params=pmos_params)
        id_m6 = predict_transistor_id("nmos", m6_state, vdd, row.temp, vd=v_out, vs=0.0, vg=v_n1, vb=0.0, process_params=nmos_params)
        inv_leakage = abs(id_m5) if m5_state == "OFF" else abs(id_m6)

        ml_leakage = nand2_leakage + inv_leakage
        rows.append(dict(corner=row.corner, input=row.input, A=a, B=b, vdd=vdd, temp=row.temp,
                          v_out=v_out, v_n1=v_n1, v_x=v_x, spice_leakage=row.spice_leakage,
                          ml_leakage=ml_leakage, nand2_stage=nand2_leakage, inv_stage=inv_leakage))

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "data" / "AND2_gate_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print(f"AND2 gate comparison ({len(out_df)} points):")
    for inp in ["00", "01", "10", "11"]:
        sub = out_df[out_df["input"] == inp]
        rel_err = np.abs(sub["ml_leakage"] - sub["spice_leakage"]) / sub["spice_leakage"]
        print(f"  AB={inp}: SPICE median={sub['spice_leakage'].median():.3e} A  "
              f"ML median={sub['ml_leakage'].median():.3e} A  "
              f"median_rel_err={rel_err.median():.3f}  p95_rel_err={rel_err.quantile(0.95):.3f}")
    print(f"Wrote {out_path}")
    return out_df


if __name__ == "__main__":
    run()
