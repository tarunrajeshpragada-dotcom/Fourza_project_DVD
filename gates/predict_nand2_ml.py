"""
predict_nand2_ml.py
---------------------
Part B Steps 4-5 for NAND2. Topology: M1(nmos,out,A,x,0), M2(nmos,x,B,0,0)
series pull-down with internal node x; M3(pmos,out,A,vdd,vdd),
M4(pmos,out,B,vdd,vdd) parallel pull-up.

LEAKAGE-COMBINATION RULE (derived and logic-checked against all 4 input
states before implementing -- see chat notes):
  Exactly one network (pull-up or pull-down) is "fully blocking" per input
  state, a guaranteed property of complementary static CMOS. Which one
  blocks, and how to combine its devices' predicted currents:
  - Pull-down (series, M1-M2): blocks if ANY device is OFF. Current = the
    OFF device's prediction (if both OFF, average the two -- they should
    agree by KCL along the series path; this is also a free consistency
    check). The ON device in a blocking series stack is NOT used (same
    near-zero-Vds unreliability established for INV).
  - Pull-up (parallel, M3/M4): blocks ONLY if BOTH are OFF. Current = SUM
    of both devices' predictions (independent parallel branches, KCL at
    the shared output node).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import PROJECT_ROOT
from predict_inv_ml import predict_transistor_id  # reuse the exact same, already-verified predictor

W_GATE = 16e-9


def run():
    # dtype=str on 'input' is required: pandas silently parses "00"/"01" as
    # integers 0/1 otherwise, stripping the leading zero and breaking the
    # "00"/"01"/"10"/"11" string matching used below.
    spice_df = pd.read_csv(PROJECT_ROOT / "data" / "NAND2_gate_spice.csv", dtype={"input": str})
    rows = []

    for row in spice_df.itertuples(index=False):
        vdd, a, b, v_out, v_x = row.vdd, row.A, row.B, row.v_out, row.v_x
        nmos_params = dict(vth0=row.nmos_vth0, toxe=row.nmos_toxe, ndep=row.nmos_ndep, u0=row.nmos_u0, xj=row.nmos_xj)
        pmos_params = dict(vth0=row.pmos_vth0, toxe=row.pmos_toxe, ndep=row.pmos_ndep, u0=row.pmos_u0, xj=row.pmos_xj)

        m1_state = "ON" if a > vdd / 2 else "OFF"   # M1: drain=out, gate=A, source=x, body=0
        m2_state = "ON" if b > vdd / 2 else "OFF"   # M2: drain=x,   gate=B, source=0, body=0
        m3_state = "OFF" if a > vdd / 2 else "ON"   # M3: drain=out, gate=A, source=vdd, body=vdd
        m4_state = "OFF" if b > vdd / 2 else "ON"   # M4: drain=out, gate=B, source=vdd, body=vdd

        id_m1 = predict_transistor_id("nmos", m1_state, vdd, row.temp, vd=v_out, vs=v_x, vg=a, vb=0.0, process_params=nmos_params)
        id_m2 = predict_transistor_id("nmos", m2_state, vdd, row.temp, vd=v_x, vs=0.0, vg=b, vb=0.0, process_params=nmos_params)
        id_m3 = predict_transistor_id("pmos", m3_state, vdd, row.temp, vd=v_out, vs=vdd, vg=a, vb=vdd, process_params=pmos_params)
        id_m4 = predict_transistor_id("pmos", m4_state, vdd, row.temp, vd=v_out, vs=vdd, vg=b, vb=vdd, process_params=pmos_params)

        pulldown_blocks = (m1_state == "OFF") or (m2_state == "OFF")
        pullup_blocks = (m3_state == "OFF") and (m4_state == "OFF")
        assert pulldown_blocks != pullup_blocks, f"expected exactly one network blocking, got pd={pulldown_blocks} pu={pullup_blocks} for AB={row.input}"

        if pulldown_blocks:
            off_vals = [abs(v) for v, s in [(id_m1, m1_state), (id_m2, m2_state)] if s == "OFF"]
            ml_leakage = float(np.mean(off_vals))
            network = "pulldown(series)"
        else:
            ml_leakage = abs(id_m3) + abs(id_m4)
            network = "pullup(parallel)"

        rows.append(dict(corner=row.corner, input=row.input, A=a, B=b, vdd=vdd, temp=row.temp,
                          v_out=v_out, v_x=v_x, spice_leakage=row.spice_leakage, ml_leakage=ml_leakage,
                          blocking_network=network))

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "data" / "NAND2_gate_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print(f"NAND2 gate comparison ({len(out_df)} points):")
    for inp in ["00", "01", "10", "11"]:
        sub = out_df[out_df["input"] == inp]
        rel_err = np.abs(sub["ml_leakage"] - sub["spice_leakage"]) / sub["spice_leakage"]
        print(f"  AB={inp} ({sub['blocking_network'].iloc[0]:17s}): SPICE median={sub['spice_leakage'].median():.3e} A  "
              f"ML median={sub['ml_leakage'].median():.3e} A  "
              f"median_rel_err={rel_err.median():.3f}  p95_rel_err={rel_err.quantile(0.95):.3f}")
    print(f"Wrote {out_path}")
    return out_df


if __name__ == "__main__":
    run()
