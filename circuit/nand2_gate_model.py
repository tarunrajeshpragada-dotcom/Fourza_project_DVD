"""
nand2_gate_model.py
----------------------
The single-NAND2-gate leakage prediction function, extracted from Part B's
predict_nand2_ml.py (same logic, verified there against 800 real SPICE
points across all 4 input states) so Part C can call it once per gate
instance in the 28-gate parity tree, rather than duplicating the logic.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gates"))
from predict_inv_ml import predict_transistor_id


def predict_nand2_leakage(vdd, temp, v_a, v_b, v_out, v_x, nmos_params, pmos_params):
    """One NAND2 gate instance: M1(nmos,out,A,x,0), M2(nmos,x,B,0,0) series
    pull-down; M3(pmos,out,A,vdd,vdd), M4(pmos,out,B,vdd,vdd) parallel
    pull-up. v_a/v_b are THAT gate's own input voltages (either a primary
    circuit input, or an upstream gate's output voltage); v_out/v_x are
    THAT gate's own SPICE-recorded output/internal-node voltages.
    Returns the predicted leakage for this one gate instance.
    """
    m1_state = "ON" if v_a > vdd / 2 else "OFF"
    m2_state = "ON" if v_b > vdd / 2 else "OFF"
    m3_state = "OFF" if v_a > vdd / 2 else "ON"
    m4_state = "OFF" if v_b > vdd / 2 else "ON"

    id_m1 = predict_transistor_id("nmos", m1_state, vdd, temp, vd=v_out, vs=v_x, vg=v_a, vb=0.0, process_params=nmos_params)
    id_m2 = predict_transistor_id("nmos", m2_state, vdd, temp, vd=v_x, vs=0.0, vg=v_b, vb=0.0, process_params=nmos_params)
    id_m3 = predict_transistor_id("pmos", m3_state, vdd, temp, vd=v_out, vs=vdd, vg=v_a, vb=vdd, process_params=pmos_params)
    id_m4 = predict_transistor_id("pmos", m4_state, vdd, temp, vd=v_out, vs=vdd, vg=v_b, vb=vdd, process_params=pmos_params)

    pulldown_blocks = (m1_state == "OFF") or (m2_state == "OFF")
    pullup_blocks = (m3_state == "OFF") and (m4_state == "OFF")
    assert pulldown_blocks != pullup_blocks, "expected exactly one network blocking"

    if pulldown_blocks:
        off_vals = [abs(v) for v, s in [(id_m1, m1_state), (id_m2, m2_state)] if s == "OFF"]
        return float(np.mean(off_vals))
    else:
        return abs(id_m3) + abs(id_m4)
