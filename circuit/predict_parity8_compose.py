"""
predict_parity8_compose.py
-----------------------------
Part C Step 7, as the spec states it: "Build the test circuit from your gate
models of Part B ... For each input vector, determine the logic value at the
inputs of every gate, call the corresponding gate model, and sum the logic
gate leakages."

For each (corner, 8-bit input pattern): evaluate the parity tree in LOGIC
(0/1) to get each of the 28 NAND2 gates' two input values, then add up the
Part B NAND2 gate model's predicted leakage for that same corner and input
pair (data/NAND2_gate_comparison.csv). No circuit SPICE voltage is used to
make the prediction; the circuit SPICE file supplies only the reference
leakage to compare against. This embodies the spec's stated assumptions
(gate inputs exactly at 0 or VDD, no loading between gates).

Ceiling check (perfect gate models, i.e. summing the TRUE SPICE NAND2 gate
leakages): 97.0% accuracy on 20 corners, so this composition method itself
costs only ~3%.

Replaces predict_parity8_ml_fast.py (which fed circuit-SPICE node voltages
into transistor models -- not what the spec describes). Writes the same
output file, so consolidate_parity8_results.py works unchanged.

Usage:  python3 predict_parity8_compose.py   (after the Part B NAND2 scripts)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import PROJECT_ROOT

from build_parity8_netlist import INPUT_NODES, build_parity8_circuit

GATES = build_parity8_circuit()


def gate_input_states(pattern: str):
    """Logic value at every node; returns each gate's input pair, e.g. '01'."""
    val = {node: int(bit) for node, bit in zip(INPUT_NODES, pattern)}
    states = []
    for _, in_a, in_b, out_node in GATES:
        a, b = val[in_a], val[in_b]
        val[out_node] = int(not (a and b))
        states.append(f"{a}{b}")
    return states


def run():
    data = PROJECT_ROOT / "data"
    nand_ml = (pd.read_csv(data / "NAND2_gate_comparison.csv", dtype={"input": str})
               .set_index(["corner", "input"])["ml_leakage"])
    circ = pd.read_csv(data / "parity8_gate_spice.csv", dtype={"input_pattern": str})

    states = {p: gate_input_states(p) for p in circ["input_pattern"].unique()}
    ml = np.array([sum(nand_ml[(r.corner, s)] for s in states[r.input_pattern])
                   for r in circ.itertuples(index=False)])

    out = circ[["corner", "input_pattern", "vdd", "temp", "spice_leakage"]].copy()
    out["ml_leakage"] = ml
    out.to_csv(data / "parity8_comparison.csv", index=False)

    err = np.abs(out["ml_leakage"] - out["spice_leakage"]) / out["spice_leakage"]
    print(f"Composed parity checker ({len(out)} points, {out['corner'].nunique()} corners):")
    print(f"  median_rel_err={err.median():.4f}  p95_rel_err={err.quantile(0.95):.4f}  "
          f"ML/SPICE median ratio={np.median(out['ml_leakage'] / out['spice_leakage']):.3f}")
    print(f"Wrote {data / 'parity8_comparison.csv'}")


if __name__ == "__main__":
    run()
