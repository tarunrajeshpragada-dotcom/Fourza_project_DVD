"""
accuracy_summary.py
----------------------
Prints gate- and circuit-level accuracy under BOTH definitions, so the report
can headline the spec's one ("up to 99 % MEAN accuracy"):
  - spec mean accuracy = 100 % x (1 - mean of |ML - SPICE| / SPICE)
  - 100 - median       = 100 % x (1 - median of |ML - SPICE| / SPICE)
Gate figures: averaged over the 18 gate/input cases (each case over 200 corners).
Also prints the composition ceiling: summing the TRUE SPICE leakages of the
stand-alone NAND2 gates (instead of ML predictions) vs the full-circuit SPICE,
i.e. the best the prescribed gate-summation method could ever do.

Run after 06_consolidate_gate_results.py and predict_parity8_compose.py:
    python3 accuracy_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import PROJECT_ROOT

from build_parity8_netlist import INPUT_NODES, build_parity8_circuit

D = PROJECT_ROOT / "data"


def rel(df, ml="ml_leakage"):
    return (df[ml] - df["spice_leakage"]).abs() / df["spice_leakage"]


med, mean = [], []
for g in ["INV", "NAND2", "NOR2", "AND2", "OR2"]:
    d = pd.read_csv(D / f"{g}_gate_comparison.csv", dtype={"input": str})
    for _, e in d.assign(e=rel(d)).groupby("input")["e"]:
        med.append(e.median()); mean.append(e.mean())
print(f"GATES   (18 cases): spec mean accuracy {100 * (1 - np.mean(mean)):.1f}%   100-median {100 * (1 - np.mean(med)):.1f}%")

c = pd.read_csv(D / "parity8_comparison.csv", dtype={"input_pattern": str})
e = rel(c)
bias = np.median(c.ml_leakage / c.spice_leakage) - 1
print(f"CIRCUIT ({len(c)} points, {c.corner.nunique()} corners): spec mean accuracy {100 * (1 - e.mean()):.1f}%   "
      f"100-median {100 * (1 - e.median()):.1f}%   p95 error {100 * e.quantile(0.95):.1f}%   median bias {100 * bias:+.1f}%")

gates = build_parity8_circuit()
nand_sp = pd.read_csv(D / "NAND2_gate_spice.csv", dtype={"input": str}).set_index(["corner", "input"])["spice_leakage"]
circ = pd.read_csv(D / "parity8_gate_spice.csv", dtype={"input_pattern": str})
states = {}
for p in circ.input_pattern.unique():
    v = {n: int(b) for n, b in zip(INPUT_NODES, p)}; s = []
    for _, a, b, out in gates:
        x, y = v[a], v[b]; v[out] = int(not (x and y)); s.append(f"{x}{y}")
    states[p] = s
circ["ceil"] = [sum(nand_sp[(r.corner, k)] for k in states[r.input_pattern]) for r in circ.itertuples(index=False)]
ec = rel(circ, "ceil")
print(f"CEILING (perfect gate models): spec mean accuracy {100 * (1 - ec.mean()):.1f}%   100-median {100 * (1 - ec.median()):.1f}%   "
      f"median bias {100 * (np.median(circ.ceil / circ.spice_leakage) - 1):+.1f}%")
if "spice_input_leak" in circ:
    f = circ.spice_input_leak / circ.spice_leakage * 100
    print(f"Input current at logic-1 inputs vs supply current (circuit): median {f.median():.3f}%   max {f.max():.3f}%")
