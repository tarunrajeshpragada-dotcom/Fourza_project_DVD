"""
make_example_netlists.py
---------------------------
Deliverable 2: example netlists, written to <project>/netlists/.

Each file is produced by the SAME netlist-building code the pipeline runs
(scripts/02_run_spice.py, gates/gates_varw.py, circuit/run_parity8_spice_varw.py),
so the examples are exactly what ngspice executes, not hand-written copies:
  - 4 single-transistor netlists: the first sample of each case's inputs file;
  - 5 gate netlists: corner 0 of data/corners_200.csv, looping over that
    corner's input combinations exactly as in Part B;
  - 1 test-circuit netlist: the full 112-transistor parity checker at corner 0.
    The production run loops all 256 input vectors; this example keeps the
    identical topology but one representative vector, so the file stays
    readable (the loop is in circuit/run_parity8_spice_varw.py).

Usage:  python3 make_example_netlists.py       (from anywhere in the project)
"""
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gates"))
sys.path.insert(0, str(HERE.parent / "circuit"))

from common import PROJECT_ROOT

run_spice = import_module("02_run_spice")
gates_varw = import_module("gates_varw")
parity = import_module("run_parity8_spice_varw")

OUT = PROJECT_ROOT / "netlists"
OUT.mkdir(exist_ok=True)
written = []


def write(name, text):
    path = OUT / name
    path.write_text(text if text.endswith("\n") else text + "\n")
    written.append(name)
    print(f"  wrote netlists/{name}")


# 1) single-transistor netlists: first sample of each case
print("[Transistor netlists]")
for case in ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]:
    inputs = PROJECT_ROOT / "data" / f"{case}_inputs.csv"
    if not inputs.exists():
        print(f"  SKIPPED {case} (missing {inputs.name})")
        continue
    row = pd.read_csv(inputs).head(1)
    write(f"transistor_{case}.net", run_spice.build_netlist(case, row))

# 2) gate netlists at corner 0
print("[Gate netlists]")
corners_csv = PROJECT_ROOT / "data" / "corners_200.csv"
if corners_csv.exists():
    corner0 = pd.read_csv(corners_csv).head(1)
    for gate in gates_varw.GATES:
        net, _ = gates_varw.build_netlist(gate, corner0)
        write(f"gate_{gate}.net", net)
else:
    print(f"  SKIPPED (missing {corners_csv.name})")

# 3) test-circuit netlist at corner 0, one representative input vector
print("[Test-circuit netlist]")
if corners_csv.exists():
    r = next(pd.read_csv(corners_csv).head(1).itertuples(index=False))
    full_patterns = parity.PATTERNS
    parity.PATTERNS = full_patterns[:1]            # keep topology, one .op for readability
    try:
        net = parity.netlist_for_corner(r)
    finally:
        parity.PATTERNS = full_patterns
    header = ("* 8-bit parity checker (28 NAND2, 112 transistors), corner 0.\n"
              "* The production run (circuit/run_parity8_spice_varw.py) repeats the\n"
              "* alter/op/print block for all 256 input vectors; topology is identical.\n")
    write("circuit_parity8.net", header + net)

print(f"\n{len(written)} netlists in {OUT}/")
