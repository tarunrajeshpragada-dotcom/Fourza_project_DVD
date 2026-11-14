"""
run_parity8_spice_varw.py
----------------------------
Part C step 6 with transistor WIDTH varied per corner: the complete 28-NAND2
(112-transistor) parity checker, all 256 input vectors, at the same 200
corners as the gates (data/corners_200.csv with wn, wp). Every NMOS gets the
corner's wn and every PMOS its wp, AD = AS = W x Ldiff.

One ngspice run per corner (256 vectors each, ~20 s), results appended to the
output CSV after every corner, so it is resumable in batches.

Records per point: the parity output voltage (logic check), the VDD supply
current (spice_leakage, the definition used throughout) and the current
delivered by inputs held at logic 1 (spice_input_leak), so the two leakage
definitions can be compared.

Usage:
    python3 run_parity8_spice_varw.py --corners 0-9
    python3 run_parity8_spice_varw.py --corners 10-19   ... up to 190-199
Output:
    data/parity8_gate_spice.csv
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import L_DIFF, L_MIN, MODEL_FILE, NGSPICE_BIN, PROJECT_ROOT

from build_parity8_netlist import INPUT_NODES, OUTPUT_NODE, build_parity8_circuit

GATES = build_parity8_circuit()
PATTERNS = [tuple((p >> i) & 1 for i in range(8)) for p in range(256)]
OUT_CSV = PROJECT_ROOT / "data" / "parity8_gate_spice.csv"
PRINT_VARS = [f"v({OUTPUT_NODE})", "i(vdd_src)"] + [f"i(v{n})" for n in INPUT_NODES]
PRINT_RE = re.compile(r"\s*\n".join(rf"{re.escape(v)}\s*=\s*([\-\d\.eE\+]+)" for v in PRINT_VARS))


def netlist_for_corner(r):
    wn, wp = r.wn, r.wp
    lines = [f"* 8-bit parity checker, corner {r.corner}: wn={wn:.4g} wp={wp:.4g}",
             f".INCLUDE {MODEL_FILE}", ".OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9", ""]
    n = f"W={wn:.6g} L={L_MIN:.6g} AD={wn * L_DIFF:.6g} AS={wn * L_DIFF:.6g}"
    p = f"W={wp:.6g} L={L_MIN:.6g} AD={wp * L_DIFF:.6g} AS={wp * L_DIFF:.6g}"
    for g, a, b, out in GATES:
        x = f"{g}_x"
        lines += [f"M{g}_1 {out} {a} {x} 0 nmos {n}", f"M{g}_2 {x} {b} 0 0 nmos {n}",
                  f"M{g}_3 {out} {a} vdd vdd pmos {p}", f"M{g}_4 {out} {b} vdd vdd pmos {p}"]
    lines.append("Vdd_src vdd 0 0")
    lines += [f"V{node} {node} 0 0" for node in INPUT_NODES]
    lines += ["", ".control",
              # one altermod line PER PARAMETER: ngspice applies only the first parameter of a multi-parameter line
              *[f"altermod @{dev}[{p}]={getattr(r, f'{dev}_{p}'):.8g}" for dev in ["nmos", "pmos"]
                for p in ["vth0", "toxe", "ndep", "u0", "xj"]],
              f"alter Vdd_src = {r.vdd:.8g}", f"option temp = {r.temp:.8g}"]
    for pat in PATTERNS:
        lines += [f"alter V{node} = {r.vdd * bit:.8g}" for node, bit in zip(INPUT_NODES, pat)]
        lines += ["op", "print " + " ".join(PRINT_VARS)]
    lines += [".endc", ".end"]
    return "\n".join(lines)


def run_corner(r):
    path = Path(__file__).resolve().parent / f"_generated_parity8_varw_c{r.corner}.net"
    path.write_text(netlist_for_corner(r))
    res = subprocess.run([NGSPICE_BIN, "-b", str(path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"corner {r.corner}: ngspice failed\n{res.stderr[-2000:]}")
    m = PRINT_RE.findall(res.stdout)
    if len(m) != 256:
        raise RuntimeError(f"corner {r.corner}: parsed {len(m)} of 256\n{res.stderr[-2000:]}")
    path.unlink()
    rows = []
    for pat, vals in zip(PATTERNS, m):
        vals = [float(v) for v in vals]
        rows.append(dict(corner=r.corner, input_pattern="".join(map(str, pat)), vdd=r.vdd, temp=r.temp, wn=r.wn, wp=r.wp,
                         parity_out=vals[0], spice_leakage=abs(vals[1]),
                         spice_input_leak=sum(-i for i, bit in zip(vals[2:], pat) if bit)))
    return rows


def main(lo, hi):
    corners = pd.read_csv(PROJECT_ROOT / "data" / "corners_200.csv")
    if "wn" not in corners:
        sys.exit("corners_200.csv has no wn/wp -- run scripts/05_generate_corners_varw.py first")
    corners = corners[(corners.corner >= lo) & (corners.corner <= hi)]
    new = []
    for r in corners.itertuples(index=False):
        t0 = time.time()
        rows = run_corner(r)
        bad = sum((row["parity_out"] > row["vdd"] / 2) != (sum(map(int, row["input_pattern"])) % 2 == 1) for row in rows)
        print(f"corner {r.corner}: 256 points in {time.time() - t0:.1f}s, logic errors: {bad}")
        new += rows
    new = pd.DataFrame(new)
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV, dtype={"input_pattern": str})
        if "wn" not in old:  # file from the earlier fixed-width run: do not mix
            sys.exit(f"{OUT_CSV} is from the fixed-width run; move it away before starting the varying-width run")
        new = pd.concat([old[~old.corner.isin(new.corner.unique())], new], ignore_index=True)
    new = new.sort_values(["corner", "input_pattern"]).reset_index(drop=True)
    new.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}: {len(new)} rows, {new.corner.nunique()} corners")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True, help="e.g. 0-9")
    lo, hi = map(int, ap.parse_args().corners.split("-"))
    main(lo, hi)
