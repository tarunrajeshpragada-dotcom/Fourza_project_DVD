"""
run_parity8_spice.py
-----------------------
Part C Step 6 (SPICE reference) for the 8-bit parity checker: every one of
the 256 input combinations x 200 corners (the SAME 200 corners used in
Part B, per spec), recording every gate's output + internal node voltage
(56 values) plus the total circuit leakage (Vdd_src current).

Processes ONE CORNER PER ngspice INVOCATION (256 input patterns each,
~20s/corner measured) rather than one giant 51,200-point run (~71 minutes
unbroken) -- each corner's results are appended to the output CSV
immediately, so progress is never lost, it can be resumed, and it's
feasible to run across several shorter sessions instead of one long one.

Usage:
    python3 run_parity8_spice.py                  # all 200 corners
    python3 run_parity8_spice.py --corners 0-9     # just corners 0 through 9
    python3 run_parity8_spice.py --corners 10-19   # resume with the next batch
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import NGSPICE_BIN, PROJECT_ROOT

from build_parity8_netlist import INPUT_NODES, OUTPUT_NODE
from spice_netlist_writer import all_internal_nodes, build_header

NODES = all_internal_nodes()
PRINT_VARS = " ".join(f"v({n})" for n in NODES) + " i(Vdd_src)"
# ngspice lowercases node names internally and prints them lowercase
# regardless of the case used in the netlist (e.g. netlist 'L1_0_c' ->
# printed 'v(l1_0_c)') -- confirmed directly against raw ngspice output
# before trusting this, not assumed.
PRINT_RE = re.compile(
    "".join(rf"v\({re.escape(n.lower())}\)\s*=\s*([\-\d\.eE\+]+)\s*\n" for n in NODES)
    + r"i\(vdd_src\)\s*=\s*([\-\d\.eE\+]+)"
)
ALL_PATTERNS = [tuple((p >> i) & 1 for i in range(8)) for p in range(256)]
OUT_CSV = PROJECT_ROOT / "data" / "parity8_gate_spice.csv"


def build_netlist_for_one_corner(row):
    lines = [build_header()]
    lines.append(
        f"altermod @nmos[vth0]={row.nmos_vth0:.8g} @nmos[toxe]={row.nmos_toxe:.8g} "
        f"@nmos[ndep]={row.nmos_ndep:.8g} @nmos[u0]={row.nmos_u0:.8g} @nmos[xj]={row.nmos_xj:.8g}"
    )
    lines.append(
        f"altermod @pmos[vth0]={row.pmos_vth0:.8g} @pmos[toxe]={row.pmos_toxe:.8g} "
        f"@pmos[ndep]={row.pmos_ndep:.8g} @pmos[u0]={row.pmos_u0:.8g} @pmos[xj]={row.pmos_xj:.8g}"
    )
    lines.append(f"alter Vdd_src = {row.vdd:.8g}")
    lines.append(f"option temp = {row.temp:.8g}")
    for pattern in ALL_PATTERNS:
        for i, bit in enumerate(pattern):
            val = row.vdd if bit == 1 else 0.0
            lines.append(f"alter Va{i} = {val:.8g}")
        lines.append("op")
        lines.append(f"print {PRINT_VARS}")
    lines += [".endc", ".end"]
    return "\n".join(lines)


def run_one_corner(row, net_dir: Path):
    netlist = build_netlist_for_one_corner(row)
    net_path = net_dir / f"_generated_parity8_corner{row.corner}.net"
    net_path.write_text(netlist)

    result = subprocess.run([NGSPICE_BIN, "-b", str(net_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"corner {row.corner}: ngspice failed:\n{result.stderr[-2000:]}")

    matches = PRINT_RE.findall(result.stdout)
    if len(matches) != 256:
        raise RuntimeError(f"corner {row.corner}: parsed {len(matches)} of 256 expected\n"
                            f"--- stderr tail ---\n{result.stderr[-2000:]}")
    net_path.unlink()  # per-corner netlists are large (~230KB each); clean up as we go
    return matches


def parse_corner_range(s):
    lo, hi = s.split("-")
    return int(lo), int(hi)


def run(corner_range=None):
    corners = pd.read_csv(PROJECT_ROOT / "data" / "corners_200.csv")
    if corner_range is not None:
        lo, hi = corner_range
        corners = corners[(corners["corner"] >= lo) & (corners["corner"] <= hi)]

    net_dir = Path(__file__).resolve().parent
    rows_out = []
    for row in corners.itertuples(index=False):
        t0 = time.time()
        matches = run_one_corner(row, net_dir)
        elapsed = time.time() - t0
        for pattern, vals in zip(ALL_PATTERNS, matches):
            node_vals = {n: float(v) for n, v in zip(NODES, vals[:-1])}
            rows_out.append(dict(
                corner=row.corner, input_pattern="".join(str(b) for b in pattern),
                vdd=row.vdd, temp=row.temp, spice_leakage=abs(float(vals[-1])),
                **node_vals,
            ))
        print(f"corner {row.corner}: 256 points in {elapsed:.1f}s")

    out_df = pd.DataFrame(rows_out)
    if OUT_CSV.exists():
        # dtype=str is required here too: without it, a column of all-digit
        # strings like "00000000" gets silently parsed as integer 0,
        # stripping leading zeros -- the same bug class caught earlier for
        # NAND2/NOR2's "00"/"01" input labels, recurring in a new spot.
        existing = pd.read_csv(OUT_CSV, dtype={"input_pattern": str})
        existing = existing[~existing["corner"].isin(out_df["corner"].unique())]
        out_df = pd.concat([existing, out_df], ignore_index=True)
    out_df = out_df.sort_values(["corner", "input_pattern"]).reset_index(drop=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out_df)} total rows across {out_df['corner'].nunique()} corners)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corners", type=str, default=None, help="e.g. '0-9' to run only corners 0 through 9")
    args = parser.parse_args()
    run(corner_range=parse_corner_range(args.corners) if args.corners else None)
