"""
run_or2_spice.py
------------------
Part B Step 3 (SPICE reference) for OR2 = NOR2 stage + INV stage, one
complete 6-transistor netlist. M1(pmos,n1,A,y,vdd), M2(pmos,y,B,vdd,vdd):
NOR2 pull-up, internal node y. M3(nmos,n1,A,0,0), M4(nmos,n1,B,0,0): NOR2
pull-down. M5(pmos,out,n1,vdd,vdd), M6(nmos,out,n1,0,0): INV stage.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import L_DIFF, L_MIN, MODEL_FILE, NGSPICE_BIN, PROJECT_ROOT

W_GATE = L_MIN
AD_AS_GATE = W_GATE * L_DIFF

HEADER = f"""\
* OR2 gate (NOR2+INV, one netlist) -- SPICE reference, Part B
.INCLUDE {MODEL_FILE}
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9

M1 n1  a y   vdd pmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}
M2 y   b vdd vdd pmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}
M3 n1  a 0   0   nmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}
M4 n1  b 0   0   nmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}
M5 out n1 vdd vdd pmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}
M6 out n1 0   0   nmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}

Vdd_src vdd 0 0
Va      a   0 0
Vb      b   0 0

.control
"""

PRINT_RE = re.compile(
    r"v\(out\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"v\(n1\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"v\(y\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"i\(vdd_src\)\s*=\s*([\-\d\.eE\+]+)"
)

INPUT_STATES = [(0.0, 0.0, "00"), (0.0, 1.0, "01"), (1.0, 0.0, "10"), (1.0, 1.0, "11")]


def build_netlist(corners: pd.DataFrame) -> str:
    lines = [HEADER]
    for row in corners.itertuples(index=False):
        for a_frac, b_frac, label in INPUT_STATES:
            a_val, b_val = a_frac * row.vdd, b_frac * row.vdd
            lines.append(
                f"altermod @nmos[vth0]={row.nmos_vth0:.8g} @nmos[toxe]={row.nmos_toxe:.8g} "
                f"@nmos[ndep]={row.nmos_ndep:.8g} @nmos[u0]={row.nmos_u0:.8g} @nmos[xj]={row.nmos_xj:.8g}"
            )
            lines.append(
                f"altermod @pmos[vth0]={row.pmos_vth0:.8g} @pmos[toxe]={row.pmos_toxe:.8g} "
                f"@pmos[ndep]={row.pmos_ndep:.8g} @pmos[u0]={row.pmos_u0:.8g} @pmos[xj]={row.pmos_xj:.8g}"
            )
            lines.append(f"alter Vdd_src = {row.vdd:.8g}")
            lines.append(f"alter Va = {a_val:.8g}")
            lines.append(f"alter Vb = {b_val:.8g}")
            lines.append(f"option temp = {row.temp:.8g}")
            lines.append("op")
            lines.append("print v(out) v(n1) v(y) i(Vdd_src)")
    lines += [".endc", ".end"]
    return "\n".join(lines)


def run():
    corners = pd.read_csv(PROJECT_ROOT / "data" / "corners_200.csv")
    netlist = build_netlist(corners)
    net_path = Path(__file__).resolve().parent / "_generated_OR2.net"
    net_path.write_text(netlist)

    t0 = time.time()
    result = subprocess.run([NGSPICE_BIN, "-b", str(net_path)], capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"ngspice failed:\n{result.stderr[-3000:]}")

    matches = PRINT_RE.findall(result.stdout)
    expected = len(corners) * 4
    if len(matches) != expected:
        raise RuntimeError(f"Parsed {len(matches)} but expected {expected}\n--- stderr tail ---\n{result.stderr[-2000:]}")

    rows = []
    idx = 0
    for row in corners.itertuples(index=False):
        for a_frac, b_frac, label in INPUT_STATES:
            v_out, v_n1, v_y, i_vdd = matches[idx]
            rows.append(dict(corner=row.corner, input=label, A=a_frac * row.vdd, B=b_frac * row.vdd,
                              vdd=row.vdd, temp=row.temp,
                              nmos_vth0=row.nmos_vth0, nmos_toxe=row.nmos_toxe, nmos_ndep=row.nmos_ndep,
                              nmos_u0=row.nmos_u0, nmos_xj=row.nmos_xj,
                              pmos_vth0=row.pmos_vth0, pmos_toxe=row.pmos_toxe, pmos_ndep=row.pmos_ndep,
                              pmos_u0=row.pmos_u0, pmos_xj=row.pmos_xj,
                              v_out=float(v_out), v_n1=float(v_n1), v_y=float(v_y), spice_leakage=abs(float(i_vdd))))
            idx += 1

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "data" / "OR2_gate_spice.csv"
    out_df.to_csv(out_path, index=False)
    print(f"OR2: {len(out_df)} operating points in {elapsed:.1f}s")
    print(out_df.groupby("input")[["v_out", "v_n1", "v_y", "spice_leakage"]].median())
    print(f"Wrote {out_path}")
    return out_df


if __name__ == "__main__":
    run()
