"""
run_inv_spice.py
------------------
Part B Step 3 (SPICE reference) for the INV gate: both input states (A=0,
A=VDD) at all 200 corners = 400 operating points. Output node floats --
SPICE solves the actual circuit self-consistently (not something we compute
ourselves), per Part B Step 4's "output voltage... obtained from the SPICE
simulations."

W is fixed at Wmin (16nm for this node) for every gate transistor -- NOT
randomly swept like Part A -- confirmed from the TA's own worked slide
("Isub(Wmin, 2*NWmin, ...)"). AD=AS=W*Ldiff, same convention as Part A.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import L_DIFF, L_MIN, MODEL_FILE, NGSPICE_BIN, PROJECT_ROOT

W_GATE = L_MIN  # Wmin -- fixed for all gate-level transistors (not randomized)
AD_AS_GATE = W_GATE * L_DIFF

HEADER = f"""\
* INV gate -- SPICE reference, Part B
.INCLUDE {MODEL_FILE}
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9

Mp out gate vdd vdd pmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}
Mn out gate 0   0   nmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}

Vdd_src vdd  0 0
Vin     gate 0 0

.control
"""

PRINT_RE = re.compile(r"v\(out\)\s*=\s*([\-\d\.eE\+]+)\s*\ni\(vdd_src\)\s*=\s*([\-\d\.eE\+]+)")


def build_netlist(corners: pd.DataFrame) -> str:
    lines = [HEADER]
    for row in corners.itertuples(index=False):
        for a_val, a_label in [(0.0, "A0"), (row.vdd, "A1")]:
            lines.append(
                f"altermod @nmos[vth0]={row.nmos_vth0:.8g} @nmos[toxe]={row.nmos_toxe:.8g} "
                f"@nmos[ndep]={row.nmos_ndep:.8g} @nmos[u0]={row.nmos_u0:.8g} @nmos[xj]={row.nmos_xj:.8g}"
            )
            lines.append(
                f"altermod @pmos[vth0]={row.pmos_vth0:.8g} @pmos[toxe]={row.pmos_toxe:.8g} "
                f"@pmos[ndep]={row.pmos_ndep:.8g} @pmos[u0]={row.pmos_u0:.8g} @pmos[xj]={row.pmos_xj:.8g}"
            )
            lines.append(f"alter Vdd_src = {row.vdd:.8g}")
            lines.append(f"alter Vin = {a_val:.8g}")
            lines.append(f"option temp = {row.temp:.8g}")
            lines.append("op")
            lines.append("print v(out) i(Vdd_src)")
    lines += [".endc", ".end"]
    return "\n".join(lines)


def run():
    corners = pd.read_csv(PROJECT_ROOT / "data" / "corners_200.csv")
    netlist = build_netlist(corners)
    net_path = Path(__file__).resolve().parent / "_generated_INV.net"
    net_path.write_text(netlist)

    import time
    t0 = time.time()
    result = subprocess.run([NGSPICE_BIN, "-b", str(net_path)], capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"ngspice failed:\n{result.stderr[-3000:]}")

    matches = PRINT_RE.findall(result.stdout)
    expected = len(corners) * 2
    if len(matches) != expected:
        raise RuntimeError(f"Parsed {len(matches)} but expected {expected}\n--- stderr tail ---\n{result.stderr[-2000:]}")

    rows = []
    idx = 0
    for row in corners.itertuples(index=False):
        for a_val, a_label in [(0.0, "A0"), (row.vdd, "A1")]:
            v_out, i_vdd = matches[idx]
            rows.append(dict(corner=row.corner, input=a_label, A=a_val, vdd=row.vdd, temp=row.temp,
                              nmos_vth0=row.nmos_vth0, nmos_toxe=row.nmos_toxe, nmos_ndep=row.nmos_ndep,
                              nmos_u0=row.nmos_u0, nmos_xj=row.nmos_xj,
                              pmos_vth0=row.pmos_vth0, pmos_toxe=row.pmos_toxe, pmos_ndep=row.pmos_ndep,
                              pmos_u0=row.pmos_u0, pmos_xj=row.pmos_xj,
                              v_out=float(v_out), spice_leakage=abs(float(i_vdd))))
            idx += 1

    out_df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "data" / "INV_gate_spice.csv"
    out_df.to_csv(out_path, index=False)
    print(f"INV: {len(out_df)} operating points in {elapsed:.1f}s")
    print(f"Wrote {out_path}")
    return out_df


if __name__ == "__main__":
    run()
