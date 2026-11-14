"""
02_run_spice.py
------------------
Simulates every sampled input row of one case (data/<CASE>_inputs.csv, from
01_generate_inputs.py) in ngspice and writes data/<CASE>_dataset.csv with the
16 columns required by Deliverable 1:
    vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib

Samples are simulated in chunks of CHUNK rows, one ngspice run per chunk
(ngspice slows down superlinearly when one run accumulates many thousands of
results: 1,000 samples take ~1 s, 10,000 in one run took minutes). Each
netlist loads the model card once and loops over its samples in a .control block, setting each sample's process
parameters (altermod), width and junction areas (alter), terminal voltages
and temperature, then runs .op and prints the four terminal currents.

IMPORTANT (bug fixed here): ngspice's altermod applies ONLY THE FIRST
parameter when several are given on one line, with either syntax
("altermod @nmos[vth0]=.. @nmos[toxe]=.." or "altermod nmos vth0=.. toxe=..").
Each process parameter therefore gets its OWN altermod line. Verified: five
separate lines give currents identical to editing all five values directly
into the model card.

Usage:   python3 02_run_spice.py NMOS_ON
"""
import re
import subprocess
import sys
import time

import pandas as pd

from common import L_DIFF, L_MIN, MODEL_FILE, NGSPICE_BIN, PROJECT_ROOT

CASE_MODEL = {"NMOS_ON": "nmos", "NMOS_OFF": "nmos", "PMOS_ON": "pmos", "PMOS_OFF": "pmos"}
PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]
PRINT_RE = re.compile(
    r"i\(vd\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"i\(vg\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"i\(vs\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"i\(vb\)\s*=\s*([\-\d\.eE\+]+)"
)
CHUNK = 1000
COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]


def build_netlist(case, df):
    mdl = CASE_MODEL[case]
    ad0 = L_MIN * L_DIFF
    lines = [f"* Single-transistor PVT sweep -- case {case}", f".INCLUDE {MODEL_FILE}",
             ".OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9", "",
             f"Mn drain gate source body {mdl} W={L_MIN:.6g} L={L_MIN:.6g} AD={ad0:.6g} AS={ad0:.6g}",
             "Vd drain 0 0", "Vg gate 0 0", "Vs source 0 0", "Vb body 0 0", "", ".control"]
    for r in df.itertuples(index=False):
        for p in PARAMS:                                   # one altermod line per parameter
            lines.append(f"altermod @{mdl}[{p}]={getattr(r, p):.8g}")
        lines += [f"alter @mn[w] = {r.w:.8g}", f"alter @mn[ad] = {r.w * L_DIFF:.8g}", f"alter @mn[as] = {r.w * L_DIFF:.8g}",
                  f"alter Vd = {r.vd:.8g}", f"alter Vg = {r.vg:.8g}", f"alter Vs = {r.vs:.8g}", f"alter Vb = {r.vb:.8g}",
                  f"option temp = {r.temp:.8g}", "op", "print i(vd) i(vg) i(vs) i(vb)"]
    lines += [".endc", ".end"]
    return "\n".join(lines)


def run_case(case, in_csv_name=None, out_csv_name=None):
    df = pd.read_csv(PROJECT_ROOT / "data" / (in_csv_name or f"{case}_inputs.csv"))
    net = PROJECT_ROOT / "data" / f"_generated_{case}.net"
    t0 = time.time()
    m = []
    for start in range(0, len(df), CHUNK):
        chunk = df.iloc[start:start + CHUNK]
        net.write_text(build_netlist(case, chunk))
        res = subprocess.run([NGSPICE_BIN, "-b", str(net)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ngspice failed (rows {start}+):\n{res.stderr[-3000:]}")
        got = PRINT_RE.findall(res.stdout)
        if len(got) != len(chunk):
            raise RuntimeError(f"rows {start}+: parsed {len(got)} of {len(chunk)} -- check convergence:\n{res.stderr[-2000:]}")
        m += got
    net.unlink()
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(m, columns=["id", "ig", "is", "ib"]).astype(float)], axis=1)[COLUMNS]
    path = PROJECT_ROOT / "data" / (out_csv_name or f"{case}_dataset.csv")
    out.to_csv(path, index=False)
    kcl = (out["id"] + out["ig"] + out["is"] + out["ib"]).abs()
    print(f"[{case}] {len(out)} rows in {time.time() - t0:.1f}s; KCL residual median {kcl.median():.3e} A "
          f"(median |Id| {out['id'].abs().median():.3e} A); wrote {path}")
    return out


if __name__ == "__main__":
    run_case(sys.argv[1] if len(sys.argv) > 1 else "NMOS_ON")
