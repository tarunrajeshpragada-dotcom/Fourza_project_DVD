import re, subprocess, sys, time
from pathlib import Path
import pandas as pd
from common import CASES, MODEL_FILE, L_MIN, L_DIFF, NGSPICE_BIN, PROJECT_ROOT

HEADER = """\
* Auto-generated single-transistor PVT sweep -- case: {case}
* AD/AS = W x Ldiff (junction area), per TA guidance -- re-altered every sample below.
.INCLUDE {model_file}
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9

Mn drain gate source body {devmodel} W={lmin:.6g} L={lmin:.6g} AD=0 AS=0
Vd drain 0 0
Vg gate  0 0
Vs source 0 0
Vb body  0 0

.control
"""

PRINT_RE = re.compile(
    r"i\(vd\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"i\(vg\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"i\(vs\)\s*=\s*([\-\d\.eE\+]+)\s*\n"
    r"i\(vb\)\s*=\s*([\-\d\.eE\+]+)"
)

def build_netlist(case_name, df, model_file):
    case = CASES[case_name]
    lines = [HEADER.format(case=case_name, model_file=model_file, devmodel=case["model"], lmin=L_MIN)]
    mdl = case["model"]
    for row in df.itertuples(index=False):
        ad_as = row.w * L_DIFF
        lines.append(f"altermod @{mdl}[vth0]={row.vth0:.8g} @{mdl}[toxe]={row.toxe:.8g} "
                     f"@{mdl}[ndep]={row.ndep:.8g} @{mdl}[u0]={row.u0:.8g} @{mdl}[xj]={row.xj:.8g}")
        lines.append(f"alter @mn[w] = {row.w:.8g}")
        lines.append(f"alter @mn[ad] = {ad_as:.8g}")
        lines.append(f"alter @mn[as] = {ad_as:.8g}")
        lines.append(f"alter Vd = {row.vd:.8g}")
        lines.append(f"alter Vg = {row.vg:.8g}")
        lines.append(f"alter Vs = {row.vs:.8g}")
        lines.append(f"alter Vb = {row.vb:.8g}")
        lines.append(f"option temp = {row.temp:.8g}")
        lines.append("op")
        lines.append("print i(vd) i(vg) i(vs) i(vb)")
    lines += [".endc", ".end"]
    return "\n".join(lines)

def run_case(case_name, model_file=MODEL_FILE, in_csv_name=None, out_csv_name=None, net_suffix=None):
    """in_csv_name/out_csv_name: override the default '<case>_inputs.csv' /
    '<case>_dataset.csv' filenames -- added to reuse this exact, already-verified
    netlist/parsing logic for the targeted augmentation samples (Option 1,
    accuracy improvement) without duplicating it. Defaults are unchanged, so
    every existing call site behaves exactly as before."""
    in_csv = PROJECT_ROOT / "data" / (in_csv_name or f"{case_name}_inputs.csv")
    df = pd.read_csv(in_csv)
    netlist = build_netlist(case_name, df, model_file)
    net_path = PROJECT_ROOT / "data" / f"_generated_{case_name}{net_suffix or ''}.net"
    net_path.write_text(netlist)

    t0 = time.time()
    result = subprocess.run([NGSPICE_BIN, "-b", str(net_path)], capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"ngspice failed:\n{result.stderr[-3000:]}")

    matches = PRINT_RE.findall(result.stdout)
    if len(matches) != len(df):
        raise RuntimeError(f"Parsed {len(matches)} but expected {len(df)} -- check convergence in stderr:\n{result.stderr[-2000:]}")

    currents = pd.DataFrame(matches, columns=["id", "ig", "is", "ib"]).astype(float)
    out_df = pd.concat([df.reset_index(drop=True), currents], axis=1)
    out_df = out_df[["vdd","temp","w","vd","vs","vg","vb","vth0","toxe","ndep","u0","xj","id","ig","is","ib"]]

    out_path = PROJECT_ROOT / "data" / (out_csv_name or f"{case_name}_dataset.csv")
    out_df.to_csv(out_path, index=False)

    kcl = (out_df["id"]+out_df["ig"]+out_df["is"]+out_df["ib"]).abs()
    print(f"[{case_name}] {len(out_df)} rows in {elapsed:.1f}s")
    print(f"[{case_name}] KCL residual median={kcl.median():.3e} A (should be near-zero vs mean|Id|={out_df['id'].abs().mean():.3e} A)")
    print(f"[{case_name}] Wrote {out_path}")
    return out_df

if __name__ == "__main__":
    case_name = sys.argv[1] if len(sys.argv) > 1 else "NMOS_ON"
    run_case(case_name)
