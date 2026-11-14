"""
gates_varw.py
---------------
Part B for all five gates (INV, NAND2, NOR2, AND2, OR2) with transistor WIDTH
varied per corner (data/corners_200.csv columns wn, wp: every NMOS in a gate
gets wn, every PMOS gets wp, AD = AS = W x Ldiff).

  python3 gates_varw.py spice   -> data/<GATE>_gate_spice.csv       (SPICE reference)
  python3 gates_varw.py ml      -> data/<GATE>_gate_comparison.csv  (ML prediction)

Then run 06_consolidate_gate_results.py as before.

SPICE side (Part B step 3): one netlist per gate, looping over all 200 corners
and all input combinations with ngspice 'alter'. Records the output and
internal node voltages, the VDD supply current (spice_leakage, as before), and
the current delivered by inputs held at logic 1 (spice_input_leak), so the
two possible leakage definitions can be compared.

ML side (Part B step 4): uses the SPICE-recorded node voltages (as the spec
prescribes) to set each transistor's terminal voltages, picks the ON or OFF
model from its gate voltage, and combines the predicted drain currents with
the same rule as before:
  - series stack blocking (any device OFF): the OFF device's current
    (average if both OFF);
  - parallel bank blocking (all OFF): sum of the OFF devices' currents;
  - AND2 / OR2: first stage + INV stage (two separate supply paths).
Models come from scripts/train_final.py (physics-based features, uniform
data only): the OFF-case drain-current models are the averaged models in
models/final/ensemble/, the ON-case ones the single models in models/final/.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import L_DIFF, L_MIN, MODEL_FILE, NGSPICE_BIN, PROJECT_ROOT
from physics_features import build_physics_features, inverse_target

DATA = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "final"            # written by scripts/train_final.py
ENSEMBLE_DIR = PROJECT_ROOT / "models" / "final" / "ensemble"

# device: (name, type, drain, gate, source, body)
GATES = {
    "INV": dict(devices=[("mp", "pmos", "out", "a", "vdd", "vdd"), ("mn", "nmos", "out", "a", "0", "0")],
                inputs=["a"], nodes=["out"]),
    "NAND2": dict(devices=[("m1", "nmos", "out", "a", "x", "0"), ("m2", "nmos", "x", "b", "0", "0"),
                           ("m3", "pmos", "out", "a", "vdd", "vdd"), ("m4", "pmos", "out", "b", "vdd", "vdd")],
                  inputs=["a", "b"], nodes=["out", "x"]),
    "NOR2": dict(devices=[("m1", "pmos", "out", "a", "y", "vdd"), ("m2", "pmos", "y", "b", "vdd", "vdd"),
                          ("m3", "nmos", "out", "a", "0", "0"), ("m4", "nmos", "out", "b", "0", "0")],
                 inputs=["a", "b"], nodes=["out", "y"]),
    "AND2": dict(devices=[("m1", "nmos", "n1", "a", "x", "0"), ("m2", "nmos", "x", "b", "0", "0"),
                          ("m3", "pmos", "n1", "a", "vdd", "vdd"), ("m4", "pmos", "n1", "b", "vdd", "vdd"),
                          ("m5", "pmos", "out", "n1", "vdd", "vdd"), ("m6", "nmos", "out", "n1", "0", "0")],
                 inputs=["a", "b"], nodes=["out", "n1", "x"]),
    "OR2": dict(devices=[("m1", "pmos", "n1", "a", "y", "vdd"), ("m2", "pmos", "y", "b", "vdd", "vdd"),
                         ("m3", "nmos", "n1", "a", "0", "0"), ("m4", "nmos", "n1", "b", "0", "0"),
                         ("m5", "pmos", "out", "n1", "vdd", "vdd"), ("m6", "nmos", "out", "n1", "0", "0")],
                inputs=["a", "b"], nodes=["out", "n1", "y"]),
}
PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]


def input_states(gate):
    if len(GATES[gate]["inputs"]) == 1:
        return [("A0", (0,)), ("A1", (1,))]
    return [("00", (0, 0)), ("01", (0, 1)), ("10", (1, 0)), ("11", (1, 1))]


# ----------------------------------------------------------------- SPICE side
def build_netlist(gate, corners):
    g = GATES[gate]
    lines = [f"* {gate} gate, widths varied per corner", f".INCLUDE {MODEL_FILE}",
             ".OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9", ""]
    ad0 = L_MIN * L_DIFF
    for name, typ, d, gg, s, b in g["devices"]:
        lines.append(f"{name} {d} {gg} {s} {b} {typ} W={L_MIN:.6g} L={L_MIN:.6g} AD={ad0:.6g} AS={ad0:.6g}")
    lines.append("Vdd_src vdd 0 0")
    for inp in g["inputs"]:
        lines.append(f"V{inp} {inp} 0 0")
    lines += ["", ".control"]
    pv = " ".join(f"v({n})" for n in g["nodes"]) + " i(vdd_src) " + " ".join(f"i(v{i})" for i in g["inputs"])
    for r in corners.itertuples(index=False):
        for _, bits in input_states(gate):
            # one altermod line PER PARAMETER: ngspice applies only the first parameter of a multi-parameter line
            for dev in ["nmos", "pmos"]:
                lines += [f"altermod @{dev}[{p}]={getattr(r, f'{dev}_{p}'):.8g}" for p in PARAMS]
            for name, typ, *_ in g["devices"]:
                w = r.wn if typ == "nmos" else r.wp
                lines += [f"alter @{name}[w] = {w:.8g}", f"alter @{name}[ad] = {w * L_DIFF:.8g}",
                          f"alter @{name}[as] = {w * L_DIFF:.8g}"]
            lines.append(f"alter Vdd_src = {r.vdd:.8g}")
            for inp, bit in zip(g["inputs"], bits):
                lines.append(f"alter V{inp} = {r.vdd * bit:.8g}")
            lines += [f"option temp = {r.temp:.8g}", "op", f"print {pv}"]
    lines += [".endc", ".end"]
    return "\n".join(lines), pv


def run_spice(gate, corners):
    g = GATES[gate]
    netlist, pv = build_netlist(gate, corners)
    path = Path(__file__).resolve().parent / f"_generated_{gate}_varw.net"
    path.write_text(netlist)
    t0 = time.time()
    res = subprocess.run([NGSPICE_BIN, "-b", str(path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{gate}: ngspice failed\n{res.stderr[-2000:]}")
    names = [t.lower() for t in pv.split()]
    rx = re.compile(r"\s*\n".join(rf"{re.escape(n)}\s*=\s*([\-\d\.eE\+]+)" for n in names))
    matches = rx.findall(res.stdout)
    states = input_states(gate)
    if len(matches) != len(corners) * len(states):
        raise RuntimeError(f"{gate}: parsed {len(matches)} of {len(corners) * len(states)}\n{res.stderr[-2000:]}")
    rows, k = [], 0
    for r in corners.itertuples(index=False):
        for label, bits in states:
            vals = [float(v) for v in matches[k]]; k += 1
            nv = dict(zip(g["nodes"], vals[:len(g["nodes"])]))
            i_vdd = vals[len(g["nodes"])]
            i_in = vals[len(g["nodes"]) + 1:]
            row = dict(corner=r.corner, input=label, vdd=r.vdd, temp=r.temp, wn=r.wn, wp=r.wp)
            for inp, bit in zip(g["inputs"], bits):
                row[inp.upper()] = r.vdd * bit
            for dev in ["nmos", "pmos"]:
                for p in PARAMS:
                    row[f"{dev}_{p}"] = getattr(r, f"{dev}_{p}")
            for n, v in nv.items():
                row[f"v_{n}"] = v
            row["spice_leakage"] = abs(i_vdd)                                   # VDD supply current
            row["spice_input_leak"] = sum(-i for i, bit in zip(i_in, bits) if bit)  # delivered by inputs at 1
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"{gate}_gate_spice.csv", index=False)
    path.unlink()
    frac = (out.spice_input_leak / out.spice_leakage).median() * 100
    print(f"{gate}: {len(out)} points in {time.time() - t0:.1f}s | input current / supply current (median) = {frac:.3f}%")
    for label, _ in states:
        s = out[out.input == label]
        print(f"   {label}: v_out median {s.v_out.median():.4f} V   supply leakage median {s.spice_leakage.median():.3e} A")


# -------------------------------------------------------------------- ML side
_cache = {}


class AveragedModel:
    def __init__(self, members):
        self.members = members

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


def load_id_model(case):
    if case not in _cache:
        members = sorted(ENSEMBLE_DIR.glob(f"{case}_id_*_m*.joblib")) if ENSEMBLE_DIR.exists() else []
        if members:
            _cache[case] = AveragedModel([joblib.load(f) for f in members])
        else:
            _cache[case] = joblib.load(list(MODEL_DIR.glob(f"{case}_id_*.joblib"))[0])
    return _cache[case]


def predict_id(df, dev, on_mask, vd, vs, vg, vb):
    """Predicted |Id| for one transistor position across all rows (ON rows -> ON model, OFF -> OFF model)."""
    n = len(df)
    w = df.wn.values if dev == "nmos" else df.wp.values
    feats = pd.DataFrame(dict(vdd=df.vdd.values, temp=df.temp.values, w=w,
                              vd=np.broadcast_to(vd, n), vs=np.broadcast_to(vs, n),
                              vg=np.broadcast_to(vg, n), vb=np.broadcast_to(vb, n),
                              **{p: df[f"{dev}_{p}"].values for p in PARAMS}))
    out = np.zeros(n)
    for state, mask in [("ON", on_mask), ("OFF", ~on_mask)]:
        if mask.any():
            X = build_physics_features(feats[mask])
            out[mask] = np.abs(inverse_target(load_id_model(f"{dev.upper()}_{state}").predict(X), feats["w"].values[mask]))
    return out


def stage_leakage(df, kind, a, b, out, mid):
    """One NAND2 ('nand') or NOR2 ('nor') stage; a, b = gate input voltages; out, mid = node voltages."""
    vdd, z = df.vdd.values, np.zeros(len(df))
    a_hi, b_hi = a > vdd / 2, b > vdd / 2
    if kind == "nand":
        i1 = predict_id(df, "nmos", a_hi, out, mid, a, z)     # m1: out-a-x
        i2 = predict_id(df, "nmos", b_hi, mid, z, b, z)       # m2: x-b-0
        i3 = predict_id(df, "pmos", ~a_hi, out, vdd, a, vdd)  # m3: parallel pull-up
        i4 = predict_id(df, "pmos", ~b_hi, out, vdd, b, vdd)  # m4
        s_off1, s_off2 = ~a_hi, ~b_hi                         # series devices OFF?
        series_blocks = s_off1 | s_off2
    else:
        i1 = predict_id(df, "pmos", ~a_hi, out, mid, a, vdd)  # m1: out-a-y
        i2 = predict_id(df, "pmos", ~b_hi, mid, vdd, b, vdd)  # m2: y-b-vdd
        i3 = predict_id(df, "nmos", a_hi, out, z, a, z)       # m3: parallel pull-down
        i4 = predict_id(df, "nmos", b_hi, out, z, b, z)       # m4
        s_off1, s_off2 = a_hi, b_hi
        series_blocks = s_off1 | s_off2
    series = np.where(s_off1 & s_off2, (i1 + i2) / 2, np.where(s_off1, i1, i2))
    return np.where(series_blocks, series, i3 + i4)


def inv_leakage(df, a, out):
    vdd, z = df.vdd.values, np.zeros(len(df))
    a_hi = a > vdd / 2
    i_n = predict_id(df, "nmos", a_hi, out, z, a, z)
    i_p = predict_id(df, "pmos", ~a_hi, out, vdd, a, vdd)
    return np.where(a_hi, i_p, i_n)


def run_ml(gate):
    df = pd.read_csv(DATA / f"{gate}_gate_spice.csv", dtype={"input": str})
    a = df.A.values
    b = df.B.values if "B" in df else None
    if gate == "INV":
        ml = inv_leakage(df, a, df.v_out.values)
    elif gate in ("NAND2", "NOR2"):
        mid = df.v_x.values if gate == "NAND2" else df.v_y.values
        ml = stage_leakage(df, "nand" if gate == "NAND2" else "nor", a, b, df.v_out.values, mid)
    else:
        mid = df.v_x.values if gate == "AND2" else df.v_y.values
        ml = (stage_leakage(df, "nand" if gate == "AND2" else "nor", a, b, df.v_n1.values, mid)
              + inv_leakage(df, df.v_n1.values, df.v_out.values))
    out = df.copy()
    out["ml_leakage"] = ml
    out.to_csv(DATA / f"{gate}_gate_comparison.csv", index=False)
    e = np.abs(out.ml_leakage - out.spice_leakage) / out.spice_leakage
    print(f"{gate}: " + "  ".join(f"{k}: med {v.median():.3f} mean {v.mean():.3f}" for k, v in out.assign(e=e).groupby("input")["e"]))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "spice":
        corners = pd.read_csv(DATA / "corners_200.csv")
        if "wn" not in corners:
            sys.exit("corners_200.csv has no wn/wp columns -- run scripts/05_generate_corners_varw.py first")
        for gate in GATES:
            run_spice(gate, corners)
    elif mode == "ml":
        for gate in GATES:
            run_ml(gate)
    else:
        sys.exit("usage: python3 gates_varw.py spice | ml")
