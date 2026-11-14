"""
09_generate_gatecond_inputs.py
---------------------------------
Second targeted augmentation, for the two OFF cases only (the only models
used to compute gate leakage). Diagnosis that motivated it: simulating a
single transistor at the exact terminal voltages the OFF device has inside a
gate reproduces the gate's SPICE leakage EXACTLY (ratio 1.000), so the
remaining gate error is entirely the transistor model at that operating
point. Every gate transistor is minimum width (Wmin), but W was drawn
uniformly over 1x-8x Lmin, so only ~7% of training samples were near Wmin.

These are ordinary single-transistor SPICE samples (Part A style), drawn
from physically motivated ranges, NOT copied from any gate/circuit SPICE
result, so the evaluation data stays untouched. Gate and body voltages stay
exactly at the rails defined by each case (spec Section 4).

  W     : Uniform(1.0, 1.2) x Lmin
  temp  : full -40..125 C
  NMOS_OFF: Vs = 0 (half) or Uniform(0, 0.3*VDD) (stacked-device source);
            Vd = Uniform(0.3, 1.0)*VDD
  PMOS_OFF: Vs = VDD (half) or Uniform(0.7, 1.0)*VDD;  Vd = Uniform(0, 0.7)*VDD
"""
import sys
import numpy as np
import pandas as pd
from common import NOMINAL, VDD_NOMINAL, L_MIN, PCT_VAR, PROJECT_ROOT

N = 4000
SEED = 2468


def pvar(rng, c, n):
    a, b = c * (1 - PCT_VAR), c * (1 + PCT_VAR)
    return rng.uniform(min(a, b), max(a, b), n)


def generate(case):
    if case not in ("NMOS_OFF", "PMOS_OFF"):
        raise ValueError("only NMOS_OFF / PMOS_OFF")
    rng = np.random.default_rng(SEED + (0 if case == "NMOS_OFF" else 1))
    dev = "nmos" if case == "NMOS_OFF" else "pmos"
    nom = NOMINAL[dev]
    vdd = rng.uniform(VDD_NOMINAL * (1 - PCT_VAR), VDD_NOMINAL * (1 + PCT_VAR), N)
    temp = rng.uniform(-40, 125, N)
    w = rng.uniform(1.0 * L_MIN, 1.2 * L_MIN, N)
    half = rng.random(N) < 0.5
    if case == "NMOS_OFF":
        vs = np.where(half, 0.0, rng.uniform(0, 0.3, N) * vdd)
        vd = rng.uniform(0.3, 1.0, N) * vdd
        vg = np.zeros(N); vb = np.zeros(N)
    else:
        vs = np.where(half, vdd, rng.uniform(0.7, 1.0, N) * vdd)
        vd = rng.uniform(0.0, 0.7, N) * vdd
        vg = vdd.copy(); vb = vdd.copy()
    return pd.DataFrame(dict(vdd=vdd, temp=temp, w=w, vd=vd, vs=vs, vg=vg, vb=vb,
                             vth0=pvar(rng, nom["vth0"], N), toxe=pvar(rng, nom["toxe"], N),
                             ndep=pvar(rng, nom["ndep"], N), u0=pvar(rng, nom["u0"], N),
                             xj=pvar(rng, nom["xj"], N)))


if __name__ == "__main__":
    case = sys.argv[1]
    df = generate(case)
    out = PROJECT_ROOT / "data" / f"{case}_gatecond_inputs.csv"
    df.to_csv(out, index=False)
    print(f"[{case}] {len(df)} gate-condition samples -> {out}")
