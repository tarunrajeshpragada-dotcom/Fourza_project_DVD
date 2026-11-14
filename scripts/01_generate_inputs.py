"""
01_generate_inputs.py
----------------------
Draws the N_SAMPLES random PVT input vectors for ONE transistor case
(NMOS_ON, NMOS_OFF, PMOS_ON or PMOS_OFF), exactly per Section 4 of the
project spec:

  - VDD drawn first: Uniform(VDD_nominal * 0.9, VDD_nominal * 1.1)
  - Vd, Vs:          Uniform(0, VDD)            [depend on that sample's VDD]
  - W:               Uniform(1*Lmin, 8*Lmin)
  - temp:            Uniform(-40, 125) degC
  - vth0,toxe,ndep,u0,xj : Uniform(nominal*0.9, nominal*1.1), nominal taken
                            from the model card for whichever device
                            (nmos/pmos) this case instantiates
  - gate, body:      fixed by case (see common.CASES), not independently drawn

All draws within a sample are independent of one another except for the
stated Vd/Vs <- VDD dependency.

Usage:
    python3 01_generate_inputs.py NMOS_ON
Output:
    data/<CASE>_inputs.csv   (10,000 rows x 11 columns: vdd,temp,w,vd,vs,
                               vg,vb,vth0,toxe,ndep,u0,xj)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import CASES, NOMINAL, VDD_NOMINAL, L_MIN, N_SAMPLES, RANDOM_SEED, PCT_VAR, PROJECT_ROOT


def generate(case_name: str) -> pd.DataFrame:
    if case_name not in CASES:
        raise ValueError(f"Unknown case '{case_name}'. Choices: {list(CASES)}")
    case = CASES[case_name]
    dev = case["model"]            # "nmos" or "pmos"
    nom = NOMINAL[dev]

    rng = np.random.default_rng(RANDOM_SEED)

    # 1. VDD first -- everything voltage-related below depends on it.
    vdd = rng.uniform(VDD_NOMINAL * (1 - PCT_VAR), VDD_NOMINAL * (1 + PCT_VAR), N_SAMPLES)

    # 2. Terminal voltages whose range depends on vdd (independent draws).
    vd = rng.uniform(0.0, vdd)
    vs = rng.uniform(0.0, vdd)

    # 3. Geometry and environment (independent of vdd).
    w = rng.uniform(1 * L_MIN, 8 * L_MIN, N_SAMPLES)
    temp = rng.uniform(-40.0, 125.0, N_SAMPLES)

    # 4. Process parameters, independent +/-10% draws around the model-card nominal.
    # NOTE: for a negative center (PMOS vth0 = -0.43121), center*(1-PCT_VAR) is
    # actually the LARGER value and center*(1+PCT_VAR) the smaller one (e.g.
    # -0.431*0.9=-0.388 vs -0.431*1.1=-0.474), which breaks rng.uniform(low,high)
    # if passed in the "positive-center" order. Taking min/max of the two
    # candidate bounds keeps the result correct (a true +/-10% band around
    # center) regardless of its sign.
    def pvar(center: float) -> np.ndarray:
        a, b = center * (1 - PCT_VAR), center * (1 + PCT_VAR)
        return rng.uniform(min(a, b), max(a, b), N_SAMPLES)

    vth0 = pvar(nom["vth0"])
    toxe = pvar(nom["toxe"])
    ndep = pvar(nom["ndep"])
    u0 = pvar(nom["u0"])
    xj = pvar(nom["xj"])

    # 5. Gate / body fixed by case definition (NOT independently sampled).
    vg = vdd.copy() if case["gate"] == "VDD" else np.zeros(N_SAMPLES)
    vb = vdd.copy() if case["body"] == "VDD" else np.zeros(N_SAMPLES)

    df = pd.DataFrame(
        {
            "vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs,
            "vg": vg, "vb": vb,
            "vth0": vth0, "toxe": toxe, "ndep": ndep, "u0": u0, "xj": xj,
        }
    )
    return df


if __name__ == "__main__":
    case_name = sys.argv[1] if len(sys.argv) > 1 else "NMOS_ON"
    df = generate(case_name)

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"{case_name}_inputs.csv"
    df.to_csv(out_path, index=False)

    print(f"[{case_name}] drew {len(df)} samples (seed={RANDOM_SEED})")
    print(df.describe().T[["min", "max", "mean"]])
    print(f"Wrote {out_path}")
