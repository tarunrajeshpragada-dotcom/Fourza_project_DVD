"""
05_generate_corners.py
------------------------
Part B, Step 3 prerequisite: generate the 200 random "corners" shared across
every gate simulation. Per Section 5 of the spec:
  "A corner is one random draw of VDD, temperature and process parameters,
   using the distributions in Section 4. All NMOS in a gate share one draw
   and all PMOS share another."

So a corner is NOT a full Part-A-style transistor sample -- no W (fixed at
Wmin for gates, per the TA's own worked example), no Vd/Vs (those come from
solving the actual gate circuit, not independent sampling). Just:
  vdd, temp, nmos_{vth0,toxe,ndep,u0,xj}, pmos_{vth0,toxe,ndep,u0,xj}

A SEPARATE, FIXED SEED is used here (not the same RANDOM_SEED as Part A),
so the 200 corners are a distinct draw from the transistor-level dataset,
consistent with the spec's "These results are for evaluation only and must
not be used for training."
"""
import numpy as np
import pandas as pd

from common import NOMINAL, VDD_NOMINAL, PCT_VAR, PROJECT_ROOT

N_CORNERS = 200
CORNER_SEED = 1042  # ASSUMPTION: distinct from Part A's RANDOM_SEED=42, by design (see docstring)


def pvar(rng, center, size):
    a, b = center * (1 - PCT_VAR), center * (1 + PCT_VAR)
    return rng.uniform(min(a, b), max(a, b), size)


def generate_corners():
    rng = np.random.default_rng(CORNER_SEED)
    vdd = rng.uniform(VDD_NOMINAL * (1 - PCT_VAR), VDD_NOMINAL * (1 + PCT_VAR), N_CORNERS)
    temp = rng.uniform(-40.0, 125.0, N_CORNERS)

    nmos = NOMINAL["nmos"]
    pmos = NOMINAL["pmos"]
    df = pd.DataFrame({
        "corner": np.arange(N_CORNERS),
        "vdd": vdd,
        "temp": temp,
        "nmos_vth0": pvar(rng, nmos["vth0"], N_CORNERS),
        "nmos_toxe": pvar(rng, nmos["toxe"], N_CORNERS),
        "nmos_ndep": pvar(rng, nmos["ndep"], N_CORNERS),
        "nmos_u0": pvar(rng, nmos["u0"], N_CORNERS),
        "nmos_xj": pvar(rng, nmos["xj"], N_CORNERS),
        "pmos_vth0": pvar(rng, pmos["vth0"], N_CORNERS),
        "pmos_toxe": pvar(rng, pmos["toxe"], N_CORNERS),
        "pmos_ndep": pvar(rng, pmos["ndep"], N_CORNERS),
        "pmos_u0": pvar(rng, pmos["u0"], N_CORNERS),
        "pmos_xj": pvar(rng, pmos["xj"], N_CORNERS),
    })
    return df


if __name__ == "__main__":
    df = generate_corners()
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "corners_200.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} corners (seed={CORNER_SEED})")
    print(df.describe().T[["min", "max", "mean"]])
    print(f"Wrote {out_path}")
