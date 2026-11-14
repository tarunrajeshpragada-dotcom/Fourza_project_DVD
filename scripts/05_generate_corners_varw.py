"""
05_generate_corners_varw.py
------------------------------
The 200 PVT corners shared by every gate and the parity checker, now with
transistor WIDTH varied per corner as the TA's board lists it (W = 1x to 8x
Lmin). Following the spec's "All NMOS in a gate share one draw and all PMOS
share another", each corner gets ONE NMOS width (wn) and ONE PMOS width (wp).

Same seed and same draw order as 05_generate_corners.py, so vdd, temp and all
ten process parameters are IDENTICAL to the earlier corners; wn and wp are
drawn after them. Overwrites data/corners_200.csv (adds columns wn, wp).

Usage:  python3 05_generate_corners_varw.py
"""
import numpy as np
import pandas as pd

from common import L_MIN, NOMINAL, PCT_VAR, PROJECT_ROOT, VDD_NOMINAL

N_CORNERS = 200
CORNER_SEED = 1042


def pvar(rng, center, size):
    a, b = center * (1 - PCT_VAR), center * (1 + PCT_VAR)
    return rng.uniform(min(a, b), max(a, b), size)


def generate_corners():
    rng = np.random.default_rng(CORNER_SEED)
    vdd = rng.uniform(VDD_NOMINAL * (1 - PCT_VAR), VDD_NOMINAL * (1 + PCT_VAR), N_CORNERS)
    temp = rng.uniform(-40.0, 125.0, N_CORNERS)
    nmos, pmos = NOMINAL["nmos"], NOMINAL["pmos"]
    cols = {"corner": np.arange(N_CORNERS), "vdd": vdd, "temp": temp}
    for dev, nom in [("nmos", nmos), ("pmos", pmos)]:
        for p in ["vth0", "toxe", "ndep", "u0", "xj"]:
            cols[f"{dev}_{p}"] = pvar(rng, nom[p], N_CORNERS)
    cols["wn"] = rng.uniform(1 * L_MIN, 8 * L_MIN, N_CORNERS)   # drawn last: earlier columns unchanged
    cols["wp"] = rng.uniform(1 * L_MIN, 8 * L_MIN, N_CORNERS)
    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = generate_corners()
    out = PROJECT_ROOT / "data" / "corners_200.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} corners (seed={CORNER_SEED}) with per-corner widths")
    print(df[["vdd", "temp", "wn", "wp"]].describe().T[["min", "max", "mean"]])
    print(f"Wrote {out}")
