"""
07_generate_augmented_inputs.py
----------------------------------
Targeted re-sampling (accuracy-improvement Option 1), attacking the two
specific, diagnosed weaknesses rather than drawing more generic uniform
samples and hoping some land in the sparse region:

  - NMOS_ON, PMOS_ON: near-zero |Vds| is where an ON device sits in a real
    gate, and is severely underrepresented in the original data (<1% of
    samples have |Vds|<0.01V, because independent Uniform(0,VDD) draws for
    Vd and Vs produce a triangular |Vds| distribution that is sparsest near
    its own peak-information region). New samples here force Vd close to
    Vs: Vs ~ Uniform(0,VDD) as before, then Vd = clip(Vs + delta, 0, VDD)
    with delta ~ Uniform(-0.05*VDD, +0.05*VDD).

  - NMOS_OFF, PMOS_OFF: the compound high-Vds x high-temperature corner
    (individually common conditions, rarely sampled together) is where
    PMOS_OFF specifically broke down. New samples here force a large |Vds|
    split (one terminal in the bottom 5% of its range, the other in the top
    5%, with the assignment randomized so both current directions are
    covered -- preserving the sign-convention correctness checked
    throughout this project) AND force temp into the upper end of its
    range (90-125 C, vs. the full -40 to 125 C range used originally).

Every OTHER parameter (vdd, w, vth0, toxe, ndep, u0, xj) is drawn with the
EXACT SAME distribution as 01_generate_inputs.py -- only the targeted
variable(s) change, so this augmentation adds coverage without shifting
any other part of the sampled space.

A separate seed (AUG_SEED) is used, distinct from both Part A's RANDOM_SEED
and Part B's corner CORNER_SEED.

Usage:
    python3 07_generate_augmented_inputs.py NMOS_ON
Output:
    data/<CASE>_aug_inputs.csv   (N_AUG rows, same 11 columns as the original)
"""
import sys

import numpy as np
import pandas as pd

from common import CASES, NOMINAL, VDD_NOMINAL, L_MIN, PCT_VAR, PROJECT_ROOT

N_AUG = 3000
AUG_SEED = 777  # ASSUMPTION: a new, distinct seed -- design decision, not spec-mandated
NEAR_ZERO_VDS_BAND = 0.05   # ON cases: |Vds| forced into +/-5% of VDD around 0
HIGH_VDS_EDGE = 0.05        # OFF cases: each terminal forced into the bottom/top 5% of [0,VDD]
HIGH_TEMP_LO = 90.0         # OFF cases: temp forced into [90, 125] instead of [-40, 125]


def pvar(rng, center, size):
    a, b = center * (1 - PCT_VAR), center * (1 + PCT_VAR)
    return rng.uniform(min(a, b), max(a, b), size)


def generate_augmented(case_name: str) -> pd.DataFrame:
    if case_name not in CASES:
        raise ValueError(f"Unknown case '{case_name}'. Choices: {list(CASES)}")
    case = CASES[case_name]
    dev = case["model"]
    nom = NOMINAL[dev]
    is_on_case = case_name.endswith("_ON")

    rng = np.random.default_rng(AUG_SEED + (0 if is_on_case else 1) + (0 if dev == "nmos" else 2))

    vdd = rng.uniform(VDD_NOMINAL * (1 - PCT_VAR), VDD_NOMINAL * (1 + PCT_VAR), N_AUG)
    w = rng.uniform(1 * L_MIN, 8 * L_MIN, N_AUG)

    if is_on_case:
        # Target: near-zero |Vds|, full temperature range as before.
        temp = rng.uniform(-40.0, 125.0, N_AUG)
        vs = rng.uniform(0.0, vdd)
        delta = rng.uniform(-NEAR_ZERO_VDS_BAND * vdd, NEAR_ZERO_VDS_BAND * vdd)
        vd = np.clip(vs + delta, 0.0, vdd)
    else:
        # Target: large |Vds| (one terminal near 0, the other near VDD,
        # direction randomized) AND high temperature, simultaneously.
        temp = rng.uniform(HIGH_TEMP_LO, 125.0, N_AUG)
        direction = rng.integers(0, 2, N_AUG)  # 0: Vd low/Vs high, 1: Vd high/Vs low
        low = rng.uniform(0.0, HIGH_VDS_EDGE * vdd, N_AUG)
        high = rng.uniform((1 - HIGH_VDS_EDGE) * vdd, vdd, N_AUG)
        vd = np.where(direction == 0, low, high)
        vs = np.where(direction == 0, high, low)

    vth0 = pvar(rng, nom["vth0"], N_AUG)
    toxe = pvar(rng, nom["toxe"], N_AUG)
    ndep = pvar(rng, nom["ndep"], N_AUG)
    u0 = pvar(rng, nom["u0"], N_AUG)
    xj = pvar(rng, nom["xj"], N_AUG)

    vg = vdd.copy() if case["gate"] == "VDD" else np.zeros(N_AUG)
    vb = vdd.copy() if case["body"] == "VDD" else np.zeros(N_AUG)

    return pd.DataFrame({
        "vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs, "vg": vg, "vb": vb,
        "vth0": vth0, "toxe": toxe, "ndep": ndep, "u0": u0, "xj": xj,
    })


if __name__ == "__main__":
    case_name = sys.argv[1] if len(sys.argv) > 1 else "NMOS_ON"
    df = generate_augmented(case_name)

    vds = (df["vd"] - df["vs"]).abs()
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"{case_name}_aug_inputs.csv"
    df.to_csv(out_path, index=False)

    print(f"[{case_name}] drew {len(df)} TARGETED augmentation samples (seed base={AUG_SEED})")
    print(f"  |Vds| range: {vds.min():.4f} to {vds.max():.4f} (vdd ~{df['vdd'].mean():.3f})")
    print(f"  temp range: {df['temp'].min():.2f} to {df['temp'].max():.2f}")
    print(f"Wrote {out_path}")
