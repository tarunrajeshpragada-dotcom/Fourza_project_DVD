"""
common.py
---------
Shared configuration for the ML-Driven PVT-Variation-Aware Leakage
Estimation project (Digital VLSI Design course).

Every numeric value below that is NOT the literal 10% tolerance or
sample count is sourced from one of two places, and is labelled as such:
  (A) the team assignment table (Tech Node / Voltage / Testing Circuit), or
  (F) the assigned model card models/16nm_HP.pm (read directly, not
      reproduced from memory).

ASSUMPTION (design decision, not dictated by the spec): RANDOM_SEED.
The spec fixes the seed requirement only for the train/val/test split;
we additionally fix a seed for sample *generation* for full
reproducibility of the raw dataset itself. Change RANDOM_SEED to
regenerate a different draw.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- (A) Assigned configuration -------------------------------------------------
# Source: team assignment table screenshot.
#   Tech Node       : 16nm_HP
#   Nominal VDD     : 1.0 V          <-- CONFIRM: this is read from an image;
#                                         double check against your TA sheet.
#   Test circuit    : 8-bit Parity Checker  (used in Part C, not this step)
MODEL_FILE  = PROJECT_ROOT / "models" / "16nm_HP.pm"
VDD_NOMINAL = 1.0           # V
L_MIN       = 16e-9         # m  (feature size of the 16nm_HP node; L is NOT swept)

# ASSUMPTION: drain/source diffusion length, used for AD=AS=W*L_DIFF (TA guidance).
# The course's own 22nm example netlist sets Ldiff=44n = 2*22n; extrapolating that
# same Ldiff=2*Lmin convention to this node gives 32n. CONFIRM with your TA if they
# gave an explicit 16nm value instead of expecting this derivation.
L_DIFF      = 2 * L_MIN     # m

# ---- (F) Nominal NMOS / PMOS parameters, read verbatim from models/16nm_HP.pm ----
# nmos block (lines 18, 11, 23, 26, 22 of the model card):
#   vth0 = 0.47965   toxe = 9.5e-10   ndep = 7e18   u0 = 0.03   xj = 5e-9
# pmos block (lines 87, 80, 92, 95, 91 of the model card):
#   vth0 = -0.43121  toxe = 1.0e-9    ndep = 5.5e18 u0 = 0.006  xj = 5e-9
NOMINAL = {
    "nmos": dict(vth0=0.47965,  toxe=9.5e-10, ndep=7.0e18, u0=0.03,  xj=5e-9),
    "pmos": dict(vth0=-0.43121, toxe=1.0e-9,  ndep=5.5e18, u0=0.006, xj=5e-9),
}

# ---- Case table (Section 4: "the gate and body voltages are fixed by the case") --
# model : the .model name in the card to instantiate / alter ("nmos" or "pmos")
# gate  : "VDD" or "0"  -- resolved per-sample to that sample's own vdd draw
# body  : "VDD" or "0"
CASES = {
    "NMOS_ON":  dict(model="nmos", gate="VDD", body="0"),
    "NMOS_OFF": dict(model="nmos", gate="0",   body="0"),
    "PMOS_ON":  dict(model="pmos", gate="0",   body="VDD"),
    "PMOS_OFF": dict(model="pmos", gate="VDD", body="VDD"),
}

# ---- Sampling constants -----------------------------------------------------------
N_SAMPLES   = 10_000
RANDOM_SEED = 42      # ASSUMPTION -- see module docstring
PCT_VAR     = 0.10     # +/-10%, applies to VDD and to all five process parameters

NGSPICE_BIN = "ngspice"
