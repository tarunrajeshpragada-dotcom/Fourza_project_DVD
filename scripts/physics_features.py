"""
physics_features.py
----------------------
Features and target shared by train_final.py and gates/gates_varw.py, so
training and inference can never build them differently.

Why physics-based features: OFF transistors in gates sit at |Vds| close to
VDD, where uniform sampling has very few points, so the model must
EXTRAPOLATE to the rail. The textbook subthreshold expression
    I ~ W * (1 - exp(-Vds/Vt)) * exp((Vgs - Vth) / (n Vt)),   Vt = kT/q
becomes nearly linear in the inputs below once the target is log(I / W), so a
neural network can extrapolate smoothly. (Tree models cannot extrapolate
beyond their training data, which validation in the high-|Vds| region shows.)

Features (15): Vgs/Vt, Vds/Vt, Vbs/Vt, ln(1 - exp(-|Vds|/Vt)), 1/Vt,
|vth0|/Vt, ln(W/Lmin), VDD, Vgs, Vds, Vbs, toxe, ndep, u0, xj.
The plain voltages are kept because gate tunnelling depends on the oxide
voltage itself, not on Vt.

Target: symlog of the current per unit width, I * Lmin / W; predictions are
multiplied back by W / Lmin.
"""
import numpy as np

from common import L_MIN
from ml_utils import from_symlog, to_symlog

K_OVER_Q = 8.617333e-5  # Boltzmann constant / electron charge, V/K


def build_physics_features(df):
    vt = K_OVER_Q * (df["temp"].values + 273.15)
    vgs = df["vg"].values - df["vs"].values
    vds = df["vd"].values - df["vs"].values
    vbs = df["vb"].values - df["vs"].values
    return np.column_stack([
        vgs / vt, vds / vt, vbs / vt,
        np.log(np.clip(1.0 - np.exp(-np.abs(vds) / vt), 1e-9, None)),
        1.0 / vt, np.abs(df["vth0"].values) / vt, np.log(df["w"].values / L_MIN),
        df["vdd"].values, vgs, vds, vbs,
        df["toxe"].values, df["ndep"].values, df["u0"].values, df["xj"].values,
    ])


def target(df, col):
    """Training target for current column `col`: symlog of current per unit width."""
    return to_symlog(df[col].values * L_MIN / df["w"].values)


def inverse_target(y_pred, w):
    """Model output -> current in amps for width(s) w."""
    return from_symlog(y_pred) * np.asarray(w) / L_MIN
