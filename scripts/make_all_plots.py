"""
make_all_plots.py
--------------------
Every essential report plot, regenerated from the project's CSV files alone
(no .joblib models are loaded, so it runs identically on the laptop and in
Google Colab regardless of scikit-learn version).

WHERE IT LOOKS FOR DATA
  - On the laptop:  ~/dvd_project/data   (run from anywhere)
  - In Colab:       /content/dvd_project/data
  Or pass a path:   python3 make_all_plots.py /some/other/dvd_project

COLAB SETUP (two cells):
  1) on the laptop:   cd ~/dvd_project && zip -r dvd_data.zip data
     then upload dvd_data.zip with the Files panel, and in a cell:
         !mkdir -p /content/dvd_project && unzip -o dvd_data.zip -d /content/dvd_project
  2)     !python3 make_all_plots.py            (after uploading this file too)
     or paste this whole file into a cell and run it.

PLOTS WRITTEN TO <project>/plots/
  Deliverable 3   trend_off_vs_temp.png, trend_off_vs_vth0.png, trend_gate_vs_toxe.png
  (data section)  dataset_current_ranges.png
  Deliverable 7   gate_parity_plot.png
  (gate section)  gate_accuracy_per_case.png
  Deliverable 8   parity8_parity_plot.png
  (error analysis) circuit_error_vs_conditions.png, circuit_ratio_hist.png
Missing input files are reported and skipped, never fatal.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- locations
# Notebook kernels (Colab/Jupyter) put their own flags like "-f /...kernel.json"
# into sys.argv, so only a real path argument counts: no leading "-", and not
# the kernel file.
_args = [a for a in sys.argv[1:] if not a.startswith("-") and "kernel" not in a]
if _args:
    ROOT = Path(_args[0]).expanduser()
elif Path("/content").exists():                       # Google Colab
    ROOT = Path("/content/dvd_project")
else:
    ROOT = Path.home() / "dvd_project"
DATA, PLOTS = ROOT / "data", ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
print(f"Reading {DATA}\nWriting {PLOTS}\n")

CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
GATES = ["INV", "NAND2", "NOR2", "AND2", "OR2"]
N_BINS = 20


def need(*names):
    """Return loaded CSVs, or None (with a message) if any is missing."""
    paths = [DATA / n for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"  SKIPPED (missing: {', '.join(missing)})")
        return None
    return [pd.read_csv(p, dtype={"input": str, "input_pattern": str}) for p in paths]


def binned_median(x, y, n=N_BINS):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx = np.array([np.median(x[idx == b]) for b in range(len(edges) - 1) if (idx == b).any()])
    cy = np.array([np.median(y[idx == b]) for b in range(len(edges) - 1) if (idx == b).any()])
    return cx, cy


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)
    print(f"  wrote plots/{name}")


# ---------------------------------------------- Deliverable 3: trend plots
def trend(cases, colors, subset, xcol, xfun, ycol, xlabel, title, fname):
    print(f"[{title}]")
    loaded = need(*[f"{c}_dataset.csv" for c in cases])
    if loaded is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for case, color, d in zip(cases, colors, loaded):
        d = d[subset(d)]
        x = xfun(d[xcol].values)
        y = d[ycol].abs().values / d["w"].values
        x, y = x[y > 0], y[y > 0]
        cx, cy = binned_median(x, y)
        ax.scatter(x, y, s=4, alpha=0.15, color=color)
        ax.plot(cx, cy, "o-", color=color, lw=2, ms=4, label=f"{case} (median per bin, n={len(x)})")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"|{ycol.capitalize()}| / W  (A/m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    save(fig, fname)


high_vds = lambda d: (d["vd"] - d["vs"]).abs() > 0.8 * d["vdd"]
full_oxide = lambda d: ((d["vg"] - d["vs"]).abs() > 0.9 * d["vdd"]) & ((d["vg"] - d["vd"]).abs() > 0.9 * d["vdd"])
trend(["NMOS_OFF", "PMOS_OFF"], ["tab:blue", "tab:red"], high_vds, "temp", lambda v: v, "id",
      "temperature (\u00b0C)", "OFF current vs temperature", "trend_off_vs_temp.png")
trend(["NMOS_OFF", "PMOS_OFF"], ["tab:blue", "tab:red"], high_vds, "vth0", np.abs, "id",
      "|vth0| (V)", "OFF current vs |vth0|", "trend_off_vs_vth0.png")
trend(["NMOS_ON", "PMOS_ON"], ["tab:blue", "tab:red"], full_oxide, "toxe", lambda v: v * 1e9, "ig",
      "toxe (nm)", "Gate current vs toxe", "trend_gate_vs_toxe.png")

# ------------------------------------- data section: current ranges per case
print("[Dataset current ranges]")
loaded = need(*[f"{c}_dataset.csv" for c in CASES])
if loaded is not None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-16, -2, 70)
    for case, d in zip(CASES, loaded):
        ax.hist(np.log10(d["id"].abs().clip(lower=1e-18)), bins=bins, histtype="step", lw=1.8, label=case)
    ax.set_xlabel("log10 |Id| (A)")
    ax.set_ylabel("samples")
    ax.set_title("Drain-current magnitude per case (10,000 uniform samples each)")
    ax.legend(fontsize=8)
    save(fig, "dataset_current_ranges.png")

# --------------------------------------- Deliverable 7 + gate accuracy chart
print("[Gate parity plot]")
loaded = need(*[f"{g}_gate_comparison.csv" for g in GATES])
gate_rows = []
if loaded is not None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = np.inf, 0.0
    for g, d in zip(GATES, loaded):
        ax.scatter(d["spice_leakage"], d["ml_leakage"], s=6, alpha=0.4, label=g)
        lo, hi = min(lo, d["spice_leakage"].min()), max(hi, d["spice_leakage"].max())
        e = (d["ml_leakage"] - d["spice_leakage"]).abs() / d["spice_leakage"]
        for inp, s in d.assign(e=e).groupby("input")["e"]:
            gate_rows.append((f"{g} {inp}", s.mean(), s.median()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SPICE gate leakage (A)")
    ax.set_ylabel("ML gate leakage (A)")
    n = sum(len(d) for d in loaded)
    acc = 100 * (1 - np.mean([r[1] for r in gate_rows]))
    ax.set_title(f"Gates: ML vs SPICE, {n} points, mean accuracy {acc:.1f}%")
    ax.legend(fontsize=8)
    save(fig, "gate_parity_plot.png")

    print("[Gate accuracy per case]")
    labels = [r[0] for r in gate_rows]
    means = [100 * (1 - r[1]) for r in gate_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(labels)), means, color="tab:blue")
    ax.axhline(acc, color="k", ls="--", lw=1, label=f"mean {acc:.1f}%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("mean accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy per gate and input case (200 corners each)")
    ax.bar_label(bars, fmt="%.0f", fontsize=7)
    ax.legend(fontsize=8)
    save(fig, "gate_accuracy_per_case.png")

# ------------------------------------------------- Deliverable 8 + analysis
print("[Parity-checker parity plot]")
loaded = need("parity8_comparison.csv")
if loaded is not None:
    (c,) = loaded
    e = (c["ml_leakage"] - c["spice_leakage"]).abs() / c["spice_leakage"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(c["spice_leakage"], c["ml_leakage"], s=5, alpha=0.3, color="tab:blue")
    lo, hi = c["spice_leakage"].min(), c["spice_leakage"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SPICE circuit leakage (A)")
    ax.set_ylabel("ML circuit leakage (A)")
    ax.set_title(f"8-bit parity checker: {len(c)} points, {c['corner'].nunique()} corners\n"
                 f"mean accuracy {100 * (1 - e.mean()):.1f}%, median error {100 * e.median():.1f}%")
    ax.legend(fontsize=8)
    save(fig, "parity8_parity_plot.png")

    print("[Circuit error vs corner conditions]")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    signed = (c["ml_leakage"] / c["spice_leakage"] - 1) * 100
    for ax, (col, label) in zip(axes, [("temp", "temperature (\u00b0C)"), ("vdd", "VDD (V)")]):
        cx, cy = binned_median(c[col].values, e.values * 100)
        _, cb = binned_median(c[col].values, signed.values)
        ax.plot(cx, cy, "o-", color="tab:blue", label="median |error|")
        ax.plot(cx, cb, "s--", color="tab:red", label="median signed error (bias)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(label)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("error (%)")
    fig.suptitle("Circuit error vs corner conditions (median per bin)")
    save(fig, "circuit_error_vs_conditions.png")

    print("[Circuit ML/SPICE ratio histogram]")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(c["ml_leakage"] / c["spice_leakage"], bins=60, color="tab:blue")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="ideal")
    ax.set_xlabel("ML / SPICE leakage ratio")
    ax.set_ylabel("points")
    ax.set_title("Circuit prediction ratio over all corners and vectors")
    ax.legend(fontsize=8)
    save(fig, "circuit_ratio_hist.png")

# ------------------------------------------------------------------ summary
print("\n=== Accuracy summary (from the CSVs above) ===")
if gate_rows:
    gm = [r[1] for r in gate_rows]
    gmed = [r[2] for r in gate_rows]
    print(f"GATES   (18 cases): mean accuracy {100 * (1 - np.mean(gm)):.1f}%   100-median {100 * (1 - np.mean(gmed)):.1f}%")
if (DATA / "parity8_comparison.csv").exists():
    c = pd.read_csv(DATA / "parity8_comparison.csv")
    e = (c["ml_leakage"] - c["spice_leakage"]).abs() / c["spice_leakage"]
    print(f"CIRCUIT ({len(c)} points): mean accuracy {100 * (1 - e.mean()):.1f}%   100-median {100 * (1 - e.median()):.1f}%   "
          f"p95 error {100 * e.quantile(0.95):.1f}%")
print("\nNote: transistor predicted-vs-SPICE plots (Deliverable 5) are made locally by "
      "train_final.py (plots/final_<CASE>_<current>.png), since they need the saved models.")
