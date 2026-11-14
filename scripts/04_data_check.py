"""
04_data_check.py
-------------------
Deliverable 3: the Kirchhoff check for each case and three trend plots
(OFF current vs temperature, OFF current vs vth0, gate current vs toxe).

Kirchhoff check: |Id + Ig + Is + Ib| per sample, summarised per case and
compared with the size of the currents themselves.

Trend plots: every sample is a joint random draw of ~10 parameters, so a raw
scatter is dominated by the other parameters. Each plot therefore uses a
subset at comparable bias and divides by width, then shows the median per
quantile bin over the scatter:
  - OFF current (NMOS_OFF, PMOS_OFF): |Id|/W for |Vds| > 0.8 VDD
    (an OFF device holding nearly the full supply), vs temperature and vs |vth0|;
  - gate current (NMOS_ON, PMOS_ON): |Ig|/W vs toxe for samples with nearly
    the full supply across the oxide at both ends (|Vgs| and |Vgd| > 0.9 VDD);
    otherwise the random Vd, Vs change the oxide voltage far more than toxe does.
Expected: OFF current rises with temperature and falls with |vth0|; gate
current falls with toxe (tunnelling through a thicker oxide).

Usage:   python3 04_data_check.py
Outputs: data/kirchhoff_check.csv, plots/trend_off_vs_temp.png,
         plots/trend_off_vs_vth0.png, plots/trend_gate_vs_toxe.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROJECT_ROOT

DATA = PROJECT_ROOT / "data"
PLOTS = PROJECT_ROOT / "plots"
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
COLORS = {"NMOS_OFF": "tab:blue", "PMOS_OFF": "tab:red", "NMOS_ON": "tab:blue", "PMOS_ON": "tab:red"}
N_BINS = 20


def load(case):
    return pd.read_csv(DATA / f"{case}_dataset.csv")


def kirchhoff():
    rows = []
    for case in CASES:
        d = load(case)
        kcl = (d["id"] + d["ig"] + d["is"] + d["ib"]).abs()
        largest = d[["id", "ig", "is", "ib"]].abs().max(axis=1)
        rows.append(dict(case=case, n=len(d), kcl_median_A=kcl.median(), kcl_max_A=kcl.max(),
                         median_abs_id_A=d["id"].abs().median(),
                         median_kcl_over_largest_current=(kcl / largest.clip(lower=1e-30)).median()))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "kirchhoff_check.csv", index=False)
    print("=== Kirchhoff check: |Id + Ig + Is + Ib| ===")
    print(out.to_string(index=False))


def binned(x, y):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx = [np.median(x[idx == b]) for b in range(len(edges) - 1) if (idx == b).any()]
    cy = [np.median(y[idx == b]) for b in range(len(edges) - 1) if (idx == b).any()]
    return np.array(cx), np.array(cy)


def trend(cases, subset, xcol, xfun, ycol, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for case in cases:
        d = load(case)
        d = d[subset(d)]
        x = xfun(d[xcol].values)
        y = d[ycol].abs().values / d["w"].values
        ok = y > 0
        x, y = x[ok], y[ok]
        cx, cy = binned(x, y)
        ax.scatter(x, y, s=4, alpha=0.15, color=COLORS[case])
        ax.plot(cx, cy, "o-", color=COLORS[case], lw=2, ms=4, label=f"{case} (median per bin, n={len(x)})")
        slope = np.polyfit(cx, np.log10(cy), 1)[0]
        print(f"{title} | {case}: median goes from {cy[0]:.3e} to {cy[-1]:.3e} A/m "
              f"({slope:+.3g} decades per unit of {xlabel})")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / fname, dpi=150)
    plt.close(fig)


def main():
    kirchhoff()
    print()
    high_vds = lambda d: (d["vd"] - d["vs"]).abs() > 0.8 * d["vdd"]
    full_oxide = lambda d: ((d["vg"] - d["vs"]).abs() > 0.9 * d["vdd"]) & ((d["vg"] - d["vd"]).abs() > 0.9 * d["vdd"])
    trend(["NMOS_OFF", "PMOS_OFF"], high_vds, "temp", lambda v: v, "id",
          "temperature (C)", "|Id| / W (A/m)", "OFF current vs temperature", "trend_off_vs_temp.png")
    trend(["NMOS_OFF", "PMOS_OFF"], high_vds, "vth0", np.abs, "id",
          "|vth0| (V)", "|Id| / W (A/m)", "OFF current vs |vth0|", "trend_off_vs_vth0.png")
    trend(["NMOS_ON", "PMOS_ON"], full_oxide, "toxe", lambda v: v * 1e9, "ig",
          "toxe (nm)", "|Ig| / W (A/m)", "Gate current vs toxe", "trend_gate_vs_toxe.png")
    print(f"\nWrote {DATA / 'kirchhoff_check.csv'} and plots/trend_*.png")


if __name__ == "__main__":
    main()
