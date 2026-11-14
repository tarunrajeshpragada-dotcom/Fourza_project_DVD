"""
consolidate_parity8_results.py
----------------------------------
Deliverable 8: "the same metrics as the gate results table for your test
circuit, and a parity plot of ML against SPICE leakage on log axes."
Also compares directly against the gate-level mean accuracy, per the
spec's "expect the accuracy to be somewhat lower than at the gate level."
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import PROJECT_ROOT


def run():
    df = pd.read_csv(PROJECT_ROOT / "data" / "parity8_comparison.csv", dtype={"input_pattern": str})
    rel_err = (df["ml_leakage"] - df["spice_leakage"]).abs() / df["spice_leakage"]
    bias = (df["ml_leakage"] - df["spice_leakage"]).median()

    summary = pd.DataFrame([dict(
        n=len(df), n_corners=df["corner"].nunique(), n_patterns=df["input_pattern"].nunique(),
        spice_median=df["spice_leakage"].median(), ml_median=df["ml_leakage"].median(),
        bias=bias, median_rel_err=rel_err.median(), p95_rel_err=rel_err.quantile(0.95),
    )])
    out_path = PROJECT_ROOT / "data" / "parity8_results_table.csv"
    summary.to_csv(out_path, index=False)
    print("=== Test-circuit results (Deliverable 8) ===")
    print(summary.round(4).to_string(index=False))
    print(f"Wrote {out_path}")

    gate_table_path = PROJECT_ROOT / "data" / "gate_results_table.csv"
    if gate_table_path.exists():
        gate_df = pd.read_csv(gate_table_path)
        gate_mean_acc = (1 - gate_df["median_rel_err"]).mean() * 100
        circuit_acc = (1 - rel_err.median()) * 100
        print(f"\nGate-level mean accuracy: {gate_mean_acc:.1f}%")
        print(f"Circuit-level accuracy:   {circuit_acc:.1f}%")
        print(f"(spec expects circuit-level to be somewhat lower than gate-level)")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["spice_leakage"], df["ml_leakage"], s=6, alpha=0.3, color="tab:blue")
    lims = [df[["spice_leakage", "ml_leakage"]].min().min(), df[["spice_leakage", "ml_leakage"]].max().max()]
    ax.plot(lims, lims, "k--", linewidth=1, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SPICE circuit leakage (A)")
    ax.set_ylabel("ML-predicted circuit leakage (A)")
    ax.set_title(f"8-bit parity checker parity plot ({df['corner'].nunique()} corners x {df['input_pattern'].nunique()} patterns)")
    ax.legend()
    fig.tight_layout()
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(plot_dir / "parity8_parity_plot.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_dir / 'parity8_parity_plot.png'}")


if __name__ == "__main__":
    run()
