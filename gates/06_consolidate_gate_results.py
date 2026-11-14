"""
06_consolidate_gate_results.py
---------------------------------
Deliverable 6 (gate results table: SPICE median, ML median, bias, median
error, 95th-pct error, for each of the 18 gate/input cases) and Deliverable 7
(gate parity plot: ML vs SPICE leakage, all gate simulations, log axes).
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import PROJECT_ROOT

GATES = ["INV", "NAND2", "NOR2", "AND2", "OR2"]


def check_freshness():
    """
    Refuse to silently consolidate stale comparison CSVs. If any trained model
    is newer than a gate's comparison CSV, that comparison was computed against
    an OLDER set of models (e.g. a predict_*.py run crashed and left yesterday's
    CSV sitting there) -- this caught a real failure mode once already.
    """
    model_dir = PROJECT_ROOT / "models" / "trained_v2"
    model_files = list(model_dir.glob("*.joblib"))
    if not model_files:
        raise RuntimeError(
            f"No trained models found in {model_dir} -- run 03_train_models_local.py "
            f"(fully, without interrupting it) before running the gate prediction scripts."
        )
    newest_model_time = max(f.stat().st_mtime for f in model_files)

    stale = []
    for gate in GATES:
        comp_path = PROJECT_ROOT / "data" / f"{gate}_gate_comparison.csv"
        if not comp_path.exists():
            stale.append((gate, "missing entirely"))
        elif comp_path.stat().st_mtime < newest_model_time:
            age = newest_model_time - comp_path.stat().st_mtime
            stale.append((gate, f"{age/60:.1f} min older than the current models"))
    if stale:
        lines = "\n".join(f"  - {g}_gate_comparison.csv: {reason}" for g, reason in stale)
        raise RuntimeError(
            f"STALE DATA DETECTED -- refusing to consolidate:\n{lines}\n"
            f"Re-run the corresponding predict_<gate>_ml.py script(s) (check for a "
            f"crash/traceback in their output) before re-running this script."
        )


def load_all():
    frames = []
    for gate in GATES:
        df = pd.read_csv(PROJECT_ROOT / "data" / f"{gate}_gate_comparison.csv", dtype={"input": str})
        df["gate"] = gate
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run():
    check_freshness()
    all_df = load_all()

    rows = []
    for (gate, inp), sub in all_df.groupby(["gate", "input"], sort=False):
        bias = (sub["ml_leakage"] - sub["spice_leakage"]).median()
        rel_err = (sub["ml_leakage"] - sub["spice_leakage"]).abs() / sub["spice_leakage"]
        rows.append(dict(
            gate=gate, input=inp, n=len(sub),
            spice_median=sub["spice_leakage"].median(),
            ml_median=sub["ml_leakage"].median(),
            bias=bias,
            median_rel_err=rel_err.median(),
            p95_rel_err=rel_err.quantile(0.95),
        ))
    # preserve the natural gate order (INV, NAND2, NOR2, AND2, OR2), not alphabetical
    results = pd.DataFrame(rows)
    results["gate"] = pd.Categorical(results["gate"], categories=GATES, ordered=True)
    results = results.sort_values(["gate", "input"]).reset_index(drop=True)

    out_path = PROJECT_ROOT / "data" / "gate_results_table.csv"
    results.to_csv(out_path, index=False)
    print(f"=== Gate results table ({len(results)} rows -- expect 18) ===")
    pd.set_option("display.width", 160)
    print(results.round(4).to_string(index=False))
    print(f"\nWrote {out_path}")

    # Deliverable 7: gate parity plot, all gate simulations, log axes
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"INV": "tab:blue", "NAND2": "tab:orange", "NOR2": "tab:green", "AND2": "tab:red", "OR2": "tab:purple"}
    for gate in GATES:
        sub = all_df[all_df["gate"] == gate]
        ax.scatter(sub["spice_leakage"], sub["ml_leakage"], s=8, alpha=0.4, label=gate, color=colors[gate])
    lims = [all_df[["spice_leakage", "ml_leakage"]].min().min(), all_df[["spice_leakage", "ml_leakage"]].max().max()]
    ax.plot(lims, lims, "k--", linewidth=1, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SPICE gate leakage (A)")
    ax.set_ylabel("ML-predicted gate leakage (A)")
    ax.set_title("Gate parity plot -- all 5 gates, all 18 input cases, 200 corners each")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(plot_dir / "gate_parity_plot.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_dir / 'gate_parity_plot.png'}")

    return results


if __name__ == "__main__":
    run()
