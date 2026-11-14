"""
08_run_augmented_spice.py
----------------------------
Simulates the targeted augmentation samples (07_generate_augmented_inputs.py)
through the SAME, already-verified SPICE netlist/parsing logic as the
original dataset (reuses 02_run_spice.run_case with custom filenames --
no duplicated/new SPICE-running code), then concatenates original + augmented
into one combined per-case dataset for retraining.

Usage:
    python3 08_run_augmented_spice.py NMOS_ON
Output:
    data/<CASE>_aug_dataset.csv       (3,000 simulated augmentation rows)
    data/<CASE>_combined_dataset.csv  (13,000 rows: original 10k + augmented 3k)
"""
import sys

import pandas as pd

from common import PROJECT_ROOT
from importlib import import_module

run_spice = import_module("02_run_spice")


def run(case_name: str):
    aug_df = run_spice.run_case(
        case_name,
        in_csv_name=f"{case_name}_aug_inputs.csv",
        out_csv_name=f"{case_name}_aug_dataset.csv",
        net_suffix="_aug",
    )

    original_df = pd.read_csv(PROJECT_ROOT / "data" / f"{case_name}_dataset.csv")
    combined = pd.concat([original_df, aug_df], ignore_index=True)
    out_path = PROJECT_ROOT / "data" / f"{case_name}_combined_dataset.csv"
    combined.to_csv(out_path, index=False)
    print(f"[{case_name}] combined dataset: {len(original_df)} original + {len(aug_df)} augmented = {len(combined)} rows")
    print(f"[{case_name}] Wrote {out_path}")


if __name__ == "__main__":
    case_name = sys.argv[1] if len(sys.argv) > 1 else "NMOS_ON"
    run(case_name)
