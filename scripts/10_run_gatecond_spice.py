"""
10_run_gatecond_spice.py
---------------------------
Simulates the gate-condition samples from 09_generate_gatecond_inputs.py
through the SAME verified single-transistor netlist logic as every other
dataset (02_run_spice.run_case), for NMOS_OFF and PMOS_OFF.

Usage (after 09_generate_gatecond_inputs.py NMOS_OFF and ... PMOS_OFF):
    python3 10_run_gatecond_spice.py
Output:
    data/NMOS_OFF_gatecond_dataset.csv, data/PMOS_OFF_gatecond_dataset.csv
"""
from importlib import import_module

run_spice = import_module("02_run_spice")

for case in ["NMOS_OFF", "PMOS_OFF"]:
    run_spice.run_case(case, in_csv_name=f"{case}_gatecond_inputs.csv",
                       out_csv_name=f"{case}_gatecond_dataset.csv", net_suffix="_gatecond")
