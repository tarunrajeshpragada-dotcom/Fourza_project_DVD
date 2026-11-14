"""
spice_netlist_writer.py
--------------------------
Writes the actual SPICE device lines for the 28-NAND2-gate parity checker,
reusing the exact same per-NAND2 transistor pattern already verified in
Part B's run_nand2_spice.py: M1(out,A,x,0) nmos, M2(x,B,0,0) nmos,
M3(out,A,vdd,vdd) pmos, M4(out,B,vdd,vdd) pmos -- just instantiated 28
times with the tree's own node names instead of a single gate's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import L_DIFF, L_MIN, MODEL_FILE

from build_parity8_netlist import build_parity8_circuit, INPUT_NODES, OUTPUT_NODE

W_GATE = L_MIN
AD_AS_GATE = W_GATE * L_DIFF


def device_lines_for_gate(gate_name, in_a, in_b, out_node):
    x_node = f"{gate_name}_x"
    return [
        f"M{gate_name}_1 {out_node} {in_a} {x_node} 0   nmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}",
        f"M{gate_name}_2 {x_node} {in_b} 0 0   nmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}",
        f"M{gate_name}_3 {out_node} {in_a} vdd vdd pmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}",
        f"M{gate_name}_4 {out_node} {in_b} vdd vdd pmos W={W_GATE:.6g} L={L_MIN:.6g} AD={AD_AS_GATE:.6g} AS={AD_AS_GATE:.6g}",
    ]


def build_header():
    gates = build_parity8_circuit()
    lines = [
        "* 8-bit parity checker -- 28 NAND2 gates, 112 transistors",
        f".INCLUDE {MODEL_FILE}",
        ".OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9",
        "",
    ]
    for gate_name, in_a, in_b, out_node in gates:
        lines += device_lines_for_gate(gate_name, in_a, in_b, out_node)
    lines.append("")
    lines.append("Vdd_src vdd 0 0")
    for node in INPUT_NODES:
        lines.append(f"V{node} {node} 0 0")
    lines.append("")
    lines.append(".control")
    return "\n".join(lines)


def all_internal_nodes():
    """Every gate's output AND internal x node, for recording per Part B.4's
    'pre-recorded internal node voltages ... along with the output voltage'."""
    gates = build_parity8_circuit()
    nodes = []
    for gate_name, _, _, out_node in gates:
        nodes.append(out_node)
        nodes.append(f"{gate_name}_x")
    # de-duplicate while preserving order (multiple gates can share an output node
    # name, e.g. a later gate's output IS an earlier gate's input/output alias --
    # not the case here since every gate has a distinct out_node, but safe anyway)
    seen = set()
    unique = []
    for n in nodes:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


if __name__ == "__main__":
    header = build_header()
    print(header[:2000])
    print("...")
    nodes = all_internal_nodes()
    print(f"\n{len(nodes)} internal nodes to record (expect 56 = 28 gates x 2 each)")
