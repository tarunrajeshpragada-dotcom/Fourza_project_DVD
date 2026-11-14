"""
build_parity8_netlist.py
---------------------------
Builds the complete 8-bit parity checker circuit: a binary tree of 7 XOR
units (4+2+1), each XOR built from 4 NAND2 gates (the construction verified
against all 256 input combinations before any SPICE work started) = 28
NAND2 gates = 112 transistors total.

This module only builds the GATE TOPOLOGY (reused by both the SPICE
netlist generator and the ML hierarchical-prediction script) -- it contains
no SPICE or ML code itself.
"""


def xor_unit(name, in_a, in_b, out_node):
    """4 NAND2 gates implementing XOR(in_a, in_b) -> out_node, per the
    construction verified against all 256 8-bit input patterns:
        c = NAND(a,b); d = NAND(a,c); e = NAND(b,c); out = NAND(d,e)
    Returns a list of (gate_name, inA_node, inB_node, out_node) tuples.
    """
    c, d, e = f"{name}_c", f"{name}_d", f"{name}_e"
    return [
        (f"{name}_g1", in_a, in_b, c),
        (f"{name}_g2", in_a, c, d),
        (f"{name}_g3", in_b, c, e),
        (f"{name}_g4", d, e, out_node),
    ]


def build_parity8_circuit():
    """Returns the full list of 28 (gate_name, inA, inB, out) NAND2 instances
    for the 8-bit parity tree: inputs a0..a7, final output 'parity_out'."""
    gates = []
    inputs = [f"a{i}" for i in range(8)]
    l1 = [f"l1_{i}" for i in range(4)]
    for i in range(4):
        gates += xor_unit(f"L1_{i}", inputs[2 * i], inputs[2 * i + 1], l1[i])
    l2 = [f"l2_{i}" for i in range(2)]
    gates += xor_unit("L2_0", l1[0], l1[1], l2[0])
    gates += xor_unit("L2_1", l1[2], l1[3], l2[1])
    gates += xor_unit("L3_0", l2[0], l2[1], "parity_out")
    return gates


INPUT_NODES = [f"a{i}" for i in range(8)]
OUTPUT_NODE = "parity_out"


if __name__ == "__main__":
    gates = build_parity8_circuit()
    print(f"{len(gates)} NAND2 gates, {len(gates) * 4} transistors")
    print(f"Inputs: {INPUT_NODES}")
    print(f"Output: {OUTPUT_NODE}")
