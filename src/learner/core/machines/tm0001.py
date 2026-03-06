"""
TM-0001: Binary Incrementer

Tape holds a binary number, MSB on the left, e.g. [1, 0, 1, 1] = 11.
Scans right to the end, then scans left adding 1 with carry.

States:
  scan_right  — move right to find the end of the number
  carry       — moving left, propagating carry
  done        — accept

Hard states: {carry}
"""
from learner.core.tm import TuringMachine, Transition, Move


def make_incrementer() -> TuringMachine:
    transitions = {
        ("scan_right", "0"): Transition("0", "0", Move.RIGHT, "scan_right"),
        ("scan_right", "1"): Transition("1", "1", Move.RIGHT, "scan_right"),
        ("scan_right", "_"): Transition("_", "_", Move.LEFT,  "carry"),

        ("carry", "0"):      Transition("0", "1", Move.LEFT,  "done"),
        ("carry", "1"):      Transition("1", "0", Move.LEFT,  "carry"),
        ("carry", "_"):      Transition("_", "1", Move.RIGHT, "done"),
    }

    return TuringMachine(
        name="tm0001_binary_incrementer",
        states=["scan_right", "carry", "done"],
        alphabet=["0", "1", "_"],
        blank="_",
        initial_state="scan_right",
        accept_states=["done"],
        transitions=transitions,
    )