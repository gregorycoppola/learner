"""
TM-0003: Binary Complement

Flips every bit on the tape (0→1, 1→0). Blanks are left unchanged.
Scans right until blank, then halts.

States:
  flip  — scan right, flipping each bit
  done  — accept

This is the simplest possible non-trivial TM: purely local operation,
single pass, no head reversal, no carry or borrow.

Hard states: {} (none — all transitions are equally represented)
"""
from learner.core.tm import TuringMachine, Transition, Move


def make_complement() -> TuringMachine:
    transitions = {
        ("flip", "0"): Transition("0", "1", Move.RIGHT, "flip"),
        ("flip", "1"): Transition("1", "0", Move.RIGHT, "flip"),
        ("flip", "_"): Transition("_", "_", Move.STAY,  "done"),
    }

    return TuringMachine(
        name="tm0003_binary_complement",
        states=["flip", "done"],
        alphabet=["0", "1", "_"],
        blank="_",
        initial_state="flip",
        accept_states=["done"],
        transitions=transitions,
    )