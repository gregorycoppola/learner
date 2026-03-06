"""
TM-0002: Binary Decrementer

Tape holds a binary number, MSB on the left, e.g. [1, 0, 1, 1] = 11.
Scans right to the end, then scans left subtracting 1 with borrow.

States:
  scan_right  — move right to find the end of the number
  borrow      — moving left, propagating borrow
  done        — accept

Transitions:
  borrow + 1 -> write 0, borrow resolved, done
  borrow + 0 -> write 1, borrow continues (like carry in reverse)
  borrow + _ -> write _, done (underflow: 0 - 1 wraps, left as blank)

Hard states: {borrow}
"""
from learner.core.tm import TuringMachine, Transition, Move


def make_decrementer() -> TuringMachine:
    transitions = {
        ("scan_right", "0"): Transition("0", "0", Move.RIGHT, "scan_right"),
        ("scan_right", "1"): Transition("1", "1", Move.RIGHT, "scan_right"),
        ("scan_right", "_"): Transition("_", "_", Move.LEFT,  "borrow"),

        ("borrow", "1"):     Transition("1", "0", Move.LEFT,  "done"),
        ("borrow", "0"):     Transition("0", "1", Move.LEFT,  "borrow"),
        ("borrow", "_"):     Transition("_", "_", Move.RIGHT, "done"),
    }

    return TuringMachine(
        name="tm0002_binary_decrementer",
        states=["scan_right", "borrow", "done"],
        alphabet=["0", "1", "_"],
        blank="_",
        initial_state="scan_right",
        accept_states=["done"],
        transitions=transitions,
    )