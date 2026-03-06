"""
TM-0005: Binary Right Shift (divide by 2, drop LSB)

Shifts all bits one position to the right, dropping the LSB,
inserting a 0 at the left end.
E.g. [1, 0, 1, 1] → [0, 1, 0, 1]

Implementation: single left-to-right pass carrying displaced bits.

States:
  carry_0   — moving right, carrying a 0 to write at current pos
  carry_1   — moving right, carrying a 1 to write at current pos
  done      — accept

Start in carry_0: the leftmost new bit is always 0 (right shift inserts 0).

Hard states: {carry_1} — carrying a 1 rightward, minority when input
is sparse in 1s.
"""
from learner.core.tm import TuringMachine, Transition, Move


def make_right_shift() -> TuringMachine:
    transitions = {
        # Moving right carrying a 0
        # Read 0: write carry (0), pick up 0, move right
        ("carry_0", "0"):  Transition("0", "0", Move.RIGHT, "carry_0"),
        # Read 1: write carry (0), pick up 1, move right
        ("carry_0", "1"):  Transition("1", "0", Move.RIGHT, "carry_1"),
        # Hit blank: write carry (0), done
        ("carry_0", "_"):  Transition("_", "0", Move.STAY,  "done"),

        # Moving right carrying a 1
        # Read 0: write carry (1), pick up 0, move right
        ("carry_1", "0"):  Transition("0", "1", Move.RIGHT, "carry_0"),
        # Read 1: write carry (1), pick up 1, move right
        ("carry_1", "1"):  Transition("1", "1", Move.RIGHT, "carry_1"),
        # Hit blank: write carry (1), done
        ("carry_1", "_"):  Transition("_", "1", Move.STAY,  "done"),
    }

    return TuringMachine(
        name="tm0005_binary_right_shift",
        states=["carry_0", "carry_1", "done"],
        alphabet=["0", "1", "_"],
        blank="_",
        initial_state="carry_0",
        accept_states=["done"],
        transitions=transitions,
    )