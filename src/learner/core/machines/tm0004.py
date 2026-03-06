"""
TM-0004: Binary Left Shift (multiply by 2)

Shifts all bits one position to the left, appending a 0 at the right end.
E.g. [1, 0, 1] → [0, 1, 0] with the leading bit lost, OR equivalently
the tape grows: handled here by carrying the leftmost bit through.

Implementation: two-pass.
  Pass 1 (scan_right): scan to the end of the tape.
  Pass 2 (shift_left): scan left, moving each bit one position right
                       relative to the previous, carrying the displaced bit.

Concretely:
  - Remember the bit under the head, write the carry, move left.

States:
  scan_right   — find the right end
  carry_0      — moving left, carrying a 0 to write at current pos
  carry_1      — moving left, carrying a 1 to write at current pos
  done         — accept

Hard states: {carry_1} — minority, requires writing 1 while moving left.
"""
from learner.core.tm import TuringMachine, Transition, Move


def make_left_shift() -> TuringMachine:
    transitions = {
        # Pass 1: scan right to end
        ("scan_right", "0"): Transition("0", "0", Move.RIGHT, "scan_right"),
        ("scan_right", "1"): Transition("1", "1", Move.RIGHT, "scan_right"),
        # Hit blank: start shifting left, carry a 0 (the new trailing zero)
        ("scan_right", "_"): Transition("_", "_", Move.LEFT,  "carry_0"),

        # Pass 2: shift left carrying a 0
        # Read 0: write carry (0), pick up 0, move left
        ("carry_0", "0"):    Transition("0", "0", Move.LEFT,  "carry_0"),
        # Read 1: write carry (0), pick up 1, move left
        ("carry_0", "1"):    Transition("1", "0", Move.LEFT,  "carry_1"),
        # Reached left edge: write carry (0), done
        ("carry_0", "_"):    Transition("_", "0", Move.RIGHT, "done"),

        # Pass 2: shift left carrying a 1
        # Read 0: write carry (1), pick up 0, move left
        ("carry_1", "0"):    Transition("0", "1", Move.LEFT,  "carry_0"),
        # Read 1: write carry (1), pick up 1, move left
        ("carry_1", "1"):    Transition("1", "1", Move.LEFT,  "carry_1"),
        # Reached left edge: write carry (1), done
        ("carry_1", "_"):    Transition("_", "1", Move.RIGHT, "done"),
    }

    return TuringMachine(
        name="tm0004_binary_left_shift",
        states=["scan_right", "carry_0", "carry_1", "done"],
        alphabet=["0", "1", "_"],
        blank="_",
        initial_state="scan_right",
        accept_states=["done"],
        transitions=transitions,
    )