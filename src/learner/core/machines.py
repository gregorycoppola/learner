"""Built-in Turing machine definitions."""
from learner.core.tm import TuringMachine, Transition, Move


def make_incrementer() -> TuringMachine:
    """
    Binary incrementer TM.

    Tape holds a binary number, MSB on the left, e.g. [1, 0, 1, 1] = 11.
    The TM scans right to the end, then scans left adding 1 with carry.

    States:
      scan_right  — move right to find the end of the number
      carry       — moving left, propagating carry
      done        — accept

    Alphabet: '0', '1', '_' (blank)
    """
    transitions = {
        # scan_right: keep moving right until blank
        ("scan_right", "0"): Transition("0", "0", Move.RIGHT, "scan_right"),
        ("scan_right", "1"): Transition("1", "1", Move.RIGHT, "scan_right"),
        ("scan_right", "_"): Transition("_", "_", Move.LEFT,  "carry"),

        # carry: moving left, adding 1
        # If we see 0: write 1, no more carry, move left to done
        ("carry", "0"):      Transition("0", "1", Move.LEFT,  "done"),
        # If we see 1: write 0, carry continues
        ("carry", "1"):      Transition("1", "0", Move.LEFT,  "carry"),
        # If we reach the left edge with carry still set: write 1 (new MSB)
        ("carry", "_"):      Transition("_", "1", Move.RIGHT, "done"),
    }

    return TuringMachine(
        name="binary_incrementer",
        states=["scan_right", "carry", "done"],
        alphabet=["0", "1", "_"],
        blank="_",
        initial_state="scan_right",
        accept_states=["done"],
        transitions=transitions,
    )


MACHINES = {
    "incrementer": make_incrementer,
}


def get_machine(name: str) -> TuringMachine:
    if name not in MACHINES:
        raise ValueError(f"Unknown machine '{name}'. Available: {list(MACHINES.keys())}")
    return MACHINES[name]()