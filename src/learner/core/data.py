"""Synthetic data generation from TM traces."""
import random
from learner.core.tm import TuringMachine
from learner.core.machines import get_machine


def int_to_tape(n: int, min_len: int = 4) -> list[str]:
    """Convert a non-negative integer to a binary tape (MSB first)."""
    if n == 0:
        bits = ["0"]
    else:
        bits = []
        x = n
        while x > 0:
            bits.append(str(x & 1))
            x >>= 1
        bits.reverse()
    # Pad to min_len
    while len(bits) < min_len:
        bits = ["0"] + bits
    return bits


def tape_to_int(tape: list[str], blank: str = "_") -> int:
    """Convert a binary tape back to an integer, ignoring blanks."""
    bits = [b for b in tape if b != blank]
    return int("".join(bits), 2) if bits else 0


def generate_pairs(
    machine_name: str = "incrementer",
    n_samples: int = 1000,
    min_val: int = 0,
    max_val: int = 255,
    seed: int = 42,
) -> list[dict]:
    """
    Generate (before, after) step pairs from TM traces.

    Each pair is one TM transition:
      {
        "tape_before": [...],
        "head_before": int,
        "state_before": str,
        "tape_after":  [...],
        "head_after":  int,
        "state_after": str,
      }
    """
    rng = random.Random(seed)
    tm = get_machine(machine_name)
    pairs = []

    while len(pairs) < n_samples:
        n = rng.randint(min_val, max_val)
        tape = int_to_tape(n)
        trace = tm.run(tape)

        # Each consecutive pair of steps is one training example
        for i in range(len(trace) - 1):
            before = trace[i]
            after = trace[i + 1]
            if after["halted"]:
                continue  # skip the halt transition
            pairs.append({
                "tape_before": before["tape"],
                "head_before": before["head"],
                "state_before": before["state"],
                "tape_after":  after["tape"],
                "head_after":  after["head"],
                "state_after": after["state"],
            })
            if len(pairs) >= n_samples:
                break

    return pairs[:n_samples]