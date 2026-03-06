"""Turing machine registry."""
from learner.core.machines.tm0001 import make_incrementer
from learner.core.machines.tm0002 import make_decrementer

MACHINES = {
    "tm0001": make_incrementer,
    "tm0002": make_decrementer,
    # aliases for convenience
    "incrementer": make_incrementer,
    "decrementer": make_decrementer,
}

# States that are minority / hard and should receive upweighted loss.
# Add an entry here when registering a new machine.
HARD_STATES = {
    "tm0001":      {"carry"},
    "incrementer": {"carry"},
    "tm0002":      {"borrow"},
    "decrementer": {"borrow"},
}


def get_machine(name: str):
    if name not in MACHINES:
        raise ValueError(
            f"Unknown machine '{name}'. Available: {sorted(MACHINES.keys())}"
        )
    return MACHINES[name]()


def get_hard_states(name: str) -> set[str]:
    return HARD_STATES.get(name, set())