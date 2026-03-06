"""Turing machine registry."""
from learner.core.machines.tm0001 import make_incrementer
from learner.core.machines.tm0002 import make_decrementer
from learner.core.machines.tm0003 import make_complement
from learner.core.machines.tm0004 import make_left_shift
from learner.core.machines.tm0005 import make_right_shift

MACHINES = {
    "tm0001": make_incrementer,
    "tm0002": make_decrementer,
    "tm0003": make_complement,
    "tm0004": make_left_shift,
    "tm0005": make_right_shift,
    # convenience aliases
    "incrementer":  make_incrementer,
    "decrementer":  make_decrementer,
    "complement":   make_complement,
    "left_shift":   make_left_shift,
    "right_shift":  make_right_shift,
}

HARD_STATES = {
    "tm0001":      {"carry"},
    "tm0002":      {"borrow"},
    "tm0003":      set(),          # no hard states — uniform transitions
    "tm0004":      {"carry_1"},
    "tm0005":      {"carry_1"},
    "incrementer": {"carry"},
    "decrementer": {"borrow"},
    "complement":  set(),
    "left_shift":  {"carry_1"},
    "right_shift": {"carry_1"},
}


def get_machine(name: str):
    if name not in MACHINES:
        raise ValueError(
            f"Unknown machine '{name}'. Available: {sorted(MACHINES.keys())}"
        )
    return MACHINES[name]()


def get_hard_states(name: str) -> set[str]:
    return HARD_STATES.get(name, set())