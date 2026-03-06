"""Turing machine definition and execution."""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Move(str, Enum):
    LEFT = "L"
    RIGHT = "R"
    STAY = "S"


@dataclass
class Transition:
    read: str
    write: str
    move: Move
    next_state: str


@dataclass
class TuringMachine:
    """A Turing machine definition."""
    name: str
    states: list[str]
    alphabet: list[str]          # tape symbols including blank
    blank: str
    initial_state: str
    accept_states: list[str]
    transitions: dict[tuple[str, str], Transition]  # (state, read) -> Transition

    def step(self, tape: list[str], head: int, state: str) -> tuple[list[str], int, str, bool]:
        """
        Execute one TM step.
        Returns (new_tape, new_head, new_state, halted).
        """
        symbol = tape[head] if 0 <= head < len(tape) else self.blank
        key = (state, symbol)

        if key not in self.transitions:
            return tape, head, state, True  # halt — no transition defined

        t = self.transitions[key]

        # Write
        new_tape = tape.copy()
        while head >= len(new_tape):
            new_tape.append(self.blank)
        new_tape[head] = t.write

        # Move
        if t.move == Move.RIGHT:
            new_head = head + 1
        elif t.move == Move.LEFT:
            new_head = max(0, head - 1)
        else:
            new_head = head

        # Extend tape if needed
        while new_head >= len(new_tape):
            new_tape.append(self.blank)

        halted = t.next_state in self.accept_states
        return new_tape, new_head, t.next_state, halted

    def run(self, tape: list[str], max_steps: int = 1000) -> list[dict]:
        """
        Run the TM to completion, returning a trace of every step.
        Each trace entry is a snapshot: tape, head, state, halted.
        """
        head = 0
        state = self.initial_state
        current_tape = tape.copy()

        # Extend tape to at least cover the input
        while len(current_tape) < max(len(tape), 1):
            current_tape.append(self.blank)

        trace = []
        trace.append({
            "step": 0,
            "tape": current_tape.copy(),
            "head": head,
            "state": state,
            "halted": False,
        })

        for i in range(1, max_steps + 1):
            current_tape, head, state, halted = self.step(current_tape, head, state)
            trace.append({
                "step": i,
                "tape": current_tape.copy(),
                "head": head,
                "state": state,
                "halted": halted,
            })
            if halted:
                break

        return trace


def format_tape(tape: list[str], head: int, blank: str = "_") -> str:
    """Pretty print a tape with head marker."""
    parts = []
    for i, sym in enumerate(tape):
        if i == head:
            parts.append(f"[{sym}]")
        else:
            parts.append(f" {sym} ")
    # Strip trailing blanks for display
    result = "".join(parts).rstrip()
    return result