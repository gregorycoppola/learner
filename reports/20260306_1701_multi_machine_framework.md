
# Multi-Machine Framework: Scaling from 1 to 2 Turing Machines

**Date:** 2026-03-06 17:01  
**Branch:** multi-model  
**Machines:** tm0001 (binary incrementer), tm0002 (binary decrementer)

---

## Motivation

The thesis of this project is:

> Transformer networks can learn to simulate arbitrary Turing machines
> via gradient descent.

After confirming the thesis for a single machine (tm0001, binary incrementer,
100% accuracy in 4 epochs), the immediate question is whether the result
generalizes. A single data point proves nothing about "arbitrary." The goal
of this work is to build a framework that makes it easy to add new machines
and accumulate evidence systematically.

This report documents the refactoring from a single-machine system to an
extensible multi-machine framework, and the first replication on tm0002.

---

## Part 1: Technical Report — Framework Architecture

### Before: Single-Machine Coupling

The original codebase had one implicit assumption baked in everywhere:
the machine is the binary incrementer, and its hard state is called "carry."
The specific coupling points were:

In `trainer_hybrid.py`:

    carry_mask = torch.tensor([p["state_before"] == "carry" for p in batch_pairs])

In `trainer_grpo.py` (`_balanced_pairs`):

    carry = [p for p in pool if p["state_before"] == "carry"]
    other = [p for p in pool if p["state_before"] != "carry"]

Both hardcoded the string "carry". Adding any machine with a different state
name would silently produce an empty balanced dataset and crash with:

    RuntimeError: stack expects a non-empty TensorList

### After: Machine Registry with Hard States

The machines module was refactored from a flat file into a package:

    src/learner/core/machines/
      __init__.py    — registry: MACHINES, HARD_STATES, get_machine(), get_hard_states()
      tm0001.py      — binary incrementer
      tm0002.py      — binary decrementer

The registry in `__init__.py` has two dicts:

    MACHINES = {
        "tm0001": make_incrementer,
        "tm0002": make_decrementer,
        "incrementer": make_incrementer,   # convenience alias
        "decrementer": make_decrementer,
    }

    HARD_STATES = {
        "tm0001":      {"carry"},
        "tm0002":      {"borrow"},
        "incrementer": {"carry"},
        "decrementer": {"borrow"},
    }

`get_hard_states(machine_name)` returns the set for that machine, defaulting
to the empty set if not registered (meaning equal weighting for all states).

### The Two Coupling Points Fixed

`trainer_grpo.py` — `_balanced_pairs` now uses `get_hard_states`:

    hard  = get_hard_states(machine_name)
    heavy = [p for p in pool if p["state_before"] in hard]
    easy  = [p for p in pool if p["state_before"] not in hard]

`trainer_hybrid.py` — `hybrid_loss` now takes `hard_states` as a parameter:

    hard_mask = torch.tensor([p["state_before"] in hard_states for p in batch_pairs])

The trainer calls `get_hard_states(machine_name)` at startup and passes it
through. No string literals about specific states anywhere in the training
code.

### Adding a New Machine: One File

To add tm0003, the steps are:

1. Create `src/learner/core/machines/tm0003.py` with a `make_X()` function
2. Add `"tm0003": make_X` to `MACHINES` in `__init__.py`
3. Add `"tm0003": {"hard_state_name"}` to `HARD_STATES`

No other files change. The training pipeline, encoding, model, analysis,
CLI, and server all work automatically.

### Architecture Decisions

**Numeric index (tm0001, tm0002, ...)** rather than names as primary keys.
Names are registered as aliases for convenience but the canonical identifier
is the four-digit index. This makes the experimental record unambiguous —
"tm0001 achieved 100%" is a precise claim, whereas "incrementer achieved
100%" depends on which implementation of incrementer.

**Hard states as a set, not a list.** The `in` operator on a set is O(1)
and the semantics are correct — a state is either hard or it isn't, order
doesn't matter.

**Default to empty set.** `get_hard_states` returns `set()` if the machine
isn't registered. This means a new machine with no `HARD_STATES` entry
trains with equal weighting on all states — a safe default that won't
silently break anything.

**Aliases in both dicts.** `"incrementer"` and `"tm0001"` both work as
machine names everywhere. Old commands and scripts keep working without
modification.

---

## Part 2: Results Report

### tm0001: Binary Incrementer (replication)

**Command:**

    learner hybrid run --machine tm0001 --samples 10000 --epochs 20 --analyze-every 5

**Training curve:**

    Ep  Loss        L-Easy      L-Hard      Val Acc
    1   10.384459   1.435434    3.705317    0.6496
    2   1.529398    0.747344    0.018099    0.8917
    3   0.109050    0.051505    0.002364    0.9955
    4   0.007996    0.003395    0.000639    1.0000

**Epoch 5 analysis (1000 samples):**

    state_before  carry       104  100.0%  100.0%  100.0%  100.0%
    state_before  scan_right  896  100.0%  100.0%  100.0%  100.0%
    tape_changes  True        215  100.0%  100.0%  100.0%  100.0%
    direction     left        215  100.0%  100.0%  100.0%  100.0%
    read_bit      _           111  100.0%  100.0%  100.0%  100.0%

100% on all features. Consistent with the original result.

### tm0002: Binary Decrementer (new)

**Machine definition:** scan right to end, then scan left subtracting 1
with borrow propagation. Mirror of the incrementer.

    (scan_right, 0) → write 0, move RIGHT, scan_right
    (scan_right, 1) → write 1, move RIGHT, scan_right
    (scan_right, _) → write _, move LEFT,  borrow
    (borrow, 1)     → write 0, move LEFT,  done
    (borrow, 0)     → write 1, move LEFT,  borrow
    (borrow, _)     → write _, move RIGHT, done

Hard state: `borrow` — minority class, requires writing a different value
at the head position, same structural challenge as `carry` in tm0001.

**Command:**

    learner hybrid run --machine tm0002 --samples 10000 --epochs 20 --analyze-every 5

**Training curve:**

    Ep  Loss        L-Easy      L-Hard      Val Acc
    1   10.305781   1.459274    3.588983    0.5790
    2   1.659647    0.815226    0.010499    0.9469
    3   0.050880    0.024021    0.001900    0.9988
    4   0.006379    0.002727    0.000455    1.0000

**Epoch 5 analysis (1000 samples):**

    state_before  borrow      115  100.0%  100.0%  100.0%  100.0%
    state_before  scan_right  885  100.0%  100.0%  100.0%  100.0%
    tape_changes  True        225  100.0%  100.0%  100.0%  100.0%

100% on all features in 4 epochs.

### Summary

| Machine | Description       | States | Hard State | Epochs to 100% |
|---------|-------------------|--------|------------|----------------|
| tm0001  | Binary incrementer | 3     | carry      | 4              |
| tm0002  | Binary decrementer | 3     | borrow     | 4              |

Both machines solve to 100% accuracy in 4 epochs with identical
hyperparameters. The decrementer is structurally symmetric to the
incrementer — same tape alphabet, same number of states, same pattern
of one hard minority state with a tape-write operation. The result is
therefore a replication rather than a genuinely new challenge, but it
confirms that the framework generalizes correctly and that the hard-state
balancing and encoding fix carry over without modification.

### Next Machines

The next experiments should test structurally different machines:
ones with more states, different alphabets, or qualitatively different
computational patterns (accept/reject, multi-pass, delimiter-based).
tm0003 candidates: binary left shift, unary-to-binary conversion,
or palindrome checker.