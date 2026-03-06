"""
Error analysis for trained TM step predictor.

Runs the model on a fresh set of pairs, extracts feature vectors
for each example, and builds a breakdown table of errors by feature.
"""
import math
import torch
from dataclasses import dataclass, field
from typing import Optional

from learner.core.machines import get_machine
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode, decode_snapshot


@dataclass
class StepResult:
    # Ground truth
    state_before: str
    state_after: str
    tape_before: list[str]
    tape_after: list[str]
    head_before: int
    head_after: int

    # Predictions
    pred_tape: list[str]
    pred_head: int
    pred_state: str

    # Error flags
    tape_correct: bool
    head_correct: bool
    state_correct: bool
    all_correct: bool

    # Features
    features: dict = field(default_factory=dict)


def extract_features(pair: dict, blank: str = "_") -> dict:
    """
    Compute interpretable feature vector for one step pair.
    These are the dimensions we think might correlate with errors.
    """
    tape_b = pair["tape_before"]
    tape_a = pair["tape_after"]
    head_b = pair["head_before"]
    head_a = pair["head_after"]
    state_b = pair["state_before"]
    state_a = pair["state_after"]

    # Tape content (ignoring blanks)
    real_bits = [b for b in tape_b if b != blank]
    n = len(real_bits)

    # Does the tape actually change this step?
    tape_changes = tape_b != tape_a

    # Is the head at the left boundary?
    head_at_left = (head_b == 0)

    # Is the head at the right end (pointing at blank or last real bit)?
    real_len = len([b for b in tape_b if b != blank])
    head_at_right = (head_b >= real_len - 1)

    # Is this the step that writes a new MSB?
    # i.e. carry state + head at position 0 + tape[0] == '1' (about to overflow)
    is_new_msb = (
        state_b == "carry"
        and head_b == 0
        and (len(tape_b) == 0 or tape_b[0] == "1")
    )

    # Are all real bits to the right of head equal to 1? (full carry ripple)
    bits_right_of_head = [
        tape_b[i] for i in range(head_b, len(tape_b))
        if tape_b[i] != blank
    ]
    all_ones_right = len(bits_right_of_head) > 0 and all(b == "1" for b in bits_right_of_head)

    # How many bits are in the number?
    tape_length = n

    # Head direction this step
    if head_a > head_b:
        direction = "right"
    elif head_a < head_b:
        direction = "left"
    else:
        direction = "stay"

    # Bit being read
    read_bit = tape_b[head_b] if head_b < len(tape_b) else blank

    # Number of carry steps remaining (how many 1s to the left of head in carry state)
    if state_b == "carry":
        ones_to_left = sum(
            1 for i in range(0, head_b)
            if i < len(tape_b) and tape_b[i] == "1"
        )
    else:
        ones_to_left = 0

    return {
        "state_before":     state_b,
        "state_after":      state_a,
        "tape_changes":     tape_changes,
        "head_at_left":     head_at_left,
        "head_at_right":    head_at_right,
        "is_new_msb":       is_new_msb,
        "all_ones_right":   all_ones_right,
        "tape_length":      tape_length,
        "direction":        direction,
        "read_bit":         read_bit,
        "ones_to_left":     ones_to_left,
    }


def analyze(
    model,
    state_index: dict[str, int],
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    min_tape_len: int = 16,
    seed: int = 999,  # different seed from training
) -> list[StepResult]:
    """
    Run the model on fresh pairs and collect per-step results with features.
    """
    from learner.core.machines import get_machine

    tm = get_machine(machine_name)
    pairs = generate_pairs(
        machine_name=machine_name,
        n_samples=n_samples,
        seed=seed,
    )

    X, Y = batch_encode(pairs, state_index, min_tape_len)

    model.eval()
    with torch.no_grad():
        pred = model(X)

    results = []
    for i, pair in enumerate(pairs):
        pred_tape, pred_head, pred_state = decode_snapshot(pred[i], state_index)
        true_tape = pair["tape_after"]
        true_head = pair["head_after"]
        true_state = pair["state_after"]

        # Trim to real tape length for comparison
        n_real = len([b for b in pair["tape_before"] if b != tm.blank])
        pred_tape_trimmed = pred_tape[:len(true_tape)]

        tape_correct  = pred_tape_trimmed == true_tape
        head_correct  = pred_head == true_head
        state_correct = pred_state == true_state
        all_correct   = tape_correct and head_correct and state_correct

        features = extract_features(pair)

        results.append(StepResult(
            state_before=pair["state_before"],
            state_after=pair["state_after"],
            tape_before=pair["tape_before"],
            tape_after=pair["tape_after"],
            head_before=pair["head_before"],
            head_after=pair["head_after"],
            pred_tape=pred_tape_trimmed,
            pred_head=pred_head,
            pred_state=pred_state,
            tape_correct=tape_correct,
            head_correct=head_correct,
            state_correct=state_correct,
            all_correct=all_correct,
            features=features,
        ))

    return results


def make_breakdown_table(results: list[StepResult]) -> list[dict]:
    """
    Break down accuracy by each feature value.
    Returns rows suitable for tabular display.
    """
    from collections import defaultdict

    # Features to break down by
    categorical_features = [
        "state_before",
        "tape_changes",
        "head_at_left",
        "head_at_right",
        "is_new_msb",
        "all_ones_right",
        "direction",
        "read_bit",
    ]

    rows = []

    # Overall first
    total = len(results)
    correct = sum(1 for r in results if r.all_correct)
    tape_c = sum(1 for r in results if r.tape_correct)
    head_c = sum(1 for r in results if r.head_correct)
    state_c = sum(1 for r in results if r.state_correct)
    rows.append({
        "feature": "OVERALL",
        "value": "all",
        "n": total,
        "acc": correct / total if total else 0,
        "tape_acc": tape_c / total if total else 0,
        "head_acc": head_c / total if total else 0,
        "state_acc": state_c / total if total else 0,
    })

    # Per feature breakdown
    for feat in categorical_features:
        buckets = defaultdict(list)
        for r in results:
            val = r.features.get(feat)
            buckets[str(val)].append(r)

        for val, bucket in sorted(buckets.items()):
            n = len(bucket)
            c = sum(1 for r in bucket if r.all_correct)
            tc = sum(1 for r in bucket if r.tape_correct)
            hc = sum(1 for r in bucket if r.head_correct)
            sc = sum(1 for r in bucket if r.state_correct)
            rows.append({
                "feature": feat,
                "value": val,
                "n": n,
                "acc": c / n if n else 0,
                "tape_acc": tc / n if n else 0,
                "head_acc": hc / n if n else 0,
                "state_acc": sc / n if n else 0,
            })

    # Tape length buckets
    length_buckets = defaultdict(list)
    for r in results:
        l = r.features.get("tape_length", 0)
        bucket_label = f"{(l // 4) * 4}-{(l // 4) * 4 + 3}"
        length_buckets[bucket_label].append(r)

    for val, bucket in sorted(length_buckets.items()):
        n = len(bucket)
        c = sum(1 for r in bucket if r.all_correct)
        tc = sum(1 for r in bucket if r.tape_correct)
        hc = sum(1 for r in bucket if r.head_correct)
        sc = sum(1 for r in bucket if r.state_correct)
        rows.append({
            "feature": "tape_length_bucket",
            "value": val,
            "n": n,
            "acc": c / n if n else 0,
            "tape_acc": tc / n if n else 0,
            "head_acc": hc / n if n else 0,
            "state_acc": sc / n if n else 0,
        })

    return rows