"""
Error analysis for trained TM step predictor.
Supports both regression (MSE) and categorical (GRPO) models.
"""
import math
import torch
from dataclasses import dataclass, field

from learner.core.machines import get_machine
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode, decode_snapshot


@dataclass
class StepResult:
    state_before: str
    state_after:  str
    tape_before:  list[str]
    tape_after:   list[str]
    head_before:  int
    head_after:   int
    pred_tape:    list[str]
    pred_head:    int
    pred_state:   str
    tape_correct:  bool
    head_correct:  bool
    state_correct: bool
    all_correct:   bool
    features:      dict = field(default_factory=dict)


def extract_features(pair: dict, blank: str = "_") -> dict:
    tape_b = pair["tape_before"]
    tape_a = pair["tape_after"]
    head_b = pair["head_before"]
    head_a = pair["head_after"]
    state_b = pair["state_before"]
    state_a = pair["state_after"]

    real_bits  = [b for b in tape_b if b != blank]
    real_len   = len(real_bits)
    tape_changes = tape_b != tape_a
    head_at_left  = (head_b == 0)
    head_at_right = (head_b >= real_len - 1)

    is_new_msb = (
        state_b == "carry"
        and head_b == 0
        and (len(tape_b) == 0 or tape_b[0] == "1")
    )

    bits_right = [tape_b[i] for i in range(head_b, len(tape_b))
                  if tape_b[i] != blank]
    all_ones_right = len(bits_right) > 0 and all(b == "1" for b in bits_right)

    if head_a > head_b:   direction = "right"
    elif head_a < head_b: direction = "left"
    else:                 direction = "stay"

    read_bit = tape_b[head_b] if head_b < len(tape_b) else blank

    ones_to_left = (
        sum(1 for i in range(0, head_b)
            if i < len(tape_b) and tape_b[i] == "1")
        if state_b == "carry" else 0
    )

    return {
        "state_before":   state_b,
        "state_after":    state_a,
        "tape_changes":   tape_changes,
        "head_at_left":   head_at_left,
        "head_at_right":  head_at_right,
        "is_new_msb":     is_new_msb,
        "all_ones_right": all_ones_right,
        "tape_length":    len(real_bits),
        "direction":      direction,
        "read_bit":       read_bit,
        "ones_to_left":   ones_to_left,
    }


def _decode_categorical(model, X: torch.Tensor, state_index: dict) -> list[tuple]:
    """Greedy decode from categorical model."""
    from learner.core.model import TMTransformerCategorical as Cat
    idx2sym   = {v: k for k, v in Cat.SYM2IDX.items()}
    idx2state = {v: k for k, v in state_index.items()}

    with torch.no_grad():
        logits = model(X)

    # Greedy: argmax each head
    tape_idx  = logits["value_logits"].argmax(dim=-1)  # (B, n)
    head_idx  = logits["head_logits"].argmax(dim=-1)   # (B,)
    state_idx = logits["state_logits"].argmax(dim=-1)  # (B,)

    results = []
    B = X.shape[0]
    for b in range(B):
        tape  = [idx2sym[tape_idx[b, i].item()] for i in range(X.shape[1])]
        head  = head_idx[b].item()
        state = idx2state.get(state_idx[b].item(), "unknown")
        results.append((tape, head, state))
    return results


def analyze(
    model,
    state_index: dict[str, int],
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    min_tape_len: int = 16,
    seed: int = 999,
    mode: str = "regression",  # "regression" or "categorical"
) -> list[StepResult]:
    tm    = get_machine(machine_name)
    pairs = generate_pairs(machine_name=machine_name,
                           n_samples=n_samples, seed=seed)
    X, Y  = batch_encode(pairs, state_index, min_tape_len)

    model.eval()

    if mode == "categorical":
        decoded = _decode_categorical(model, X, state_index)
    else:
        with torch.no_grad():
            pred = model(X)
        decoded = [
            decode_snapshot(pred[i], state_index)
            for i in range(len(pairs))
        ]

    results = []
    for i, pair in enumerate(pairs):
        pred_tape, pred_head, pred_state = decoded[i]
        true_tape  = pair["tape_after"]
        true_head  = pair["head_after"]
        true_state = pair["state_after"]

        pred_tape_trimmed = pred_tape[:len(true_tape)]

        tape_correct  = pred_tape_trimmed == true_tape
        head_correct  = pred_head == true_head
        state_correct = pred_state == true_state
        all_correct   = tape_correct and head_correct and state_correct

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
            features=extract_features(pair),
        ))

    return results


def make_breakdown_table(results: list[StepResult]) -> list[dict]:
    from collections import defaultdict

    categorical_features = [
        "state_before", "tape_changes", "head_at_left",
        "head_at_right", "is_new_msb", "all_ones_right",
        "direction", "read_bit",
    ]

    rows = []
    total   = len(results)
    correct = sum(1 for r in results if r.all_correct)
    tape_c  = sum(1 for r in results if r.tape_correct)
    head_c  = sum(1 for r in results if r.head_correct)
    state_c = sum(1 for r in results if r.state_correct)
    rows.append({
        "feature": "OVERALL", "value": "all", "n": total,
        "acc":       correct / total if total else 0,
        "tape_acc":  tape_c  / total if total else 0,
        "head_acc":  head_c  / total if total else 0,
        "state_acc": state_c / total if total else 0,
    })

    for feat in categorical_features:
        buckets = defaultdict(list)
        for r in results:
            buckets[str(r.features.get(feat))].append(r)
        for val, bucket in sorted(buckets.items()):
            n  = len(bucket)
            c  = sum(1 for r in bucket if r.all_correct)
            tc = sum(1 for r in bucket if r.tape_correct)
            hc = sum(1 for r in bucket if r.head_correct)
            sc = sum(1 for r in bucket if r.state_correct)
            rows.append({
                "feature": feat, "value": val, "n": n,
                "acc":       c  / n if n else 0,
                "tape_acc":  tc / n if n else 0,
                "head_acc":  hc / n if n else 0,
                "state_acc": sc / n if n else 0,
            })

    length_buckets = defaultdict(list)
    for r in results:
        l = r.features.get("tape_length", 0)
        label = f"{(l//4)*4}-{(l//4)*4+3}"
        length_buckets[label].append(r)
    for val, bucket in sorted(length_buckets.items()):
        n  = len(bucket)
        c  = sum(1 for r in bucket if r.all_correct)
        tc = sum(1 for r in bucket if r.tape_correct)
        hc = sum(1 for r in bucket if r.head_correct)
        sc = sum(1 for r in bucket if r.state_correct)
        rows.append({
            "feature": "tape_length_bucket", "value": val, "n": n,
            "acc":       c  / n if n else 0,
            "tape_acc":  tc / n if n else 0,
            "head_acc":  hc / n if n else 0,
            "state_acc": sc / n if n else 0,
        })

    return rows