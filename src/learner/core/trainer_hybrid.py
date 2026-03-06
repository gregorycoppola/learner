"""
Hybrid loss trainer: cross-entropy for scan_right, verifiable binary
penalty for carry.

For each batch:
  - scan_right examples: standard cross-entropy on all three heads
  - carry examples: run tm.step() on the greedy prediction, penalize
    hard if wrong (weighted by carry_weight)

No sampling, no K candidates. Single forward pass, deterministic.
The carry loss is: carry_weight * (1 - correct) per example.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Generator

from learner.core.machines import get_machine
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode
from learner.core.model import TMTransformerCategorical
from learner.core.trainer_sft_grpo import (
    _categorical_sft_loss, _sft_accuracy, make_state_index, _balanced_pairs
)


def _greedy_decode(logits: dict, state_index: dict) -> tuple:
    """
    Greedy decode logits to (tape_syms, head_pos, state_str).
    Returns lists/ints for a single example (logits are unbatched).
    """
    from learner.core.model import TMTransformerCategorical as Cat
    idx2sym   = {v: k for k, v in Cat.SYM2IDX.items()}
    idx2state = {v: k for k, v in state_index.items()}

    tape  = [idx2sym[i.item()] for i in logits["value_logits"].argmax(dim=-1)]
    head  = logits["head_logits"].argmax(dim=-1).item()
    state = idx2state.get(logits["state_logits"].argmax(dim=-1).item(), "unknown")
    return tape, head, state


def _verify_carry_batch(
    logits: dict,
    batch_pairs: list[dict],
    carry_mask: torch.Tensor,
    state_index: dict,
    tm,
) -> torch.Tensor:
    """
    For carry examples, run tm.step() on greedy prediction.
    Returns per-example binary correct tensor (B,) float.
    Only meaningful where carry_mask is True.
    """
    from learner.core.model import TMTransformerCategorical as Cat
    idx2sym   = {v: k for k, v in Cat.SYM2IDX.items()}
    idx2state = {v: k for k, v in state_index.items()}

    B = logits["value_logits"].shape[0]
    correct = torch.zeros(B)

    tape_preds  = logits["value_logits"].argmax(dim=-1)   # (B, n)
    head_preds  = logits["head_logits"].argmax(dim=-1)    # (B,)
    state_preds = logits["state_logits"].argmax(dim=-1)   # (B,)

    for b in range(B):
        if not carry_mask[b]:
            continue

        pair = batch_pairs[b]
        true_tape, true_head, true_state, _ = tm.step(
            pair["tape_before"], pair["head_before"], pair["state_before"]
        )

        pred_tape  = [idx2sym[tape_preds[b, i].item()]
                      for i in range(tape_preds.shape[1])]
        pred_head  = head_preds[b].item()
        pred_state = idx2state.get(state_preds[b].item(), "unknown")

        tape_ok  = pred_tape[:len(true_tape)] == true_tape
        head_ok  = pred_head == true_head
        state_ok = pred_state == true_state

        if tape_ok and head_ok and state_ok:
            correct[b] = 1.0

    return correct


def hybrid_loss(
    logits: dict,
    yb: torch.Tensor,
    batch_pairs: list[dict],
    state_index: dict,
    tm,
    carry_weight: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss:
      scan_right: cross-entropy (standard)
      carry:      carry_weight * (1 - correct)  (verifiable binary)

    Returns (loss, stats_dict).
    """
    B = yb.shape[0]

    # Identify carry examples
    carry_mask = torch.tensor([
        p["state_before"] == "carry" for p in batch_pairs
    ])
    scan_mask = ~carry_mask

    n_carry = carry_mask.sum().item()
    n_scan  = scan_mask.sum().item()

    # ── Cross-entropy loss on scan_right examples ──
    if n_scan > 0:
        loss_scan = _categorical_sft_loss(
            {k: v[scan_mask] for k, v in logits.items()},
            yb[scan_mask],
            state_index,
        )
    else:
        loss_scan = torch.tensor(0.0)

    # ── Verifiable binary loss on carry examples ──
    if n_carry > 0:
        correct = _verify_carry_batch(
            {k: v for k, v in logits.items()},
            batch_pairs, carry_mask, state_index, tm,
        )
        # Loss is high when wrong, zero when correct
        carry_losses = carry_weight * (1.0 - correct[carry_mask])
        loss_carry   = carry_losses.mean()
        carry_acc    = correct[carry_mask].mean().item()
    else:
        loss_carry = torch.tensor(0.0)
        carry_acc  = 0.0

    # Weighted combination
    if n_carry > 0 and n_scan > 0:
        total = loss_scan + loss_carry
    elif n_carry > 0:
        total = loss_carry
    else:
        total = loss_scan

    stats = {
        "loss_scan":  round(loss_scan.item(), 6),
        "loss_carry": round(loss_carry.item(), 6),
        "carry_acc":  round(carry_acc, 4),
        "n_carry":    int(n_carry),
        "n_scan":     int(n_scan),
    }

    return total, stats


def train_hybrid_streaming(
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    n_epochs: int = 100000,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    min_tape_len: int = 16,
    seed: int = 42,
    carry_weight: float = 10.0,
    analyze_every: int = 5,
    analyze_samples: int = 500,
) -> Generator[dict, None, None]:
    """
    Hybrid loss training. Yields one dict per epoch + periodic analysis.
    """
    from learner.core.analysis import analyze, make_breakdown_table

    torch.manual_seed(seed)
    state_index = make_state_index(machine_name)
    tm          = get_machine(machine_name)

    pairs = _balanced_pairs(machine_name, n_samples, seed)
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    X_train, Y_train = batch_encode(train_pairs, state_index, min_tape_len)
    X_val,   Y_val   = batch_encode(val_pairs,   state_index, min_tape_len)

    n_tape   = X_train.shape[1]
    n_states = len(state_index)
    d_input  = X_train.shape[2]

    model = TMTransformerCategorical(
        d_input=d_input, n_tape=n_tape, n_states=n_states,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Keep indices so we can look up raw pairs per batch
    train_dl = DataLoader(
        TensorDataset(X_train, Y_train, torch.arange(len(train_pairs))),
        batch_size=batch_size, shuffle=True,
    )

    yield {
        "type":          "init",
        "machine":       machine_name,
        "n_train":       len(train_pairs),
        "n_val":         len(val_pairs),
        "d_input":       d_input,
        "n_epochs":      n_epochs,
        "carry_weight":  carry_weight,
        "analyze_every": analyze_every,
    }

    best_val_acc  = 0.0
    best_carry_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss      = 0.0
        epoch_carry_acc = 0.0
        epoch_carry_n   = 0
        n_batches       = 0

        for xb, yb, idx_b in train_dl:
            batch_pairs = [train_pairs[i.item()] for i in idx_b]

            optimizer.zero_grad()
            logits = model(xb)
            loss, stats = hybrid_loss(
                logits, yb, batch_pairs, state_index, tm,
                carry_weight=carry_weight,
            )
            loss.backward()
            optimizer.step()

            epoch_loss      += loss.item() * len(xb)
            epoch_carry_acc += stats["carry_acc"] * stats["n_carry"]
            epoch_carry_n   += stats["n_carry"]
            n_batches       += 1

        epoch_loss      /= len(train_pairs)
        epoch_carry_acc  = epoch_carry_acc / max(epoch_carry_n, 1)

        val_acc = _sft_accuracy(model, X_val, Y_val, state_index)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if epoch_carry_acc > best_carry_acc:
            best_carry_acc = epoch_carry_acc

        yield {
            "type":            "epoch",
            "epoch":           epoch,
            "train_loss":      round(epoch_loss, 6),
            "train_carry_acc": round(epoch_carry_acc, 4),
            "val_acc":         round(val_acc, 4),
        }

        if epoch % analyze_every == 0:
            results = analyze(
                model=model, state_index=state_index,
                machine_name=machine_name, n_samples=analyze_samples,
                min_tape_len=min_tape_len, seed=seed + epoch,
                mode="categorical",
            )
            table   = make_breakdown_table(results)
            overall = sum(1 for r in results if r.all_correct) / len(results)
            cats    = {
                row["value"]: row["acc"]
                for row in table if row["feature"] == "state_before"
            }
            yield {
                "type":             "analysis",
                "epoch":            epoch,
                "overall_acc":      round(overall, 4),
                "category_summary": cats,
                "table":            table,
            }

    yield {"type": "done", "best_val_acc": round(best_val_acc, 4)}