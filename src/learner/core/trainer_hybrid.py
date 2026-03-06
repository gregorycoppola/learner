"""
Hybrid loss trainer: weighted cross-entropy on carry examples.

For each batch:
  - scan_right examples: standard cross-entropy on all three heads
  - carry examples:      same cross-entropy, multiplied by carry_weight

Both losses are fully differentiable. Gradients flow through logits
for all examples. The carry_weight forces the model to treat carry
errors as proportionally more costly than scan_right errors.

Optional per-head weights (tape_weight, head_weight, state_weight)
address the imbalance between the tape head (n terms) vs head/state
(1 term each) in the CE sum.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Generator

from learner.core.machines import get_machine
from learner.core.encoding import batch_encode
from learner.core.model import TMTransformerCategorical
from learner.core.trainer_sft_grpo import (
    _sft_accuracy, make_state_index, _balanced_pairs
)


def _weighted_ce_loss(
    logits: dict,
    yb: torch.Tensor,
    state_index: dict,
    tape_weight: float = 1.0,
    head_weight: float = 1.0,
    state_weight: float = 1.0,
) -> torch.Tensor:
    """
    Cross-entropy loss on all three heads with optional per-head weights.
    """
    B, n, d = yb.shape
    b = max(1, math.ceil(math.log2(n + 1)))
    S = len(state_index)

    val_targets   = (yb[:, :, 0] > 0.5).long()
    head_slot     = 1 + b
    head_targets  = yb[:, :, head_slot].argmax(dim=1)
    state_start   = 1 + b + 1
    state_targets = yb[:, :, state_start:state_start + S].mean(dim=1).argmax(dim=1)

    vl = logits["value_logits"]   # (B, n, 3)
    hl = logits["head_logits"]    # (B, n)
    sl = logits["state_logits"]   # (B, S)

    loss_tape  = F.cross_entropy(vl.reshape(B * n, 3), val_targets.reshape(B * n))
    loss_head  = F.cross_entropy(hl, head_targets)
    loss_state = F.cross_entropy(sl, state_targets)

    return tape_weight * loss_tape + head_weight * loss_head + state_weight * loss_state


def hybrid_loss(
    logits: dict,
    yb: torch.Tensor,
    batch_pairs: list[dict],
    state_index: dict,
    carry_weight: float = 10.0,
    tape_weight: float = 1.0,
    head_weight: float = 1.0,
    state_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined differentiable loss:
      scan_right: CE (standard weight)
      carry:      CE * carry_weight
    """
    carry_mask = torch.tensor([
        p["state_before"] == "carry" for p in batch_pairs
    ])
    scan_mask = ~carry_mask

    n_carry = carry_mask.sum().item()
    n_scan  = scan_mask.sum().item()

    kwargs = dict(
        state_index=state_index,
        tape_weight=tape_weight,
        head_weight=head_weight,
        state_weight=state_weight,
    )

    if n_scan > 0:
        loss_scan = _weighted_ce_loss(
            {k: v[scan_mask] for k, v in logits.items()},
            yb[scan_mask],
            **kwargs,
        )
    else:
        loss_scan = torch.tensor(0.0)

    if n_carry > 0:
        loss_carry = carry_weight * _weighted_ce_loss(
            {k: v[carry_mask] for k, v in logits.items()},
            yb[carry_mask],
            **kwargs,
        )
    else:
        loss_carry = torch.tensor(0.0)

    if n_carry > 0 and n_scan > 0:
        total = loss_scan + loss_carry
    elif n_carry > 0:
        total = loss_carry
    else:
        total = loss_scan

    stats = {
        "loss_scan":  round(loss_scan.item(), 6),
        "loss_carry": round(loss_carry.item(), 6),
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
    tape_weight: float = 1.0,
    head_weight: float = 1.0,
    state_weight: float = 1.0,
    analyze_every: int = 5,
    analyze_samples: int = 500,
) -> Generator[dict, None, None]:
    """
    Hybrid weighted CE training. Yields one dict per epoch + periodic analysis.
    """
    from learner.core.analysis import analyze, make_breakdown_table

    torch.manual_seed(seed)
    state_index = make_state_index(machine_name)

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

    train_dl = DataLoader(
        TensorDataset(X_train, Y_train, torch.arange(len(train_pairs))),
        batch_size=batch_size, shuffle=True,
    )

    yield {
        "type":          "init",
        "machine":       machine_name,
        "n_train":       len(train_pairs),
        "n_val":         len(val_pairs),
        "carry_weight":  carry_weight,
        "tape_weight":   tape_weight,
        "head_weight":   head_weight,
        "state_weight":  state_weight,
        "analyze_every": analyze_every,
    }

    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_lscan  = 0.0
        epoch_lcarry = 0.0

        for xb, yb, idx_b in train_dl:
            batch_pairs = [train_pairs[i.item()] for i in idx_b]

            optimizer.zero_grad()
            logits = model(xb)
            loss, stats = hybrid_loss(
                logits, yb, batch_pairs, state_index,
                carry_weight=carry_weight,
                tape_weight=tape_weight,
                head_weight=head_weight,
                state_weight=state_weight,
            )
            loss.backward()
            optimizer.step()

            epoch_loss   += loss.item() * len(xb)
            epoch_lscan  += stats["loss_scan"]  * stats["n_scan"]
            epoch_lcarry += stats["loss_carry"] * stats["n_carry"]

        n = len(train_pairs)
        epoch_loss   /= n
        epoch_lscan  /= n
        epoch_lcarry /= n

        val_acc = _sft_accuracy(model, X_val, Y_val, state_index)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        yield {
            "type":       "epoch",
            "epoch":      epoch,
            "train_loss": round(epoch_loss, 6),
            "loss_scan":  round(epoch_lscan, 6),
            "loss_carry": round(epoch_lcarry, 6),
            "val_acc":    round(val_acc, 4),
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