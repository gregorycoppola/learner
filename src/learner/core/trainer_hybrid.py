"""
Hybrid loss trainer: weighted cross-entropy on hard-state examples.

Both losses are fully differentiable. The hard_weight multiplier
forces the model to prioritize hard-state transitions (e.g. carry,
borrow) that are minority classes in the natural distribution.

Uses the 3-class value encoding from encoding.py so that blank
cells are distinguishable from zero cells in both input and target.

Stops early when val_acc reaches 1.0.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Generator

from learner.core.encoding import batch_encode
from learner.core.model import TMTransformerCategorical
from learner.core.machines import get_hard_states
from learner.core.trainer_sft_grpo import (
    _sft_accuracy, _extract_targets, make_state_index, _balanced_pairs
)


def _weighted_ce_loss(
    logits: dict,
    yb: torch.Tensor,
    state_index: dict,
    tape_weight: float = 1.0,
    head_weight: float = 1.0,
    state_weight: float = 1.0,
) -> torch.Tensor:
    B, n, d = yb.shape
    val_targets, head_targets, state_targets = _extract_targets(yb, state_index)

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
    hard_states: set[str],
    carry_weight: float = 10.0,
    tape_weight: float = 1.0,
    head_weight: float = 1.0,
    state_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    hard_mask = torch.tensor([p["state_before"] in hard_states for p in batch_pairs])
    easy_mask = ~hard_mask
    n_hard    = hard_mask.sum().item()
    n_easy    = easy_mask.sum().item()

    kwargs = dict(
        state_index=state_index,
        tape_weight=tape_weight,
        head_weight=head_weight,
        state_weight=state_weight,
    )

    loss_easy = _weighted_ce_loss(
        {k: v[easy_mask] for k, v in logits.items()}, yb[easy_mask], **kwargs
    ) if n_easy > 0 else torch.tensor(0.0)

    loss_hard = carry_weight * _weighted_ce_loss(
        {k: v[hard_mask] for k, v in logits.items()}, yb[hard_mask], **kwargs
    ) if n_hard > 0 else torch.tensor(0.0)

    total = loss_easy + loss_hard

    return total, {
        "loss_easy": round(loss_easy.item(), 6),
        "loss_hard": round(loss_hard.item(), 6),
        "n_hard":    int(n_hard),
        "n_easy":    int(n_easy),
    }


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
    from learner.core.analysis import analyze, make_breakdown_table

    torch.manual_seed(seed)
    state_index = make_state_index(machine_name)
    hard_states = get_hard_states(machine_name)

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
        "hard_states":   sorted(hard_states),
        "tape_weight":   tape_weight,
        "head_weight":   head_weight,
        "state_weight":  state_weight,
        "analyze_every": analyze_every,
    }

    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_leasy = 0.0
        epoch_lhard = 0.0

        for xb, yb, idx_b in train_dl:
            batch_pairs = [train_pairs[i.item()] for i in idx_b]

            optimizer.zero_grad()
            logits = model(xb)
            loss, stats = hybrid_loss(
                logits, yb, batch_pairs, state_index,
                hard_states=hard_states,
                carry_weight=carry_weight,
                tape_weight=tape_weight,
                head_weight=head_weight,
                state_weight=state_weight,
            )
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()        * len(xb)
            epoch_leasy += stats["loss_easy"] * stats["n_easy"]
            epoch_lhard += stats["loss_hard"] * stats["n_hard"]

        n = len(train_pairs)
        epoch_loss  /= n
        epoch_leasy /= n
        epoch_lhard /= n

        val_acc = _sft_accuracy(model, X_val, Y_val, state_index)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        yield {
            "type":       "epoch",
            "epoch":      epoch,
            "train_loss": round(epoch_loss,  6),
            "loss_easy":  round(epoch_leasy, 6),
            "loss_hard":  round(epoch_lhard, 6),
            "val_acc":    round(val_acc, 4),
        }

        # Early stopping on perfect accuracy
        if val_acc >= 1.0:
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
            yield {"type": "done", "best_val_acc": 1.0, "stopped_early": True, "epoch": epoch}
            return

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

    yield {"type": "done", "best_val_acc": round(best_val_acc, 4), "stopped_early": False}