"""
Training loop for TM step prediction with dynamic curriculum.

After each analysis checkpoint, per-example weights are updated:
- Examples the model got wrong get upweighted
- Examples the model got right get downweighted
- This shifts training budget toward hard examples in real time
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Generator

from learner.core.machines import get_machine
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode
from learner.core.model import TMTransformer


def make_state_index(machine_name: str) -> dict[str, int]:
    tm = get_machine(machine_name)
    return {s: i for i, s in enumerate(sorted(tm.states))}


def _balanced_pairs(machine_name: str, n_samples: int, seed: int) -> list[dict]:
    """
    Generate pairs with balanced carry/scan_right representation.
    We generate a large pool then sample 50/50 by state.
    """
    import random
    rng = random.Random(seed)

    # Generate a big pool to ensure enough carry examples
    pool = generate_pairs(machine_name=machine_name, n_samples=n_samples * 4, seed=seed)

    carry_pairs = [p for p in pool if p["state_before"] == "carry"]
    other_pairs = [p for p in pool if p["state_before"] != "carry"]

    # 50/50 split, capped at available carry examples
    n_each = min(n_samples // 2, len(carry_pairs))
    rng.shuffle(carry_pairs)
    rng.shuffle(other_pairs)

    balanced = carry_pairs[:n_each] + other_pairs[:n_each]
    rng.shuffle(balanced)
    return balanced


def _make_dataloader(
    X: torch.Tensor,
    Y: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
) -> DataLoader:
    """Build a DataLoader with weighted sampling."""
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


def _per_example_correct(
    pred: torch.Tensor,
    target: torch.Tensor,
    state_index: dict[str, int],
) -> torch.Tensor:
    """
    Returns a boolean tensor (B,) — True if that example is fully correct.
    """
    B, n, d = pred.shape
    b = max(1, math.ceil(math.log2(n + 1)))
    S = len(state_index)

    pred_val  = (pred[:, :, 0] > 0.5).int()
    true_val  = (target[:, :, 0] > 0.5).int()
    val_match = (pred_val == true_val).all(dim=1)

    head_slot  = 1 + b
    pred_head  = pred[:, :, head_slot].argmax(dim=1)
    true_head  = target[:, :, head_slot].argmax(dim=1)
    head_match = (pred_head == true_head)

    state_start = 1 + b + 1
    pred_state  = pred[:, :, state_start:state_start + S].mean(dim=1).argmax(dim=1)
    true_state  = target[:, :, state_start:state_start + S].mean(dim=1).argmax(dim=1)
    state_match = (pred_state == true_state)

    return val_match & head_match & state_match


def _step_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    state_index: dict[str, int],
) -> float:
    return _per_example_correct(pred, target, state_index).float().mean().item()


def _update_weights(
    weights: torch.Tensor,
    correct: torch.Tensor,
    hard_multiplier: float = 4.0,
    easy_multiplier: float = 0.5,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """
    Upweight wrong examples, downweight correct ones.
    Normalise so weights sum to len(weights).
    """
    new_weights = weights.clone()
    new_weights[~correct] *= hard_multiplier
    new_weights[correct]  *= easy_multiplier
    new_weights = new_weights.clamp(min=min_weight)
    # Renormalise
    new_weights = new_weights / new_weights.sum() * len(new_weights)
    return new_weights


def train_streaming(
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    min_tape_len: int = 16,
    seed: int = 42,
    analyze_every: int = 5,
    analyze_samples: int = 500,
    hard_multiplier: float = 4.0,
    easy_multiplier: float = 0.5,
    mastered_threshold: float = 0.95,
) -> Generator[dict, None, None]:
    """
    Train with dynamic curriculum and yield updates every epoch.
    Every analyze_every epochs: run analysis, update example weights.
    """
    from learner.core.analysis import analyze, make_breakdown_table

    torch.manual_seed(seed)
    state_index = make_state_index(machine_name)

    # Balanced initial dataset
    pairs = _balanced_pairs(machine_name, n_samples, seed)

    split       = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    X_train, Y_train = batch_encode(train_pairs, state_index, min_tape_len)
    X_val,   Y_val   = batch_encode(val_pairs,   state_index, min_tape_len)

    d_input = X_train.shape[-1]

    # Start with uniform weights
    weights = torch.ones(len(train_pairs))

    model     = TMTransformer(d_input=d_input, d_model=d_model,
                              n_layers=n_layers, n_heads=n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    train_dl = _make_dataloader(X_train, Y_train, weights, batch_size)

    yield {
        "type": "init",
        "machine":        machine_name,
        "n_train":        len(train_pairs),
        "n_val":          len(val_pairs),
        "d_input":        d_input,
        "n_epochs":       n_epochs,
        "n_samples":      n_samples,
        "analyze_every":  analyze_every,
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_pairs)

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss = loss_fn(pred_val, Y_val).item()
            acc      = _step_accuracy(pred_val, Y_val, state_index)

        # Weight summary for this epoch
        w_min  = round(weights.min().item(), 3)
        w_max  = round(weights.max().item(), 3)
        w_mean = round(weights.mean().item(), 3)

        yield {
            "type":       "epoch",
            "epoch":      epoch,
            "train_loss": round(epoch_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(acc, 4),
            "w_min":      w_min,
            "w_max":      w_max,
            "w_mean":     w_mean,
        }

        # Analysis + weight update every analyze_every epochs
        if epoch % analyze_every == 0:
            # --- Per-example correctness on training set ---
            model.eval()
            with torch.no_grad():
                pred_train = model(X_train)
                correct    = _per_example_correct(pred_train, Y_train, state_index)

            # Update weights
            weights  = _update_weights(
                weights, correct,
                hard_multiplier=hard_multiplier,
                easy_multiplier=easy_multiplier,
            )
            train_dl = _make_dataloader(X_train, Y_train, weights, batch_size)

            # How many examples are effectively "mastered"
            n_mastered = int(correct.sum().item())
            n_total    = len(correct)

            # Full analysis on fresh held-out samples
            results = analyze(
                model=model,
                state_index=state_index,
                machine_name=machine_name,
                n_samples=analyze_samples,
                min_tape_len=min_tape_len,
                seed=seed + epoch,
            )
            table   = make_breakdown_table(results)
            overall = sum(1 for r in results if r.all_correct) / len(results)

            # Per-category mastery summary
            category_summary = {}
            for row in table:
                if row["feature"] == "state_before":
                    category_summary[row["value"]] = row["acc"]

            yield {
                "type":             "analysis",
                "epoch":            epoch,
                "overall_acc":      round(overall, 4),
                "n_mastered":       n_mastered,
                "n_total":          n_total,
                "weight_min":       round(weights.min().item(), 3),
                "weight_max":       round(weights.max().item(), 3),
                "category_summary": category_summary,
                "table":            table,
            }

    yield {"type": "done"}


def _train_and_return(
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    min_tape_len: int = 16,
    seed: int = 42,
):
    """Train silently and return (model, state_index)."""
    torch.manual_seed(seed)
    state_index  = make_state_index(machine_name)
    pairs        = _balanced_pairs(machine_name, n_samples, seed)
    split        = int(0.9 * len(pairs))
    train_pairs  = pairs[:split]

    X_train, Y_train = batch_encode(train_pairs, state_index, min_tape_len)
    weights          = torch.ones(len(train_pairs))
    train_dl         = _make_dataloader(X_train, Y_train, weights, batch_size)

    d_input   = X_train.shape[-1]
    model     = TMTransformer(d_input=d_input, d_model=d_model,
                              n_layers=n_layers, n_heads=n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    return model, state_index