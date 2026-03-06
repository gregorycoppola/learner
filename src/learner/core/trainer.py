"""
Training loop for TM step prediction.

Generates synthetic pairs, encodes them, trains the transformer,
yields epoch results as they complete (for streaming).
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Generator

from learner.core.machines import get_machine
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode
from learner.core.model import TMTransformer


def make_state_index(machine_name: str) -> dict[str, int]:
    tm = get_machine(machine_name)
    return {s: i for i, s in enumerate(sorted(tm.states))}


def _build_data_and_model(
    machine_name, n_samples, batch_size,
    lr, d_model, n_layers, n_heads, min_tape_len, seed
):
    """Shared setup: data, model, optimizer, loss."""
    torch.manual_seed(seed)

    state_index = make_state_index(machine_name)
    pairs = generate_pairs(machine_name=machine_name, n_samples=n_samples, seed=seed)

    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    X_train, Y_train = batch_encode(train_pairs, state_index, min_tape_len)
    X_val,   Y_val   = batch_encode(val_pairs,   state_index, min_tape_len)

    train_ds = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    d_input = X_train.shape[-1]
    model = TMTransformer(
        d_input=d_input, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    return (
        model, optimizer, loss_fn,
        train_dl, X_val, Y_val,
        train_pairs, val_pairs,
        d_input, state_index,
    )


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
) -> Generator[dict, None, None]:
    """
    Train and yield one dict per epoch.
    Every analyze_every epochs also yields a full error breakdown.
    """
    from learner.core.analysis import analyze, make_breakdown_table

    (
        model, optimizer, loss_fn,
        train_dl, X_val, Y_val,
        train_pairs, val_pairs,
        d_input, state_index,
    ) = _build_data_and_model(
        machine_name, n_samples, batch_size,
        lr, d_model, n_layers, n_heads, min_tape_len, seed,
    )

    yield {
        "type": "init",
        "machine": machine_name,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "d_input": d_input,
        "n_epochs": n_epochs,
        "n_samples": n_samples,
        "analyze_every": analyze_every,
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
            acc = _step_accuracy(pred_val, Y_val, state_index)

        yield {
            "type": "epoch",
            "epoch": epoch,
            "train_loss": round(epoch_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(acc, 4),
        }

        # Full error breakdown every analyze_every epochs
        if epoch % analyze_every == 0:
            results = analyze(
                model=model,
                state_index=state_index,
                machine_name=machine_name,
                n_samples=analyze_samples,
                min_tape_len=min_tape_len,
                seed=seed + epoch,  # fresh sample each time
            )
            table = make_breakdown_table(results)
            overall = sum(1 for r in results if r.all_correct) / len(results)
            yield {
                "type": "analysis",
                "epoch": epoch,
                "overall_acc": round(overall, 4),
                "table": table,
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
    """Train silently and return (model, state_index) for analysis."""
    (
        model, optimizer, loss_fn,
        train_dl, X_val, Y_val,
        train_pairs, val_pairs,
        d_input, state_index,
    ) = _build_data_and_model(
        machine_name, n_samples, batch_size,
        lr, d_model, n_layers, n_heads, min_tape_len, seed,
    )

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, state_index


def _step_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    state_index: dict[str, int],
) -> float:
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

    return (val_match & head_match & state_match).float().mean().item()