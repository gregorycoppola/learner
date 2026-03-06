"""
Training loop for TM step prediction.

Generates synthetic pairs, encodes them, trains the transformer,
returns a loss curve and final accuracy.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from learner.core.machines import get_machine
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode
from learner.core.model import TMTransformer


def make_state_index(machine_name: str) -> dict[str, int]:
    tm = get_machine(machine_name)
    return {s: i for i, s in enumerate(sorted(tm.states))}


def train(
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    min_tape_len: int = 16,
    seed: int = 42,
) -> dict:
    """
    Train a transformer to predict one TM step.
    Returns a results dict with loss curve, accuracy, and the trained model.
    """
    torch.manual_seed(seed)

    # --- Data ---
    state_index = make_state_index(machine_name)
    pairs = generate_pairs(
        machine_name=machine_name,
        n_samples=n_samples,
        seed=seed,
    )

    # Train/val split
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    X_train, Y_train = batch_encode(train_pairs, state_index, min_tape_len)
    X_val, Y_val = batch_encode(val_pairs, state_index, min_tape_len)

    train_ds = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # --- Model ---
    d_input = X_train.shape[-1]
    model = TMTransformer(d_input=d_input, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- Training loop ---
    history = []

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

        # Val accuracy: check val slot and head slot predictions
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss = loss_fn(pred_val, Y_val).item()
            acc = _step_accuracy(pred_val, Y_val, state_index)

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(acc, 4),
        })

    return {
        "machine": machine_name,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "d_input": d_input,
        "history": history,
        "final_val_acc": history[-1]["val_acc"],
        "final_val_loss": history[-1]["val_loss"],
        "model": model,
        "state_index": state_index,
    }


def _step_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    state_index: dict[str, int],
) -> float:
    """
    Fraction of examples where the predicted tape + head + state
    all match the target exactly.
    """
    B, n, d = pred.shape
    b = max(1, __import__("math").ceil(__import__("math").log2(n + 1)))
    S = len(state_index)

    # val accuracy: slot 0, threshold 0.5
    pred_val = (pred[:, :, 0] > 0.5).int()
    true_val = (target[:, :, 0] > 0.5).int()
    val_match = (pred_val == true_val).all(dim=1)  # (B,)

    # head accuracy: argmax of slot (1+b)
    head_slot = 1 + b
    pred_head = pred[:, :, head_slot].argmax(dim=1)
    true_head = target[:, :, head_slot].argmax(dim=1)
    head_match = (pred_head == true_head)  # (B,)

    # state accuracy: argmax of mean over positions
    state_start = 1 + b + 1
    pred_state = pred[:, :, state_start:state_start + S].mean(dim=1).argmax(dim=1)
    true_state = target[:, :, state_start:state_start + S].mean(dim=1).argmax(dim=1)
    state_match = (pred_state == true_state)  # (B,)

    all_match = val_match & head_match & state_match
    return all_match.float().mean().item()