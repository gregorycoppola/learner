"""
Token encoding for TM snapshots.

Each tape cell becomes one token vector with slots:
  - val:      1 bit  (the cell's current value, 0 or 1)
  - pos:      b bits (binary position encoding, ±1)
  - is_head:  1 bit  (1 if the TM head is here)
  - state:    S bits (one-hot over TM states)

This follows the typed proposition encoding in the paper (Section 3).
Total dim d = 1 + b + 1 + S where b = ceil(log2(tape_len)).
"""
import math
import torch


def pos_encoding(j: int, b: int) -> list[float]:
    """Binary positional encoding for index j using b bits, values in {-1, +1}."""
    bits = []
    for k in range(b):
        bits.append(1.0 if (j >> k) & 1 else -1.0)
    return bits


def encode_snapshot(
    tape: list[str],
    head: int,
    state: str,
    state_index: dict[str, int],
    blank: str = "_",
    min_tape_len: int = 16,
) -> torch.Tensor:
    """
    Encode a TM snapshot as a tensor of shape (n, d).

    n = tape length (padded to min_tape_len)
    d = 1 + b + 1 + S
    """
    # Pad tape
    n = max(len(tape), min_tape_len)
    padded = tape + [blank] * (n - len(tape))

    b = max(1, math.ceil(math.log2(n + 1)))
    S = len(state_index)
    d = 1 + b + 1 + S

    tokens = []
    for j, sym in enumerate(padded):
        val = [1.0 if sym == "1" else 0.0]
        pos = pos_encoding(j, b)
        is_head = [1.0 if j == head else 0.0]
        state_oh = [0.0] * S
        if state in state_index:
            state_oh[state_index[state]] = 1.0
        token = val + pos + is_head + state_oh
        tokens.append(token)

    return torch.tensor(tokens, dtype=torch.float32)  # (n, d)


def decode_snapshot(
    output: torch.Tensor,
    state_index: dict[str, int],
    blank: str = "_",
) -> tuple[list[str], int, str]:
    """
    Decode model output tensor (n, d) back to (tape, head, state).

    val slot  -> threshold at 0.5
    is_head   -> argmax over positions
    state     -> argmax over state slots
    """
    n, d = output.shape
    b = max(1, math.ceil(math.log2(n + 1)))
    S = len(state_index)

    # val is slot 0
    vals = output[:, 0]
    tape = ["1" if v.item() > 0.5 else "0" for v in vals]

    # is_head is slot 1 + b
    head_slot = 1 + b
    head = int(output[:, head_slot].argmax().item())

    # state is slots (1 + b + 1) : (1 + b + 1 + S)
    state_start = 1 + b + 1
    state_logits = output[:, state_start:state_start + S]
    # average over positions (all tokens carry the same state signal)
    state_avg = state_logits.mean(dim=0)
    state_idx = int(state_avg.argmax().item())

    index_state = {v: k for k, v in state_index.items()}
    state = index_state.get(state_idx, "unknown")

    return tape, head, state


def batch_encode(
    pairs: list[dict],
    state_index: dict[str, int],
    min_tape_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a list of step pairs into (inputs, targets) tensors.

    Returns:
      inputs:  (B, n, d)
      targets: (B, n, d)
    """
    inputs = []
    targets = []

    for pair in pairs:
        x = encode_snapshot(
            pair["tape_before"], pair["head_before"], pair["state_before"],
            state_index, min_tape_len=min_tape_len,
        )
        y = encode_snapshot(
            pair["tape_after"], pair["head_after"], pair["state_after"],
            state_index, min_tape_len=min_tape_len,
        )
        inputs.append(x)
        targets.append(y)

    return torch.stack(inputs), torch.stack(targets)