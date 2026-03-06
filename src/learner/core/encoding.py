"""
Token encoding for TM snapshots.

Each tape cell becomes one token vector with slots:
  - val_oh:   3 bits (one-hot over {0, 1, _})
  - pos:      b bits (binary position encoding, ±1)
  - is_head:  1 bit  (1 if the TM head is here)
  - state:    S bits (one-hot over TM states)

Total dim d = 3 + b + 1 + S where b = ceil(log2(tape_len)).

The 3-class value encoding matches TMTransformerCategorical.SYM2IDX:
  "0" → [1, 0, 0]
  "1" → [0, 1, 0]
  "_" → [0, 0, 1]
"""
import math
import torch

SYM2IDX = {"0": 0, "1": 1, "_": 2}
IDX2SYM = {v: k for k, v in SYM2IDX.items()}


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
    d = 3 + b + 1 + S
    """
    n = max(len(tape), min_tape_len)
    padded = tape + [blank] * (n - len(tape))

    b = max(1, math.ceil(math.log2(n + 1)))
    S = len(state_index)

    tokens = []
    for j, sym in enumerate(padded):
        # 3-class one-hot value encoding
        val_oh = [0.0, 0.0, 0.0]
        val_oh[SYM2IDX.get(sym, 2)] = 1.0

        pos     = pos_encoding(j, b)
        is_head = [1.0 if j == head else 0.0]
        state_oh = [0.0] * S
        if state in state_index:
            state_oh[state_index[state]] = 1.0

        token = val_oh + pos + is_head + state_oh
        tokens.append(token)

    return torch.tensor(tokens, dtype=torch.float32)  # (n, d)


def decode_snapshot(
    output: torch.Tensor,
    state_index: dict[str, int],
    blank: str = "_",
) -> tuple[list[str], int, str]:
    """
    Decode model output tensor (n, d) back to (tape, head, state).

    val slots 0:3 -> argmax → symbol
    is_head       -> argmax over positions
    state         -> argmax over state slots
    """
    n, d = output.shape
    b = max(1, math.ceil(math.log2(n + 1)))
    S = len(state_index)

    # val is slots 0:3
    val_logits = output[:, 0:3]                        # (n, 3)
    tape = [IDX2SYM[val_logits[j].argmax().item()] for j in range(n)]

    # is_head is slot 3 + b
    head_slot = 3 + b
    head = int(output[:, head_slot].argmax().item())

    # state is slots (3 + b + 1) : (3 + b + 1 + S)
    state_start  = 3 + b + 1
    state_logits = output[:, state_start:state_start + S]
    state_avg    = state_logits.mean(dim=0)
    state_idx    = int(state_avg.argmax().item())

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
    inputs  = []
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