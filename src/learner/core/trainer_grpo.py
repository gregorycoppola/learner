"""
GRPO training loop for TM step prediction.
"""
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Generator

from learner.core.machines import get_machine, get_hard_states
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode
from learner.core.model import TMTransformerCategorical
from learner.core.grpo import GRPOConfig, verify_batch, grpo_loss


def make_state_index(machine_name: str) -> dict[str, int]:
    tm = get_machine(machine_name)
    return {s: i for i, s in enumerate(sorted(tm.states))}


def _balanced_pairs(machine_name: str, n_samples: int, seed: int) -> list[dict]:
    import random
    rng   = random.Random(seed)
    pool  = generate_pairs(machine_name=machine_name,
                           n_samples=n_samples * 4, seed=seed)
    hard  = get_hard_states(machine_name)
    heavy = [p for p in pool if p["state_before"] in hard]
    easy  = [p for p in pool if p["state_before"] not in hard]
    n_each = min(n_samples // 2, len(heavy))
    rng.shuffle(heavy)
    rng.shuffle(easy)
    out = heavy[:n_each] + easy[:n_each]
    rng.shuffle(out)
    return out


def _accuracy_from_rewards(rewards: torch.Tensor) -> float:
    return (rewards.max(dim=1).values > 0).float().mean().item()


def train_grpo_streaming(
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    min_tape_len: int = 16,
    seed: int = 42,
    K: int = 8,
    kl_coef: float = 0.01,
    analyze_every: int = 5,
    analyze_samples: int = 500,
) -> Generator[dict, None, None]:
    from learner.core.analysis import analyze, make_breakdown_table

    torch.manual_seed(seed)
    state_index = make_state_index(machine_name)
    tm          = get_machine(machine_name)
    config      = GRPOConfig(K=K, lr=lr, kl_coef=kl_coef)

    pairs = _balanced_pairs(machine_name, n_samples, seed)
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    X_train, Y_train = batch_encode(train_pairs, state_index, min_tape_len)
    X_val,   Y_val   = batch_encode(val_pairs,   state_index, min_tape_len)

    n_tape   = X_train.shape[1]
    n_states = len(state_index)
    d_input  = X_train.shape[2]

    model     = TMTransformerCategorical(
        d_input=d_input, n_tape=n_tape, n_states=n_states,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl = DataLoader(
        TensorDataset(X_train, torch.arange(len(train_pairs))),
        batch_size=batch_size, shuffle=True,
    )

    yield {
        "type":          "init",
        "machine":       machine_name,
        "n_train":       len(train_pairs),
        "n_val":         len(val_pairs),
        "d_input":       d_input,
        "n_tape":        n_tape,
        "n_states":      n_states,
        "n_epochs":      n_epochs,
        "K":             K,
        "analyze_every": analyze_every,
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_reward = 0.0
        n_batches    = 0

        for xb, idx_b in train_dl:
            batch_pairs  = [train_pairs[i.item()] for i in idx_b]
            tape_before  = [p["tape_before"]  for p in batch_pairs]
            head_before  = [p["head_before"]  for p in batch_pairs]
            state_before = [p["state_before"] for p in batch_pairs]

            with torch.no_grad():
                samples = model.sample(xb, K=K)

            rewards = verify_batch(
                samples["tape_samples"], samples["head_samples"],
                samples["state_samples"],
                tape_before, head_before, state_before,
                state_index, tm,
            )

            old_log_probs = samples["log_probs"].detach()

            B, K_actual, n = samples["tape_samples"].shape
            xb_expanded = xb.unsqueeze(1).expand(B, K_actual, n, d_input)\
                           .reshape(B * K_actual, n, d_input)

            tape_idx  = samples["tape_samples"].reshape(B * K_actual, n)
            head_idx  = samples["head_samples"].reshape(B * K_actual)
            state_idx = samples["state_samples"].reshape(B * K_actual)

            log_probs_flat = model.log_prob_of(
                xb_expanded, tape_idx, head_idx, state_idx
            ).reshape(B, K_actual)

            loss, stats = grpo_loss(log_probs_flat, old_log_probs, rewards, config)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss   += stats["total_loss"]
            epoch_reward += stats["mean_reward"]
            n_batches    += 1

        avg_loss   = epoch_loss   / max(n_batches, 1)
        avg_reward = epoch_reward / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_samples = model.sample(X_val, K=K)
            val_rewards = verify_batch(
                val_samples["tape_samples"], val_samples["head_samples"],
                val_samples["state_samples"],
                [p["tape_before"]  for p in val_pairs],
                [p["head_before"]  for p in val_pairs],
                [p["state_before"] for p in val_pairs],
                state_index, tm,
            )
        val_acc = _accuracy_from_rewards(val_rewards)

        yield {
            "type":         "epoch",
            "epoch":        epoch,
            "train_loss":   round(avg_loss,   6),
            "train_reward": round(avg_reward, 4),
            "val_acc":      round(val_acc,    4),
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
            category_summary = {
                row["value"]: row["acc"]
                for row in table if row["feature"] == "state_before"
            }
            yield {
                "type":             "analysis",
                "epoch":            epoch,
                "overall_acc":      round(overall, 4),
                "category_summary": category_summary,
                "table":            table,
            }

    yield {"type": "done"}