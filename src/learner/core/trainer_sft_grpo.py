"""
Two-phase SFT -> GRPO training pipeline.

Phase 1 (SFT): supervised cross-entropy on the categorical model until
scan_right accuracy hits sft_threshold. Saves checkpoint.

Phase 2 (GRPO): loads SFT checkpoint, switches to GRPO with
TM verifier. Now that the model is warm, carry steps will
occasionally be correct, giving GRPO a positive reward signal
to bootstrap from.

Streams events throughout both phases.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Generator

from learner.core.machines import get_machine
from learner.core.data import generate_pairs
from learner.core.encoding import batch_encode
from learner.core.model import TMTransformerCategorical
from learner.core.grpo import GRPOConfig, verify_batch, grpo_loss
from learner.core.checkpoint import save as save_checkpoint, models_dir
from learner.core.trainer_grpo import (
    make_state_index, _balanced_pairs, _accuracy_from_rewards
)


def _extract_targets(Y: torch.Tensor, state_index: dict):
    """
    Extract (val_targets, head_targets, state_targets) from encoded Y.

    Slot layout (new 3-class encoding):
      [val_oh(3), pos(b), is_head(1), state_oh(S)]
      d = 3 + b + 1 + S

    val_targets:   (B, n)  class in {0='0', 1='1', 2='_'}
    head_targets:  (B,)    position index
    state_targets: (B,)    state index
    """
    B, n, d = Y.shape
    b = max(1, math.ceil(math.log2(n + 1)))
    S = len(state_index)

    val_targets   = Y[:, :, 0:3].argmax(dim=-1)                              # (B, n)

    head_slot     = 3 + b
    head_targets  = Y[:, :, head_slot].argmax(dim=1)                          # (B,)

    state_start   = 3 + b + 1
    state_targets = Y[:, :, state_start:state_start + S].mean(dim=1).argmax(dim=1)  # (B,)

    return val_targets, head_targets, state_targets


def _categorical_sft_loss(logits: dict, Y: torch.Tensor, state_index: dict) -> torch.Tensor:
    """Cross-entropy loss on each categorical head."""
    B, n, d = Y.shape
    val_targets, head_targets, state_targets = _extract_targets(Y, state_index)

    vl = logits["value_logits"]   # (B, n, 3)
    hl = logits["head_logits"]    # (B, n)
    sl = logits["state_logits"]   # (B, S)

    loss_val   = F.cross_entropy(vl.reshape(B * n, 3), val_targets.reshape(B * n))
    loss_head  = F.cross_entropy(hl, head_targets)
    loss_state = F.cross_entropy(sl, state_targets)

    return loss_val + loss_head + loss_state


def _sft_accuracy(model, X_val, Y_val, state_index):
    """Overall step accuracy on val set using greedy decode."""
    model.eval()
    with torch.no_grad():
        logits = model(X_val)

    tape_idx  = logits["value_logits"].argmax(dim=-1)   # (B, n)
    head_idx  = logits["head_logits"].argmax(dim=-1)    # (B,)
    state_idx = logits["state_logits"].argmax(dim=-1)   # (B,)

    true_val, true_head, true_state = _extract_targets(Y_val, state_index)

    val_match   = (tape_idx == true_val).all(dim=1)
    head_match  = (head_idx == true_head)
    state_match = (state_idx == true_state)

    return (val_match & head_match & state_match).float().mean().item()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def train_sft_then_grpo_streaming(
    machine_name: str = "incrementer",
    n_samples: int = 2000,
    sft_max_epochs: int = 50,
    sft_threshold: float = 0.90,
    sft_lr: float = 1e-3,
    sft_batch_size: int = 32,
    grpo_epochs: int = 100000,
    grpo_lr: float = 1e-4,
    grpo_batch_size: int = 16,
    K: int = 8,
    kl_coef: float = 0.01,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 2,
    min_tape_len: int = 16,
    seed: int = 42,
    analyze_every: int = 5,
    analyze_samples: int = 500,
    checkpoint_name: str = None,
) -> Generator[dict, None, None]:
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

    arch = dict(
        d_input=d_input, n_tape=n_tape, n_states=n_states,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
    )

    model = TMTransformerCategorical(
        d_input=d_input, n_tape=n_tape, n_states=n_states,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    )

    yield {
        "type":           "init",
        "machine":        machine_name,
        "n_train":        len(train_pairs),
        "n_val":          len(val_pairs),
        "d_input":        d_input,
        "n_tape":         n_tape,
        "n_states":       n_states,
        "sft_max_epochs": sft_max_epochs,
        "sft_threshold":  sft_threshold,
        "grpo_epochs":    grpo_epochs,
        "K":              K,
        "analyze_every":  analyze_every,
    }

    # ── Phase 1: SFT ──────────────────────────────────────────────────────────
    yield {"type": "phase", "phase": "sft"}

    optimizer = torch.optim.Adam(model.parameters(), lr=sft_lr)
    train_dl  = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=sft_batch_size, shuffle=True,
    )

    sft_epoch    = 0
    best_sft_acc = 0.0

    for epoch in range(1, sft_max_epochs + 1):
        sft_epoch = epoch
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = _categorical_sft_loss(logits, yb, state_index)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(train_pairs)
        val_acc     = _sft_accuracy(model, X_val, Y_val, state_index)

        if val_acc > best_sft_acc:
            best_sft_acc = val_acc

        yield {
            "type":       "epoch",
            "phase":      "sft",
            "epoch":      epoch,
            "train_loss": round(epoch_loss, 6),
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
                "phase":            "sft",
                "epoch":            epoch,
                "overall_acc":      round(overall, 4),
                "category_summary": cats,
                "table":            table,
            }

        if val_acc >= sft_threshold:
            yield {
                "type":    "sft_done",
                "epoch":   epoch,
                "val_acc": round(val_acc, 4),
                "reason":  f"reached threshold {sft_threshold:.0%}",
            }
            break
    else:
        yield {
            "type":    "sft_done",
            "epoch":   sft_epoch,
            "val_acc": round(best_sft_acc, 4),
            "reason":  f"reached max epochs {sft_max_epochs}",
        }

    ckpt_name = checkpoint_name or f"{machine_name}_sft_e{sft_epoch}"
    ckpt_path = save_checkpoint(
        model=model, model_class="categorical",
        arch=arch, state_index=state_index,
        machine=machine_name, epoch=sft_epoch,
        val_acc=best_sft_acc, phase="sft",
        name=ckpt_name,
    )
    yield {"type": "checkpoint_saved", "path": str(ckpt_path), "name": ckpt_name}

    # ── Phase 2: GRPO ─────────────────────────────────────────────────────────
    yield {"type": "phase", "phase": "grpo"}

    config    = GRPOConfig(K=K, lr=grpo_lr, kl_coef=kl_coef)
    optimizer = torch.optim.Adam(model.parameters(), lr=grpo_lr)

    grpo_train_dl = DataLoader(
        TensorDataset(X_train, torch.arange(len(train_pairs))),
        batch_size=grpo_batch_size, shuffle=True,
    )

    best_grpo_acc = 0.0

    for epoch in range(1, grpo_epochs + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_reward = 0.0
        n_batches    = 0

        for xb, idx_b in grpo_train_dl:
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
            xb_exp    = xb.unsqueeze(1).expand(B, K_actual, n, d_input)\
                          .reshape(B * K_actual, n, d_input)
            tape_idx  = samples["tape_samples"].reshape(B * K_actual, n)
            head_idx  = samples["head_samples"].reshape(B * K_actual)
            state_idx = samples["state_samples"].reshape(B * K_actual)

            log_probs_flat = model.log_prob_of(
                xb_exp, tape_idx, head_idx, state_idx,
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
        val_acc    = _sft_accuracy(model, X_val, Y_val, state_index)

        if val_acc > best_grpo_acc:
            best_grpo_acc = val_acc

        yield {
            "type":         "epoch",
            "phase":        "grpo",
            "epoch":        epoch,
            "train_loss":   round(avg_loss, 6),
            "train_reward": round(avg_reward, 4),
            "val_acc":      round(val_acc, 4),
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
                "phase":            "grpo",
                "epoch":            epoch,
                "overall_acc":      round(overall, 4),
                "category_summary": cats,
                "table":            table,
            }

    yield {"type": "done", "best_grpo_acc": round(best_grpo_acc, 4)}