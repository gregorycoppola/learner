"""
GRPO training for TM step prediction.

The verifier is the TM itself — tm.step() gives the exact correct
next state for any input. Reward is binary: 1 if fully correct, 0 if not.

Algorithm (per batch):
  1. For each example, sample K candidate next-states from the model
  2. Verify each candidate against tm.step()
  3. Compute advantages via GRPO normalization within each group of K
  4. Policy gradient update: maximize E[advantage * log_prob]
  5. KL penalty against reference (old) log_probs for stability
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    K: int   = 8          # candidates per example
    lr: float = 1e-4      # lower than supervised — policy gradient is noisier
    kl_coef: float = 0.01 # KL penalty coefficient
    eps: float = 1e-8     # for advantage normalization
    clip_eps: float = 0.2 # PPO-style clipping (optional stability)


def verify_batch(
    tape_samples: torch.Tensor,   # (B, K, n)
    head_samples: torch.Tensor,   # (B, K)
    state_samples: torch.Tensor,  # (B, K)
    tape_before: list[list[str]], # B tapes
    head_before: list[int],
    state_before: list[str],
    state_index: dict[str, int],
    tm,
) -> torch.Tensor:
    """
    Run verifier on all B*K candidates.
    Returns rewards: (B, K) float tensor, 1.0 if correct else 0.0.
    """
    from learner.core.model import TMTransformerCategorical as Cat
    idx2sym   = {v: k for k, v in Cat.SYM2IDX.items()}
    idx2state = {v: k for k, v in state_index.items()}

    B, K, n = tape_samples.shape
    rewards = torch.zeros(B, K)

    for b in range(B):
        tb = tape_before[b]
        hb = head_before[b]
        sb = state_before[b]

        # Ground truth next state
        true_tape, true_head, true_state, _ = tm.step(tb, hb, sb)

        for k in range(K):
            # Decode candidate
            pred_tape  = [idx2sym[tape_samples[b, k, i].item()]
                          for i in range(n)]
            pred_head  = head_samples[b, k].item()
            pred_state = idx2state.get(state_samples[b, k].item(), "unknown")

            # Compare against ground truth (trim to true tape length)
            tape_match  = pred_tape[:len(true_tape)] == true_tape
            head_match  = pred_head == true_head
            state_match = pred_state == true_state

            if tape_match and head_match and state_match:
                rewards[b, k] = 1.0

    return rewards


def grpo_loss(
    log_probs: torch.Tensor,      # (B, K) current policy
    old_log_probs: torch.Tensor,  # (B, K) reference policy
    rewards: torch.Tensor,        # (B, K)
    config: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO loss with KL penalty.

    Advantage = normalized reward within each group of K.
    Loss = -mean(advantage * log_prob) + kl_coef * KL(current || reference)
    """
    # Normalize rewards within each group → advantages
    r_mean = rewards.mean(dim=1, keepdim=True)   # (B, 1)
    r_std  = rewards.std(dim=1, keepdim=True) + config.eps
    advantages = (rewards - r_mean) / r_std       # (B, K)

    # Policy gradient loss
    # Clipped ratio for stability (PPO-style)
    ratio = torch.exp(log_probs - old_log_probs)  # (B, K)
    clipped = torch.clamp(ratio, 1 - config.eps, 1 + config.clip_eps)
    pg_loss = -torch.min(
        ratio   * advantages,
        clipped * advantages,
    ).mean()

    # KL penalty: KL(current || reference) ≈ exp(old) * (old - new)
    kl = (torch.exp(old_log_probs) * (old_log_probs - log_probs)).mean()

    total_loss = pg_loss + config.kl_coef * kl

    stats = {
        "pg_loss":    round(pg_loss.item(), 6),
        "kl":         round(kl.item(), 6),
        "total_loss": round(total_loss.item(), 6),
        "mean_reward": round(rewards.mean().item(), 4),
        "reward_std":  round(rewards.std().item(), 4),
        "advantages_mean": round(advantages.mean().item(), 4),
    }

    return total_loss, stats