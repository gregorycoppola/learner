# SFT + GRPO Experiment — Carry Still Zero

**Date:** 2026-03-06 13:53  
**Git hash:** `4c8dcbd4`  
**Command:** `learner sft-grpo run --samples 10000 --sft-epochs 30 --grpo-epochs 100000 --analyze-every 5 --analyze-samples 1000 --K 64`

## Setup

Two-phase training pipeline: supervised fine-tuning (SFT) with cross-entropy
loss on the categorical model, followed by GRPO with the TM verifier as reward
signal. The cold start problem from the previous experiment is solved by
warming up with SFT before switching to GRPO. K=64 candidates per example
(up from K=8) to increase within-group variance and stabilize GRPO gradients.

**Architecture:** TMTransformerCategorical, 2 layers, 2 heads, d_model=32  
**Data:** 8056 train / 896 val, balanced 50/50 carry/scan_right  
**SFT:** cross-entropy loss, lr=1e-3, up to 30 epochs, threshold 90%  
**GRPO:** binary TM verifier reward, lr=1e-4, K=64, kl_coef=0.01

## Results

### Phase 1: SFT

| Epoch | Loss | Val Acc |
|-------|------|---------|
| 1 | 1.605464 | 87.4% |
| 2 | 0.207715 | 97.1% ★ |

SFT hit the 90% threshold in just 2 epochs and reached 97.1% val accuracy.
The categorical model with cross-entropy loss learns dramatically faster than
the original MSE regression model, which required 50+ epochs to reach 92%.

### Phase 2: GRPO

| Epoch | Loss | Train Reward | Val Acc |
|-------|------|-------------|---------|
| 1 | 0.000000 | 0.3757 | 95.3% |
| 2 | 0.000000 | 0.3775 | 95.3% |
| 5 | 0.000000 | 0.3776 | 95.3% |
| 7 | 0.000000 | 0.3779 | 95.3% |

Train reward immediately jumped to 37.6% — confirming the cold start problem
is solved. But loss is 0.000000 throughout and val accuracy is frozen at 95.3%.

### Analysis at GRPO Epoch 5 (1000 samples, natural distribution)

| Feature | Value | N | Acc | Tape | Head | State |
|---------|-------|---|-----|------|------|-------|
| OVERALL | all | 1000 | 67.4% | 67.4% | 94.7% | 94.6% |
| state_before | carry | 104 | 0.0% | 0.0% | 100.0% | 100.0% |
| state_before | scan_right | 896 | 75.2% | 75.2% | 94.1% | 94.0% |
| tape_changes | False | 785 | 85.9% | 85.9% | 96.4% | 93.1% |
| tape_changes | True | 215 | 0.0% | 0.0% | 88.4% | 100.0% |
| direction | left | 215 | 0.0% | 0.0% | 87.0% | 74.9% |
| direction | right | 785 | 85.9% | 85.9% | 96.8% | 100.0% |
| read_bit | _ | 111 | 0.0% | 0.0% | 74.8% | 51.4% |
| head_at_right | True | 275 | 0.0% | 0.0% | 80.7% | 80.4% |

## Interpretation

### The 97% Val Accuracy is Misleading

The val set is balanced 50/50 carry/scan_right. The analysis on the natural
distribution reveals the true picture: carry accuracy is exactly 0.0% across
every checkpoint, every approach, every run. The model has learned only
scan_right. Overall accuracy on the natural distribution is 67.4% — identical
to the very first MSE experiment. No progress on carry has been made across
any training approach tried so far.

### Why GRPO Loss is Zero

Train reward of 37.6% means most scan_right candidates are verified correct.
Within each group of K=64, the advantages normalize to near-zero because
variance collapses when most samples get the same reward. Carry examples
contribute 0 reward but are drowned out by scan_right. The gradient is
effectively zero — GRPO has nothing to push against.

### The Consistent Pattern

Across six distinct experimental approaches, carry accuracy has never moved
off 0.0%:

| Approach | Carry Acc |
|----------|-----------|
| MSE regression, natural distribution | 0.0% |
| MSE regression, balanced sampling | 0.0% |
| MSE + curriculum/hard example mining | 0.0% |
| Categorical model + cross-entropy SFT | 0.0% |
| GRPO cold start | 0.0% |
| SFT then GRPO (this experiment) | 0.0% |

This is not a training dynamics problem. It is a representation or loss
signal problem. The model is not struggling with carry — it is structurally
ignoring it because every training approach allows it to minimize loss
without ever learning carry.

## Root Cause Hypothesis

The carry transition requires the model to write a different value at the
head position and move left. The model has learned a dominant strategy —
copy the tape forward and move right — that covers 85% of training examples
with near-zero loss. No training signal has been strong enough to overcome
this local minimum because:

1. Carry examples are a minority even in balanced data (~10% of natural distribution)
2. MSE and cross-entropy both give partial credit for nearly-correct predictions
3. GRPO advantages collapse when scan_right dominates the reward signal

## Next Experiment

Apply a separate, stronger loss function specifically for carry transitions:
binary verifiable penalty (1 - correct) weighted 10x relative to the
cross-entropy loss on scan_right. This forces the model to prioritize carry
correctness without partial credit. The verifier is tm.step() — deterministic,
exact, no sampling needed. Implementation: `trainer_hybrid.py` with a
`--carry-weight` hyperparameter.