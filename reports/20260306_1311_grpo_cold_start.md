# GRPO Cold Start Experiment

**Date:** 2026-03-06 13:11  
**Git hash:** `97c417ee`  
**Command:** `learner grpo run --epochs 100000 --samples 10000 --analyze-every 5 --K 8`

## Setup

First attempt at GRPO training for TM step prediction. The model is a
TMTransformerCategorical — a small transformer with separate categorical
output heads for tape values, head position, and next state. The verifier
is the TM itself: tm.step() provides a binary reward (1 if fully correct,
0 otherwise). No supervised pretraining — GRPO from random initialization.

**Architecture:**
- 2 layers, 2 heads, d_model=32
- d_input=10, n_tape=16, n_states=3
- K=8 candidates per example
- lr=1e-4, kl_coef=0.01
- 8056 train / 896 val pairs, balanced 50/50 carry/scan_right

## Results

| Epoch | Loss | Train Reward | Val Acc |
|-------|------|-------------|---------|
| 1 | 0.000000 | 0.0000 | 0.0000 |
| 5 | 0.000000 | 0.0006 | 0.0045 |
| 10 | 0.000000 | 0.0015 | 0.0000 |
| 15 | 0.000000 | 0.0015 | 0.0000 |
| 20 | 0.000000 | 0.0015 | 0.0000 |
| 23 | 0.000000 | 0.0015 | 0.0000 |

Training reward plateaued at 0.0015 and never moved. Loss was 0.000000
throughout. Val accuracy never exceeded 0.45%.

## Analysis (Epoch 5, 500 samples)

| Feature | Value | Acc | Tape | Head | State |
|---------|-------|-----|------|------|-------|
| OVERALL | all | 0.0% | 0.0% | 13.8% | 78.8% |
| state_before | carry | 0.0% | 0.0% | 17.6% | 0.0% |
| state_before | scan_right | 0.0% | 0.0% | 13.4% | 87.8% |
| direction | left | 0.0% | 0.0% | 13.2% | 0.0% |
| direction | right | 0.0% | 0.0% | 14.0% | 100.0% |
| read_bit | _ | 0.0% | 0.0% | 9.1% | 0.0% |

## Interpretation

The experiment confirmed the **GRPO cold start problem**. Three failure
signatures are visible:

**Loss is identically zero.** When all K=8 candidates receive reward 0,
the group advantages are all zero (or undefined from zero std), and the
policy gradient is exactly zero. No update occurs.

**Train reward is stuck at 0.0015.** This is not learning — it is the
baseline probability of randomly sampling the correct tape from an
untrained model. The tape has 16 positions each with 3 possible symbols
(3^16 ≈ 43 million combinations). With K=8 random samples, the probability
of accidentally hitting the correct answer is negligible. The model never
gets a positive reward signal to bootstrap from.

**State accuracy is partially non-zero (78.8%) but tape accuracy is 0%.**
The model has learned nothing about tape values or head position. State
prediction is partially above chance because there are only 3 states and
the model is collapsing to a majority prediction.

## Root Cause

GRPO requires the model to already be close enough to the correct answer
that some fraction of K samples occasionally succeed. DeepSeek-R1 and
similar systems always apply supervised fine-tuning (SFT) first, then
switch to GRPO. Without SFT warmup, the reward signal is too sparse to
bootstrap learning from random initialization.

## Next Steps

The fix is a two-phase training pipeline:

1. **SFT warmup** — run the MSE supervised trainer for ~20 epochs until
   scan_right accuracy reaches ~90%. This gives the model a reasonable
   starting point.
2. **Model serialization** — save the SFT checkpoint with architecture
   metadata so it can be loaded into the GRPO trainer.
3. **GRPO from SFT init** — load the pretrained weights, switch to GRPO
   with the TM verifier. At 90% scan_right accuracy the model will
   occasionally sample correct carry transitions, providing the positive
   reward signal GRPO needs to bootstrap.

The hypothesis is that once GRPO has a non-zero reward signal on carry
steps, it will learn the carry transition faster than supervised MSE
because the binary verifiable reward penalizes partial correctness — the
exact property MSE lacks.