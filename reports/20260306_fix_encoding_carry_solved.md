
# Encoding Fix: Carry Solved in 4 Epochs

**Date:** 2026-03-06  
**Branch:** fix-encoding  
**Command:** `learner hybrid run --samples 10000 --epochs 100000 --carry-weight 10 --analyze-every 5`

## The Problem

After six distinct training approaches across multiple experiments, carry
accuracy was stuck at exactly 0.0%. Every approach — MSE regression,
balanced sampling, curriculum learning, categorical cross-entropy SFT,
GRPO cold start, and SFT+GRPO — failed to move carry off zero.

The consistent pattern pointed away from training dynamics and toward
something structural. The smoking gun emerged from examining the analysis
tables closely:

    state_before  carry   0.0% tape,  100% head,  100% state

The model knew it was in carry state. It knew where the head was. It
predicted the correct next state. Only the tape write was wrong — always.

## Root Cause

The bug was in `encoding.py`. The value slot used a single bit:

    val = [1.0 if sym == "1" else 0.0]

Both `"0"` and `"_"` (blank) encoded as `0.0`. They were indistinguishable
in the input representation.

This silently corrupted the supervision signal for the carry transition that
reads a blank cell:

    ("carry", "_") → write "1", move RIGHT, go to "done"

This is the new-MSB case — when carry propagates all the way to the left
edge of the tape and needs to extend the number. In the encoded target Y,
the cell where `"1"` should be written looked identical to a `"0"` cell.
The model was being told to predict class 0 when the correct answer was
class 1. The label was wrong.

Meanwhile, `value_logits` in the model output has always had 3 classes
`{0, 1, _}` matching `TMTransformerCategorical.SYM2IDX`. The output head
was correctly specified — only the input encoding and target extraction
were broken.

## The Fix

Three files changed:

**`encoding.py`** — replaced the single value bit with a 3-class one-hot:

    "0" → [1, 0, 0]
    "1" → [0, 1, 0]
    "_" → [0, 0, 1]

This changes `d` from `1 + b + 1 + S` to `3 + b + 1 + S`, and shifts
the slot offsets for `is_head` and `state` accordingly.

**`trainer_sft_grpo.py`** — extracted `_extract_targets` helper. The old
val target extraction was:

    val_targets = (Y[:, :, 0] > 0.5).long()

The new extraction is:

    val_targets = Y[:, :, 0:3].argmax(dim=-1)

Head and state slot offsets updated from `1+b` / `1+b+1` to `3+b` / `3+b+1`.

**`trainer_hybrid.py`** — imports and uses `_extract_targets` from
`trainer_sft_grpo.py`. No independent slot logic.

Files not changed: `model.py`, `trainer_grpo.py`, `analysis.py`, `data.py`,
`machines.py`, `grpo.py`. None of these extract val targets from Y or
depend on the slot layout.

## Results

    Ep  Loss        L-Scan      L-Carry     Val Acc
    1   10.384459   1.435434    3.705317    0.6496
    2   1.529398    0.747344    0.018099    0.8917
    3   0.109050    0.051505    0.002364    0.9955
    4   0.007996    0.003395    0.000639    1.0000

100% accuracy on epoch 4. Carry accuracy at epoch 5 analysis:

    state_before  carry       104  100.0%  100.0%  100.0%  100.0%
    state_before  scan_right  896  100.0%  100.0%  100.0%  100.0%

Every feature bucket hit 100%. The model generalizes correctly across
all tape lengths, read bits, directions, and carry sub-cases including
the new-MSB edge case (`read_bit: _`).

## Interpretation

The model was never failing to learn carry. It was being given wrong labels
for the hardest carry case and correct labels for everything else. Once the
encoding correctly distinguished blank from zero, the supervised signal was
clean and the model solved the task in 4 epochs from random initialization.

The six previous experiments — including GRPO, curriculum learning, and
carry-weighted loss — were all fighting a data corruption problem, not a
learning dynamics problem. No amount of loss weighting or reward shaping
can overcome systematically incorrect supervision targets.

## Lesson

When a model learns head position and state perfectly but fails only on
tape values, and the failure is confined to transitions that write to blank
cells, check the encoding of blank first.