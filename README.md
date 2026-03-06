# learner

An empirical companion to the paper:

> **Agent Completeness via Circuit Simulation:
> A Natural Proof that Transformer Agents are Universal Computers**
> Greg Coppola, PhD (March 2026)

The paper proves theoretically that a 4-layer transformer operating in a
read-compute-write agent loop is Turing complete. This project asks the
empirical follow-up question:

> Can gradient descent actually find those weights?

## The Experiment

We pick a specific Turing machine (the binary incrementer), generate
synthetic training data by running it forward, and train a small transformer
to predict one TM step at a time. If the transformer learns the transition
function, we unroll the agent loop and watch it run the full computation.

This directly tests the open question in Section 8.3 of the paper: whether
trained transformers learn weight configurations that approximate the
circuit simulation structure — not just that such weights exist, but that
SGD finds them.

## Architecture

Each TM snapshot (tape + head position + state) is encoded as a sequence
of typed proposition tokens, one per tape cell, following the representation
in the paper: