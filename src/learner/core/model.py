"""
Transformer model for TM step prediction.

Two modes:
  - regression: continuous output (original MSE approach)
  - categorical: discrete output over {0,1,_} per cell (for GRPO)

The categorical model outputs logits for each slot independently:
  - value_logits:  (B, n, 3)   over {0, 1, _}
  - head_logits:   (B, n)      over positions
  - state_logits:  (B, S)      over states
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TMTransformer(nn.Module):
    """Original regression model — kept for MSE baseline."""
    def __init__(
        self,
        d_input: int,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.output_proj(h)


class TMTransformerCategorical(nn.Module):
    """
    Categorical output model for GRPO.

    Outputs separate logit heads for:
      - tape value at each position: 3 classes {0, 1, _}
      - head position: n classes
      - next state: S classes
    """
    # Symbol vocabulary
    SYMBOLS = ["0", "1", "_"]
    SYM2IDX = {s: i for i, s in enumerate(SYMBOLS)}

    def __init__(
        self,
        d_input: int,
        n_tape: int,
        n_states: int,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_tape   = n_tape
        self.n_states = n_states
        n_syms        = len(self.SYMBOLS)

        self.input_proj = nn.Linear(d_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Separate output heads
        self.value_head = nn.Linear(d_model, n_syms)    # per token → 3 classes
        self.head_head  = nn.Linear(d_model, 1)          # per token → scalar, softmax over n
        self.state_head = nn.Linear(d_model, n_states)   # mean pool → S classes

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, n, d_input)
        returns dict of logits:
          value_logits: (B, n, 3)
          head_logits:  (B, n)
          state_logits: (B, S)
        """
        h = self.input_proj(x)           # (B, n, d_model)
        h = self.transformer(h)          # (B, n, d_model)

        value_logits = self.value_head(h)             # (B, n, 3)
        head_logits  = self.head_head(h).squeeze(-1)  # (B, n)
        state_logits = self.state_head(h.mean(dim=1)) # (B, S)

        return {
            "value_logits": value_logits,
            "head_logits":  head_logits,
            "state_logits": state_logits,
        }

    def sample(self, x: torch.Tensor, K: int = 8) -> dict:
        """
        Sample K candidate next-states for each example in the batch.

        x: (B, n, d_input)
        returns:
          tape_samples:  (B, K, n)  integer indices into SYMBOLS
          head_samples:  (B, K)     integer head positions
          state_samples: (B, K)     integer state indices
          log_probs:     (B, K)     total log prob of each sample
        """
        B, n, _ = x.shape
        logits  = self.forward(x)

        vl = logits["value_logits"]   # (B, n, 3)
        hl = logits["head_logits"]    # (B, n)
        sl = logits["state_logits"]   # (B, S)

        # Expand for K samples: (B*K, ...)
        vl_k = vl.unsqueeze(1).expand(B, K, n, 3).reshape(B*K, n, 3)
        hl_k = hl.unsqueeze(1).expand(B, K, n).reshape(B*K, n)
        sl_k = sl.unsqueeze(1).expand(B, K, self.n_states).reshape(B*K, self.n_states)

        # Sample tape values: (B*K, n)
        tape_flat  = torch.distributions.Categorical(logits=vl_k.reshape(B*K*n, 3))\
                         .sample().reshape(B*K, n)
        # Sample head position: (B*K,)
        head_flat  = torch.distributions.Categorical(logits=hl_k).sample()
        # Sample state: (B*K,)
        state_flat = torch.distributions.Categorical(logits=sl_k).sample()

        # Compute log probs
        lp_tape  = F.log_softmax(vl_k.reshape(B*K*n, 3), dim=-1)
        lp_tape  = lp_tape.gather(1, tape_flat.reshape(B*K*n, 1)).reshape(B*K, n).sum(dim=1)

        lp_head  = F.log_softmax(hl_k, dim=-1)
        lp_head  = lp_head.gather(1, head_flat.unsqueeze(1)).squeeze(1)

        lp_state = F.log_softmax(sl_k, dim=-1)
        lp_state = lp_state.gather(1, state_flat.unsqueeze(1)).squeeze(1)

        log_probs = lp_tape + lp_head + lp_state  # (B*K,)

        return {
            "tape_samples":  tape_flat.reshape(B, K, n),
            "head_samples":  head_flat.reshape(B, K),
            "state_samples": state_flat.reshape(B, K),
            "log_probs":     log_probs.reshape(B, K),
        }

    def log_prob_of(
        self,
        x: torch.Tensor,
        tape_idx: torch.Tensor,
        head_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log prob of specific (tape, head, state) targets.
        Used during GRPO policy update.

        tape_idx:  (B, n)  integer indices
        head_idx:  (B,)
        state_idx: (B,)
        returns:   (B,) log probs
        """
        logits = self.forward(x)

        lp_tape = F.log_softmax(logits["value_logits"], dim=-1)  # (B, n, 3)
        lp_tape = lp_tape.gather(2, tape_idx.unsqueeze(2)).squeeze(2).sum(dim=1)

        lp_head = F.log_softmax(logits["head_logits"], dim=-1)   # (B, n)
        lp_head = lp_head.gather(1, head_idx.unsqueeze(1)).squeeze(1)

        lp_state = F.log_softmax(logits["state_logits"], dim=-1) # (B, S)
        lp_state = lp_state.gather(1, state_idx.unsqueeze(1)).squeeze(1)

        return lp_tape + lp_head + lp_state  # (B,)