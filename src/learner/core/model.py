"""
Small transformer model for TM step prediction.

Free-mode: 2 layers, 2 heads, d_model=32.
Input:  (B, n, d_input)  — encoded TM snapshot
Output: (B, n, d_input)  — predicted next snapshot

We project d_input -> d_model, run the transformer, project back.
"""
import torch
import torch.nn as nn


class TMTransformer(nn.Module):
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
        """
        x: (B, n, d_input)
        returns: (B, n, d_input)
        """
        h = self.input_proj(x)        # (B, n, d_model)
        h = self.transformer(h)       # (B, n, d_model)
        return self.output_proj(h)    # (B, n, d_input)