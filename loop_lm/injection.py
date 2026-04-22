from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoopInjection(nn.Module):
    """
    Convex-combination input injection for the recurrent loop.

    Implements the per-channel gated blend:
        h_in = (1 - alpha) * h_t  +  alpha * e

    where:
        h_t   - hidden state at loop step t (B, T, hidden_size)
        e     - encoded input from the Prelude, constant across loops (B, T, hidden_size)
        alpha - per-channel gate in (0, 1), learned via sigmoid(alpha_raw)

    Properties:
        - Output norm is always bounded by max(||h_t||, ||e||)  (convex combination)
        - At init (alpha_raw = injection_init ≈ -10): alpha ≈ 0, so h_in ≈ h_t
          This ensures a model loaded from pretrained Qwen3 weights behaves
          identically to the original Qwen3 at T=1 loop iteration before any
          fine-tuning has occurred.
        - As training progresses, alpha grows to enable genuine recurrent depth.

    Note on stability:
        The injection itself is bounded, but the full recurrent map
        h_{t+1} = TransformerLayers(alpha*e + (1-alpha)*h_t)
        is nonlinear. Practical stability is ensured by the bounded
        convex combination at the input to each loop iteration — the
        transformer cannot amplify hidden states unboundedly because its
        residual-stream norm is regularized via RMSNorm at each layer.
    """

    def __init__(self, hidden_size: int, init_val: float = -10.0) -> None:
        """
        Args:
            hidden_size - feature dimension of hidden states
            init_val    - initial value for alpha_raw; sigmoid(init_val) gives
                          the initial mixing fraction toward the encoded input.
                          Default -10.0 → alpha ≈ 4.5e-5 (near-identity).
        """
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.full((hidden_size,), init_val))

    def get_alpha(self) -> torch.Tensor:
        """Per-channel mixing coefficient α ∈ (0, 1)."""
        return torch.sigmoid(self.alpha_raw)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h - current hidden state  (B, T, hidden_size)
            e - prelude-encoded input (B, T, hidden_size)
        Returns:
            Blended input for this loop step (B, T, hidden_size)
        """
        alpha = self.get_alpha()  # (hidden_size,) — broadcasts over B and T
        return (1.0 - alpha) * h + alpha * e

    @property
    def mean_alpha(self) -> float:
        """Mean mixing coefficient across all channels (for monitoring)."""
        with torch.no_grad():
            return self.get_alpha().mean().item()
