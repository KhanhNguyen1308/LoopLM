from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopLMConfig:
    """
    Configuration for LoopLM — a Recurrent-Depth Transformer built on Qwen3.

    The 28 Qwen3 decoder layers are split into three functional stages:

        Input → [Prelude × 1] → e (encoded input)
                                 ↓
                    [Recurrent block × T loops]
                    h_{t+1} = TransformerLayers(alpha*e + (1-alpha)*h_t)
                                 ↓
                    [Coda × 1] → Norm → LM Head

    The loop depth T is variable at inference time, allowing test-time scaling
    of compute depth without changing any parameters.

    Defaults match Qwen3-1.7B-Base (28 layers split 4/20/4).
    """

    # ── Qwen3 base hyperparameters ────────────────────────────────────────────
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True

    # ── Loop architecture ─────────────────────────────────────────────────────
    # Layer split must satisfy: prelude_layers + recurrent_layers + coda_layers
    #                           == num_hidden_layers
    prelude_layers: int = 4      # first N layers, run exactly once
    recurrent_layers: int = 20   # middle N layers, looped T times
    coda_layers: int = 4         # last N layers, run exactly once

    # Default loop depth used at inference (overridable per forward() call)
    max_loop_iters: int = 8

    # ── Loop injection ────────────────────────────────────────────────────────
    # Convex-combination injection: h_in = (1 - alpha) * h_t  +  alpha * e
    #   alpha = sigmoid(alpha_raw), per-channel, initialized near 0
    #   At init: alpha ≈ 0, so h_in ≈ h_t (near-identity, pretrained weights
    #   transfer cleanly). Training shifts alpha upward to enable proper mixing.
    injection_init: float = -10.0  # initial value of alpha_raw; sigmoid(-10) ≈ 0

    # ── Loop-step embeddings ──────────────────────────────────────────────────
    # If True, adds a learned per-loop-step bias so each iteration can learn
    # to perform a distinct role (like positional encoding over loop depth).
    use_loop_embeddings: bool = False

    # ── Adaptive Computation Time (ACT) ───────────────────────────────────────
    # If True, enables per-token early exit from the recurrent loop based on
    # a learned halting signal.  Training adds an ACT regularization term.
    use_act: bool = False
    act_threshold: float = 0.99  # cumulative halting probability to stop

    def __post_init__(self) -> None:
        total = self.prelude_layers + self.recurrent_layers + self.coda_layers
        if total != self.num_hidden_layers:
            raise ValueError(
                f"prelude_layers({self.prelude_layers}) + "
                f"recurrent_layers({self.recurrent_layers}) + "
                f"coda_layers({self.coda_layers}) = {total} "
                f"!= num_hidden_layers({self.num_hidden_layers})"
            )
