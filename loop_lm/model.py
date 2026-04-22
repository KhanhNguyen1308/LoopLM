from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from .config import LoopLMConfig
from .injection import LoopInjection


# ─────────────────────────────────────────────────────────────────────────────
# Attention mask helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a full lower-triangular additive causal mask.

    Returns shape (1, 1, seq_len, seq_len) where:
        - 0.0     : positions that CAN attend (on/below diagonal)
        - -inf    : positions that CANNOT attend (above diagonal / future)

    The (1, 1, …) shape broadcasts over batch size and attention heads.
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask = mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def _extend_mask_for_padding(
    causal_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Extend a (1, 1, T, T) causal mask with a (B, T) padding mask.

    Padding positions (attention_mask == 0) receive -inf in the key dimension
    so all query positions ignore them.

    Args:
        causal_mask    - (1, 1, T, T) base causal mask
        attention_mask - (B, T) binary mask; 1 = real token, 0 = padding
        dtype          - target float dtype
    Returns:
        (B, 1, T, T) combined mask
    """
    # (B, T) → (B, 1, 1, T) key mask; 0.0 for attend, -inf for pad
    pad_mask = (1.0 - attention_mask.to(dtype))[:, None, None, :]
    pad_mask = pad_mask * torch.finfo(dtype).min  # use dtype min, not literal -inf
    return causal_mask + pad_mask  # broadcasts to (B, 1, T, T)


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent block
# ─────────────────────────────────────────────────────────────────────────────


class RecurrentBlock(nn.Module):
    """
    The looped core of LoopLM.

    Runs `n_loops` iterations of:
        h_in   = injection(h_t, e)         # convex blend with encoded input
        h_{t+1} = TransformerLayers(h_in)   # depth computation

    where `e` is the encoded Prelude output, injected at every step to prevent
    the recurrent state from drifting away from the original input signal.
    """

    def __init__(
        self,
        layers: list,
        injection: LoopInjection,
        loop_embed: Optional[nn.Embedding] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.injection = injection
        self.loop_embed = loop_embed
        self.gradient_checkpointing = False

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        n_loops: int,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h                   - initial hidden state from the Prelude (B, T, D)
                                  Should equal `e` at the start so that T=1 with
                                  near-identity injection behaves like the original
                                  single-pass transformer.
            e                   - encoded input; held constant across all loops (B, T, D)
            n_loops             - number of recurrent iterations T
            position_ids        - (1, T) token position indices
            position_embeddings - (cos, sin) from RotaryEmbedding, shape (B, T, head_dim)
            attention_mask      - additive causal mask (1/B, 1, T, T) or None

        Returns:
            Final hidden state after n_loops iterations (B, T, D)
        """
        for t in range(n_loops):
            if self.gradient_checkpointing and self.training:
                # Wrap the iteration in a closure so only tensors cross the
                # checkpoint boundary; non-tensor args are captured by the closure.
                def make_iter_fn(t_idx: int):
                    def _iter(h_: torch.Tensor, e_: torch.Tensor) -> torch.Tensor:
                        h_in = self.injection(h_, e_)
                        if self.loop_embed is not None:
                            step_idx = torch.tensor(t_idx, device=h_.device)
                            h_in = h_in + self.loop_embed(step_idx)
                        for layer in self.layers:
                            h_in = layer(
                                h_in,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                                use_cache=False,
                            )
                        return h_in
                    return _iter

                h = gradient_checkpoint(make_iter_fn(t), h, e, use_reentrant=False)
            else:
                h_in = self.injection(h, e)
                if self.loop_embed is not None:
                    step_idx = torch.tensor(t, device=h.device)
                    h_in = h_in + self.loop_embed(step_idx)
                for layer in self.layers:
                    h_in = layer(
                        h_in,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                    )
                h = h_in

        return h


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────


class LoopLMForCausalLM(nn.Module):
    """
    LoopLM: Recurrent-Depth Transformer for Causal Language Modeling.

    Converts a pretrained Qwen3 model into a looped architecture by splitting
    its decoder layers into Prelude / Recurrent / Coda stages.

    Architecture:

        tokens
          ↓  embed_tokens
        hidden_states
          ↓  [Prelude layers × 1]
        e  (encoded input, frozen across loop iterations)
          ↓  [Recurrent block × T loops]
             loop_t:  h_in = (1-α)*h + α*e
                      h    = TransformerLayers(h_in)
          ↓  [Coda layers × 1]
          ↓  RMSNorm
          ↓  lm_head
        logits

    The loop depth T is set at inference time via the `n_loops` argument to
    forward() and generate(), allowing test-time compute scaling with no extra
    parameters.

    Quick start:
        model = LoopLMForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base")
        logits = model(input_ids, n_loops=8)["logits"]
    """

    def __init__(self, config: LoopLMConfig) -> None:
        super().__init__()
        self.config = config
        # Submodules are populated by _build_from_qwen3; declaring types here
        # so type checkers and IDE tooling work correctly.
        self.embed_tokens: nn.Embedding
        self.rotary_emb: nn.Module
        self.prelude: nn.ModuleList
        self.recurrent: RecurrentBlock
        self.coda: nn.ModuleList
        self.norm: nn.Module
        self.lm_head: nn.Linear

    # ── Construction helpers ──────────────────────────────────────────────────

    def _build_from_qwen3(self, qwen3_for_causal_lm: nn.Module) -> None:
        """
        Wire up LoopLM internals from a Qwen3ForCausalLM instance.

        The decoder layers are split by *reassigning references*, not copying
        weights.  Each layer module ends up registered under exactly one
        LoopLM sub-module (prelude, recurrent.layers, or coda).

        Calling code must `del qwen3_for_causal_lm` after this returns so that
        the original Qwen3Model.layers ModuleList is freed, leaving LoopLM as
        the sole owner of each layer module.
        """
        cfg = self.config
        m = qwen3_for_causal_lm.model  # Qwen3Model

        # Extract all layers as a plain Python list (no Module ownership yet)
        all_layers = [m.layers[i] for i in range(len(m.layers))]

        p = cfg.prelude_layers
        r = cfg.recurrent_layers

        self.embed_tokens = m.embed_tokens
        self.rotary_emb = m.rotary_emb
        self.norm = m.norm
        self.lm_head = qwen3_for_causal_lm.lm_head

        self.prelude = nn.ModuleList(all_layers[:p])

        injection = LoopInjection(cfg.hidden_size, init_val=cfg.injection_init)
        loop_embed = (
            nn.Embedding(cfg.max_loop_iters, cfg.hidden_size)
            if cfg.use_loop_embeddings
            else None
        )
        self.recurrent = RecurrentBlock(
            layers=all_layers[p : p + r],
            injection=injection,
            loop_embed=loop_embed,
        )

        self.coda = nn.ModuleList(all_layers[p + r :])

        # Verify embedding/lm_head weight tie is preserved (Qwen3-1.7B has
        # tie_word_embeddings=True, meaning lm_head.weight IS embed_tokens.weight)
        if cfg.tie_word_embeddings:
            assert self.lm_head.weight is self.embed_tokens.weight, (
                "tie_word_embeddings=True but lm_head.weight and embed_tokens.weight "
                "are not the same tensor after construction. "
                "This indicates an unexpected change in the upstream model."
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[LoopLMConfig] = None,
        **hf_kwargs,
    ) -> "LoopLMForCausalLM":
        """
        Load a pretrained Qwen3 model and convert it to LoopLM.

        The 28 Qwen3 decoder layers are distributed as:
            layers[0 : prelude_layers]                      → prelude
            layers[prelude_layers : -coda_layers]           → recurrent block
            layers[-coda_layers :]                          → coda

        Only the LoopInjection alpha parameters are newly initialized (near 0).
        All other weights are loaded from the Qwen3 checkpoint.

        Args:
            model_name_or_path - HuggingFace model ID or local path,
                                  e.g. "Qwen/Qwen3-1.7B-Base"
            config             - LoopLMConfig; defaults to Qwen3-1.7B-Base split
            **hf_kwargs        - forwarded to Qwen3ForCausalLM.from_pretrained
                                  e.g. torch_dtype=torch.bfloat16, device_map="auto"

        Returns:
            LoopLMForCausalLM with pretrained weights and loop injection ready
            for fine-tuning.

        Example:
            model = LoopLMForCausalLM.from_pretrained(
                "Qwen/Qwen3-1.7B-Base",
                torch_dtype=torch.bfloat16,
            )
        """
        from transformers import Qwen3ForCausalLM

        if config is None:
            config = LoopLMConfig()

        qwen3 = Qwen3ForCausalLM.from_pretrained(model_name_or_path, **hf_kwargs)
        model = cls(config)
        model._build_from_qwen3(qwen3)
        del qwen3  # release the Qwen3 container; layers are now owned by LoopLM
        return model

    @classmethod
    def from_config(
        cls,
        config: Optional[LoopLMConfig] = None,
    ) -> "LoopLMForCausalLM":
        """
        Build a randomly initialized LoopLM (for training from scratch).

        Args:
            config - LoopLMConfig; defaults to Qwen3-1.7B-Base architecture

        Returns:
            LoopLMForCausalLM with random weights.
        """
        from transformers import Qwen3Config, Qwen3ForCausalLM

        if config is None:
            config = LoopLMConfig()

        qwen3_config = Qwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        qwen3 = Qwen3ForCausalLM(qwen3_config)
        model = cls(config)
        model._build_from_qwen3(qwen3)
        del qwen3
        return model

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full forward pass (training and evaluation).

        Args:
            input_ids      - (B, T) integer token IDs
            n_loops        - recurrent depth; defaults to config.max_loop_iters.
                             Increase at inference for more compute / harder problems.
            attention_mask - (B, T) binary mask; 1 = real token, 0 = padding.
                             If None, all positions are treated as real tokens.
            labels         - (B, T) integer token IDs for language modelling loss.
                             -100 entries are ignored in the loss.

        Returns:
            dict with keys:
                "logits"  : (B, T, vocab_size) float tensor
                "loss"    : scalar cross-entropy loss  [only if labels provided]

        Example:
            out = model(input_ids, n_loops=8, labels=input_ids)
            loss = out["loss"]
            loss.backward()
        """
        if n_loops is None:
            n_loops = self.config.max_loop_iters

        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.lm_head.weight.dtype

        # ── Embed ─────────────────────────────────────────────────────────────
        h = self.embed_tokens(input_ids).to(dtype)

        # ── Positional encoding (computed once, reused across all loop iters) ─
        position_ids = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        position_embeddings = self.rotary_emb(h, position_ids)      # (cos, sin)

        # ── Causal mask ───────────────────────────────────────────────────────
        causal_mask = _make_causal_mask(T, device=device, dtype=dtype)
        if attention_mask is not None:
            causal_mask = _extend_mask_for_padding(causal_mask, attention_mask, dtype)

        # ── Prelude ───────────────────────────────────────────────────────────
        for layer in self.prelude:
            h = layer(
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        # The Prelude output is the "encoded input" e.
        # Setting h_0 = e means T=1 with near-identity injection (alpha≈0)
        # approximates the original Qwen3 forward pass.
        e = h

        # ── Recurrent block ───────────────────────────────────────────────────
        h = self.recurrent(
            h=h,          # h_0 = e  (explicit initialization)
            e=e,
            n_loops=n_loops,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )

        # ── Coda ──────────────────────────────────────────────────────────────
        for layer in self.coda:
            h = layer(
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        # ── Final norm + LM head ──────────────────────────────────────────────
        h = self.norm(h)
        logits = self.lm_head(h)

        result: dict = {"logits": logits}

        if labels is not None:
            # Standard next-token prediction: shift logits and labels by 1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return result

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive token generation.

        Runs a full forward pass (prelude + T recurrent loops + coda) for each
        generated token.  This is correct but O(gen_len²) in sequence length;
        it is suitable for research and short outputs.

        Args:
            input_ids      - (B, T_prompt) prompt token IDs
            max_new_tokens - number of tokens to generate
            n_loops        - recurrent depth; defaults to config.max_loop_iters
            temperature    - softmax temperature (only used when do_sample=True)
            top_k          - restrict sampling to top-k logits (do_sample=True only)
            do_sample      - if False (default), use greedy decoding

        Returns:
            (B, T_prompt + max_new_tokens) token IDs including the prompt
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            out = self.forward(generated, n_loops=n_loops)
            next_logits = out["logits"][:, -1, :]  # (B, vocab_size)

            if do_sample:
                next_logits = next_logits / temperature
                if top_k is not None:
                    top_vals, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < top_vals[:, [-1]]] = float("-inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Total (or trainable-only) parameter count."""
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if only_trainable
            else self.parameters()
        )
        return sum(p.numel() for p in params)

    def injection_stats(self) -> dict:
        """
        Returns LTI injection statistics for monitoring training.

        Tracks mean_alpha (the current injection strength).
        At init: mean_alpha ≈ 0 (near-identity).
        As training progresses: mean_alpha rises toward 1 (strong injection).
        """
        return {"mean_alpha": self.recurrent.injection.mean_alpha}

    def gradient_checkpointing_enable(self) -> None:
        """
        Enable gradient checkpointing on the recurrent block.

        Each loop iteration's activations are recomputed during backward instead
        of being stored, trading ~33% extra compute for significantly lower peak
        activation memory. Essential for training with large n_loops or long sequences.
        """
        self.recurrent.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing (default)."""
        self.recurrent.gradient_checkpointing = False

    def freeze_base(self) -> None:
        """
        Freeze all weights except the LoopInjection parameters.

        Useful for a two-stage fine-tuning regime:
          1. Train only the injection to learn the recurrence mixing.
          2. Unfreeze and fine-tune the full model jointly.
        """
        for param in self.parameters():
            param.requires_grad = False
        for param in self.recurrent.injection.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
