"""
Qwen3-1.7B → LoopLM (Recurrent-Depth Transformer)
Finetuning with nvidia/Nemotron-Post-Training-Dataset-v2

Architecture overview:
  Input → [Prelude: first N layers, run once]
        → [Recurrent Block: middle layers, looped T times]
              h_{t+1} = A·h_t + B·e + TransformerLayers(h_t, e)
              Stability: ρ(A) < 1 enforced via LTI constraint
        → [Coda: last N layers, run once]
        → Output logits

Usage:
    # Single GPU
    python qwen3_looplm_train.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 qwen3_looplm_train.py
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopLMConfig:
    # Base model
    base_model_id: str = "Qwen/Qwen3-1.7B"

    # LoopLM architecture
    prelude_layers: int = 4      # first N Qwen layers used as Prelude (run once)
    coda_layers: int = 4         # last N Qwen layers used as Coda (run once)
    max_loops: int = 8           # max recurrent iterations at inference
    train_loops: int = 4         # loops used during training (curriculum)
    loop_curriculum: bool = True # gradually increase loops during training

    # LTI injection stability
    # A_discrete = exp(Δt · diag(-exp(log_A)))  →  ρ(A) < 1 always
    lti_init_log_A: float = 0.0   # init for log(-A_continuous diagonal)
    lti_init_delta: float = 0.1   # init for Δt scalar

    # Connect layer type (GatedResidual = stable across n_loops, from ClawMeBao/looped-lm)
    use_gated_connect: bool = True   # True=GatedResidual, False=LTI

    # LoRA on recurrent block (depth-wise adaptation)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    # Dataset
    dataset_id: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    dataset_splits: list = field(default_factory=lambda: ["math", "code", "stem"])
    max_samples_per_split: int = 5000   # set None for full dataset
    max_seq_len: int = 2048

    # Training
    output_dir: str = "./qwen3_looplm_checkpoint"
    num_epochs: int = 3
    batch_size: int = 2           # per-GPU
    grad_accum_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    bf16: bool = True             # use bfloat16 if available (A100/H100)
    save_every_steps: int = 500
    log_every_steps: int = 20
    seed: int = 42


CFG = LoopLMConfig()


# ──────────────────────────────────────────────────────────────────────────────
# LTI Stable Injection  (Prelude → Recurrent)
# ──────────────────────────────────────────────────────────────────────────────

class LTIInjection(nn.Module):
    """
    Implements the stable injection:
        h_{t+1} = A·h_t + B·e  (linear part, before transformer contribution)

    A_discrete = diag(exp(Δt · -exp(log_A)))
      → all diagonal elements in (0, 1) by construction → ρ(A) < 1 always

    B is a learned dense projection from encoder hidden dim to hidden dim.
    """

    def __init__(self, hidden_size: int, delta_init: float = 0.1, log_a_init: float = 0.0):
        super().__init__()
        # Parameterize log(-A_continuous) for each hidden dimension
        self.log_A = nn.Parameter(torch.full((hidden_size,), log_a_init))
        # Learned time-step scalar Δt (kept positive via softplus)
        self.log_delta = nn.Parameter(torch.tensor(math.log(delta_init)))
        # B projection
        self.B = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.B.weight, std=0.02)

    def get_A(self) -> torch.Tensor:
        """Return A_discrete diagonal vector, all in (0,1)."""
        delta = F.softplus(self.log_delta)          # Δt > 0
        A_cont = -torch.exp(self.log_A)             # A_continuous < 0
        return torch.exp(delta * A_cont)            # A_discrete ∈ (0, 1)

    def forward(
        self,
        h: torch.Tensor,   # (B, S, D) hidden state from previous loop
        e: torch.Tensor,   # (B, S, D) encoded prelude output
    ) -> torch.Tensor:
        A = self.get_A()   # (D,)
        return A * h + self.B(e)




# ──────────────────────────────────────────────────────────────────────────────
# GatedResidualConnectLayer  (từ ClawMeBao/looped-lm — depth-invariant injection)
# ──────────────────────────────────────────────────────────────────────────────

class GatedResidualConnectLayer(nn.Module):
    """
    Inspired by ClawMeBao/looped-lm Phase 0 results:
        MLP connect:          PPL 59 @ n_iter=1, degrades to 9860 @ n_iter=4 ❌
        GatedResidual:        PPL 98 @ n_iter=1, stays ~98 for n_iter 1–4    ✅

    Formula:
        gate   = sigmoid(W_g · [h_out; h_prev])      ∈ (0,1)^D
        output = gate * transform(h_out) + (1-gate) * h_prev

    This is depth-invariant: the gate learns to blend transformed output
    with the previous hidden state, naturally preventing drift across loops.
    Only ~6M params for d_model=2048 (0.3% of Qwen3-1.7B).
    """

    def __init__(self, hidden_size: int, expand: int = 2, dropout: float = 0.05):
        super().__init__()
        inner = hidden_size * expand
        # Transform branch: h_out → same space as h_prev
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, inner, bias=False),
            nn.SiLU(),
            nn.Linear(inner, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
        )
        # Gate: concat [h_out, h_prev] → scalar gate per dim
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Init gate bias to 0 so gate≈0.5 at start → balanced blend
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.02)

    def forward(self, h_out: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_out:  (B, S, D) — output of loop block this iteration
            h_prev: (B, S, D) — hidden state from previous iteration (= h_prev loop input)
        Returns:
            (B, S, D) — blended state to feed into next iteration
        """
        gate = torch.sigmoid(self.gate_proj(
            self.dropout(torch.cat([h_out, h_prev], dim=-1))
        ))
        return gate * self.transform(h_out) + (1.0 - gate) * h_prev


# ──────────────────────────────────────────────────────────────────────────────
# LoRA layer  (applied depth-wise inside the recurrent block)
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank ΔW = B·A."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base weights
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base + lora


def apply_lora_to_qwen_layer(layer: nn.Module, rank: int, alpha: float, dropout: float):
    """Replace q_proj, k_proj, v_proj, o_proj in a Qwen attention block with LoRA variants."""
    attn = layer.self_attn
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if hasattr(attn, proj_name):
            orig = getattr(attn, proj_name)
            setattr(attn, proj_name, LoRALinear(orig, rank, alpha, dropout))
    return layer


# ──────────────────────────────────────────────────────────────────────────────
# LoopLM  (wraps Qwen3-1.7B layers in Prelude / Recurrent / Coda structure)
# ──────────────────────────────────────────────────────────────────────────────

class LoopLM(nn.Module):
    """
    Converts a loaded Qwen3 model into a Recurrent-Depth Transformer:

        [embed] → [Prelude layers 0..P-1]
                → [LTI Injection]
                → [Recurrent layers P..L-C-1]  ×  n_loops
                → [Coda layers L-C..L-1]
                → [lm_head] → logits
    """

    def __init__(self, qwen_model, cfg: LoopLMConfig):
        super().__init__()
        self.cfg = cfg
        model = qwen_model

        # Pull out shared components
        self.embed_tokens = model.model.embed_tokens
        self.rotary_emb = model.model.rotary_emb   # needed to pre-compute (cos, sin) per forward
        self.norm = model.model.norm
        self.lm_head = model.lm_head

        layers = list(model.model.layers)
        n = len(layers)
        P = cfg.prelude_layers
        C = cfg.coda_layers
        assert P + C < n, f"prelude ({P}) + coda ({C}) must be < total layers ({n})"

        self.prelude = nn.ModuleList(layers[:P])
        self.recurrent = nn.ModuleList(layers[P : n - C])
        self.coda = nn.ModuleList(layers[n - C :])

        hidden_size = model.config.hidden_size
        base_dtype = next(model.parameters()).dtype

        # GatedResidual connect layer (depth-invariant, from ClawMeBao/looped-lm)
        # Replaces LTI injection — more stable across varying n_loops.
        # Also keep LTI as fallback via cfg.use_gated_connect flag.
        if getattr(cfg, 'use_gated_connect', True):
            self.injection = GatedResidualConnectLayer(hidden_size)
        else:
            self.injection = LTIInjection(
                hidden_size,
                delta_init=cfg.lti_init_delta,
                log_a_init=cfg.lti_init_log_A,
            )

        # Optionally wrap recurrent layers with LoRA
        if cfg.use_lora:
            self.recurrent = nn.ModuleList([
                apply_lora_to_qwen_layer(layer, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
                for layer in self.recurrent
            ])

        # Cast all trainable modules (injection + LoRA adapters) to match the base
        # model dtype (e.g. bfloat16) to avoid dtype mismatch during forward pass.
        self.injection.to(base_dtype)
        for layer in self.recurrent:
            attn = layer.self_attn
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, 'lora_A'):
                    proj.lora_A.to(base_dtype)
                    proj.lora_B.to(base_dtype)

        # Freeze everything except injection + LoRA params
        self._freeze_base()

        self._eos_token_id = -1   # overridden by set_eos_token_id() before generate()

        log.info(
            f"LoopLM ready — Prelude: {P} layers | "
            f"Recurrent: {len(self.recurrent)} layers | "
            f"Coda: {C} layers"
        )
        log.info(f"Trainable params: {self._count_trainable():,}")

    def _freeze_base(self):
        """Freeze all base weights; only injection and LoRA adapters are trained."""
        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for layer in self.prelude:
            for p in layer.parameters():
                p.requires_grad = False
        for layer in self.coda:
            for p in layer.parameters():
                p.requires_grad = False
        # Recurrent base weights already frozen by LoRALinear; injection is trainable.

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _run_layers(self, hidden_states, position_ids, layers, cache_position=None, position_embeddings=None):
        """Run a list of transformer layers sequentially.

        Passes pre-computed position_embeddings (cos, sin) to each layer, as
        required by transformers ≥ 5.x where Qwen3Attention.forward requires
        position_embeddings as a positional argument.

        Handles both old (tuple return) and new (tensor return) transformers API.
        """
        for layer in layers:
            out = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        n_loops: Optional[int] = None,
    ):
        if n_loops is None:
            n_loops = self.cfg.train_loops

        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        hidden = self.embed_tokens(input_ids)

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(S, device=device)

        # Pre-compute rotary (cos, sin) once — shared across all layers and loops
        position_embeddings = self.rotary_emb(hidden, position_ids)

        # ── Prelude (run once) ──────────────────────────────────────────────
        e = self._run_layers(hidden, position_ids, self.prelude, cache_position, position_embeddings)

        # ── Recurrent block (looped n_loops times) ──────────────────────────
        h = e.clone()   # initialise hidden state from prelude output
        for _ in range(n_loops):
            if isinstance(self.injection, GatedResidualConnectLayer):
                # GatedResidual: run transformer first, then blend output with pre-loop state
                # gate*transform(h_out) + (1-gate)*h_prev
                h_prev = h
                h_out = self._run_layers(h, position_ids, self.recurrent, cache_position, position_embeddings)
                h = self.injection(h_out, h_prev)
            else:
                # LTI: inject prelude encoding into hidden state before transformer
                # A*h + B*e
                h_in = self.injection(h, e)
                h = self._run_layers(h_in, position_ids, self.recurrent, cache_position, position_embeddings)

        # ── Coda (run once) ──────────────────────────────────────────────────
        out = self._run_layers(h, position_ids, self.coda, cache_position, position_embeddings)

        # Final norm + head
        out = self.norm(out)
        logits = self.lm_head(out)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.LongTensor:
        if n_loops is None:
            n_loops = self.cfg.max_loops
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(generated, n_loops=n_loops)
            next_logits = out["logits"][:, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            # Top-p sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cum_probs - sorted_probs > top_p
            sorted_probs[remove_mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(1, torch.multinomial(sorted_probs, num_samples=1))  # [B, 1]
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == self._eos_token_id:
                break
        return generated

    def set_eos_token_id(self, eos_id: int):
        self._eos_token_id = eos_id


# ──────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ──────────────────────────────────────────────────────────────────────────────

def load_nemotron_v2(cfg: LoopLMConfig, tokenizer):
    """
    Load nvidia/Nemotron-Post-Training-Dataset-v2 (SFT config).
    Schema: {uuid, license, generator, version, category, reasoning, messages[{role, content}]}
    Returns a list of tokenized (input_ids, labels) pairs.
    """
    log.info(f"Loading Nemotron-v2 splits: {cfg.dataset_splits}")
    all_examples = []

    for split in cfg.dataset_splits:
        log.info(f"  Fetching split '{split}'...")
        ds = load_dataset(
            cfg.dataset_id,
            "SFT",
            split=split,
            streaming=False,
        )
        if cfg.max_samples_per_split:
            ds = ds.select(range(min(cfg.max_samples_per_split, len(ds))))

        for sample in ds:
            messages = sample.get("messages", [])
            if not messages:
                continue
            # Format messages into a single prompt using Qwen chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback: manual format
                text = ""
                for m in messages:
                    role, content = m["role"], m["content"]
                    text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            enc = tokenizer(
                text,
                truncation=True,
                max_length=cfg.max_seq_len,
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]
            if len(ids) < 8:
                continue

            # Labels: -100 on all tokens except assistant turns
            labels = ids.clone()
            labels = mask_non_assistant(labels, ids, tokenizer)

            all_examples.append({"input_ids": ids, "labels": labels})

    log.info(f"Total examples loaded: {len(all_examples)}")
    return all_examples


def mask_non_assistant(labels: torch.Tensor, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Set labels to -100 for system/user tokens.
    Keeps loss only on assistant-generated tokens.
    Heuristic: mask everything before first assistant response end.
    """
    # Simple approach: find <|im_start|>assistant token boundaries
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if im_start_id is None or im_end_id is None:
        return labels  # fallback: train on all tokens

    ids = input_ids.tolist()
    in_assistant = False
    for i, tok in enumerate(ids):
        if tok == im_start_id:
            # Peek: check next tokens for "assistant"
            window = tokenizer.decode(ids[i : i + 4])
            in_assistant = "assistant" in window
        if not in_assistant:
            labels[i] = -100
        if tok == im_end_id:
            in_assistant = False
    return labels


class NemotronCollator:
    """Pads a batch of variable-length examples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(ex["input_ids"].shape[0] for ex in batch)
        input_ids_padded, labels_padded = [], []
        for ex in batch:
            n = ex["input_ids"].shape[0]
            pad = max_len - n
            input_ids_padded.append(
                F.pad(ex["input_ids"], (0, pad), value=self.pad_token_id)
            )
            labels_padded.append(
                F.pad(ex["labels"], (0, pad), value=-100)
            )
        return {
            "input_ids": torch.stack(input_ids_padded),
            "labels": torch.stack(labels_padded),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    """Init DDP if running under torchrun, else return rank=0."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        return rank, dist.get_world_size()
    return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_loop_schedule(step: int, total_steps: int, cfg: LoopLMConfig) -> int:
    """
    Curriculum: linearly ramp loops from 1 to cfg.train_loops over first 30% of training,
    then fix at cfg.train_loops.
    """
    if not cfg.loop_curriculum:
        return cfg.train_loops
    ramp_end = int(0.3 * total_steps)
    if step >= ramp_end:
        return cfg.train_loops
    frac = step / max(ramp_end, 1)
    return max(1, round(1 + frac * (cfg.train_loops - 1)))


def train(cfg: LoopLMConfig = CFG):
    rank, world_size = setup_distributed()
    is_main = rank == 0

    torch.manual_seed(cfg.seed + rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.bf16 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # ── Load base tokenizer & model ─────────────────────────────────────────
    if is_main:
        log.info(f"Loading base model: {cfg.base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # ── Wrap in LoopLM ───────────────────────────────────────────────────────
    model = LoopLM(qwen, cfg).to(device)
    model.set_eos_token_id(tokenizer.eos_token_id)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    inner_model = model.module if world_size > 1 else model

    # ── Dataset ──────────────────────────────────────────────────────────────
    examples = load_nemotron_v2(cfg, tokenizer)
    collator = NemotronCollator(tokenizer.pad_token_id)

    # Simple train split (90/10)
    n_train = int(0.9 * len(examples))
    train_data = examples[:n_train]
    val_data = examples[n_train:]

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = (len(train_loader) * cfg.num_epochs) // cfg.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(not use_bf16 and torch.cuda.is_available()))

    if is_main:
        log.info(f"Training for {total_steps} optimizer steps over {cfg.num_epochs} epochs")
        log.info(f"Loop curriculum: {cfg.loop_curriculum} | max training loops: {cfg.train_loops}")
        os.makedirs(cfg.output_dir, exist_ok=True)

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            n_loops = compute_loop_schedule(global_step, total_steps, cfg)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                out = inner_model(input_ids, labels=labels, n_loops=n_loops)
                loss = out["loss"] / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            running_loss += loss.item() * cfg.grad_accum_steps

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % cfg.log_every_steps == 0:
                    avg_loss = running_loss / cfg.log_every_steps
                    if hasattr(inner_model.injection, 'get_A'):
                        connect_info = f"ρ(A) max: {inner_model.injection.get_A().max().item():.4f}"
                    else:
                        gate_mean = torch.sigmoid(inner_model.injection.gate_proj.bias).mean().item()
                        connect_info = f"gate mean: {gate_mean:.4f}"
                    log.info(
                        f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | Loops: {n_loops} | "
                        f"{connect_info} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0

                if is_main and global_step % cfg.save_every_steps == 0:
                    _save_checkpoint(inner_model, tokenizer, cfg, global_step)

        # ── Validation ────────────────────────────────────────────────────────
        if is_main:
            val_loss = _evaluate(inner_model, val_loader, device, dtype, cfg)
            log.info(f"Epoch {epoch+1} complete | Val loss: {val_loss:.4f}")

    if is_main:
        _save_checkpoint(inner_model, tokenizer, cfg, global_step, final=True)
        log.info("Training complete.")

    cleanup_distributed()


def _evaluate(model, loader, device, dtype, cfg: LoopLMConfig) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                out = model(input_ids, labels=labels, n_loops=cfg.train_loops)
            if out["loss"] is not None:
                total_loss += out["loss"].item()
                n_batches += 1
            if n_batches >= 50:   # quick eval
                break
    model.train()
    return total_loss / max(n_batches, 1)


def _save_checkpoint(model, tokenizer, cfg: LoopLMConfig, step: int, final: bool = False):
    tag = "final" if final else f"step_{step}"
    save_path = os.path.join(cfg.output_dir, tag)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "looplm_state_dict.pt"))
    tokenizer.save_pretrained(save_path)
    log.info(f"Checkpoint saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check (no dataset needed)
# ──────────────────────────────────────────────────────────────────────────────

def sanity_check():
    """Smoke-test: load model, run one forward pass, verify ρ(A) < 1."""
    log.info("Running sanity check (no dataset)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = AutoModelForCausalLM.from_pretrained(
        CFG.base_model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model = LoopLM(qwen, CFG).to(device)

    text = "Explain why the sky is blue."
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model(ids, n_loops=4)
    log.info(f"Forward pass OK | logits: {out['logits'].shape}")

    if hasattr(model.injection, 'get_A'):
        A = model.injection.get_A()
        log.info(f"Spectral radius ρ(A) max = {A.max().item():.6f}  (must be < 1)")
        assert A.max().item() < 1.0, "Stability violated!"
    else:
        log.info("Using GatedResidualConnectLayer — LTI spectral check skipped")
        # Verify gate is in valid range
        with torch.no_grad():
            dummy_h = torch.zeros(1, 4, model.injection.gate_proj.in_features // 2, device=device, dtype=dtype)
            gate_val = torch.sigmoid(model.injection.gate_proj(
                torch.cat([dummy_h, dummy_h], dim=-1)
            ))
            log.info(f"Gate init range: [{gate_val.min().item():.3f}, {gate_val.max().item():.3f}] (expect ~0.5)")

    generated = model.generate(ids, max_new_tokens=20, n_loops=4)
    log.info(f"Generated: {tokenizer.decode(generated[0], skip_special_tokens=True)}")
    log.info("Sanity check PASSED ✓")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="Run sanity check only")
    parser.add_argument("--prelude_layers", type=int, default=CFG.prelude_layers)
    parser.add_argument("--coda_layers", type=int, default=CFG.coda_layers)
    parser.add_argument("--train_loops", type=int, default=CFG.train_loops)
    parser.add_argument("--max_loops", type=int, default=CFG.max_loops)
    parser.add_argument("--splits", nargs="+", default=CFG.dataset_splits)
    parser.add_argument("--max_samples", type=int, default=CFG.max_samples_per_split)
    parser.add_argument("--epochs", type=int, default=CFG.num_epochs)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lora_rank", type=int, default=CFG.lora_rank)
    parser.add_argument("--output_dir", type=str, default=CFG.output_dir)
    args = parser.parse_args()

    CFG.prelude_layers = args.prelude_layers
    CFG.coda_layers = args.coda_layers
    CFG.train_loops = args.train_loops
    CFG.max_loops = args.max_loops
    CFG.dataset_splits = args.splits
    CFG.max_samples_per_split = args.max_samples
    CFG.num_epochs = args.epochs
    CFG.learning_rate = args.lr
    CFG.batch_size = args.batch_size
    CFG.lora_rank = args.lora_rank
    CFG.output_dir = args.output_dir

    if args.sanity:
        sanity_check()
    else:
        train(CFG)
