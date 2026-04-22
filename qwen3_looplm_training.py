"""
Qwen3-1.7B → LoopLM (Recurrent-Depth Transformer)
Finetuning with nvidia/Nemotron-Post-Training-Dataset-v2

Architecture overview:
  Input → [Prelude: first N layers, run once]
        → [Recurrent Block: middle layers, looped T times]
              GatedResidual: gate*transform(h_out) + (1-gate)*h_prev
              LTI fallback:  A·h + B·e  (ρ(A) < 1 enforced)
        → [Coda: last N layers, run once]
        → Output logits

Usage:
    python qwen3_looplm_train.py --sanity
    python qwen3_looplm_train.py --splits math code --max_samples 2000
    torchrun --nproc_per_node=4 qwen3_looplm_train.py
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

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
    base_model_id: str = "Qwen/Qwen3-1.7B"
    prelude_layers: int = 4
    coda_layers: int = 4
    max_loops: int = 8
    train_loops: int = 4
    loop_curriculum: bool = True

    lti_init_log_A: float = 0.0
    lti_init_delta: float = 0.1
    use_gated_connect: bool = True

    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    # nvidia/Nemotron-Post-Training-Dataset-v2 splits (no config name needed):
    #   math(239k), code(175k), stem(355k), chat(627k), multilingual_*/ja/de/it/es/fr
    dataset_id: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    dataset_splits: list = field(default_factory=lambda: ["math", "code", "stem"])
    max_samples_per_split: int = 5000
    max_seq_len: int = 2048

    output_dir: str = "./qwen3_looplm_checkpoint"
    num_epochs: int = 3
    batch_size: int = 2
    grad_accum_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    bf16: bool = True
    save_every_steps: int = 500
    log_every_steps: int = 20
    seed: int = 42


CFG = LoopLMConfig()


# ──────────────────────────────────────────────────────────────────────────────
# LTI Stable Injection
# ──────────────────────────────────────────────────────────────────────────────

class LTIInjection(nn.Module):
    """h_{t+1} = A·h + B·e   A_discrete ∈ (0,1) by construction → ρ(A) < 1"""

    def __init__(self, hidden_size, delta_init=0.1, log_a_init=0.0):
        super().__init__()
        self.log_A    = nn.Parameter(torch.full((hidden_size,), log_a_init))
        self.log_delta = nn.Parameter(torch.tensor(math.log(delta_init)))
        self.B        = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.B.weight, std=0.02)

    def get_A(self):
        return torch.exp(F.softplus(self.log_delta) * -torch.exp(self.log_A))

    def forward(self, h, e):
        return self.get_A() * h + self.B(e)


# ──────────────────────────────────────────────────────────────────────────────
# GatedResidual Connect Layer  (ClawMeBao/looped-lm — depth-invariant)
# ──────────────────────────────────────────────────────────────────────────────

class GatedResidualConnectLayer(nn.Module):
    """
    Phase 0 results (ClawMeBao/looped-lm):
        MLP:          PPL 59→9860 as n_iter 1→4  ❌
        GatedResidual: PPL 98→~98 as n_iter 1→4  ✅

    gate   = sigmoid(W_g · [h_out; h_prev])
    output = gate * transform(h_out) + (1-gate) * h_prev
    """

    def __init__(self, hidden_size, expand=2, dropout=0.05):
        super().__init__()
        inner = hidden_size * expand
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, inner, bias=False),
            nn.SiLU(),
            nn.Linear(inner, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
        )
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.dropout   = nn.Dropout(dropout)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.02)

    def forward(self, h_out, h_prev):
        gate = torch.sigmoid(self.gate_proj(
            self.dropout(torch.cat([h_out, h_prev], dim=-1))
        ))
        return gate * self.transform(h_out) + (1.0 - gate) * h_prev


# ──────────────────────────────────────────────────────────────────────────────
# LoRA
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, dropout):
        super().__init__()
        self.linear = linear
        self.scale  = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Linear(in_f, rank,  bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def apply_lora_to_qwen_layer(layer, rank, alpha, dropout):
    attn = layer.self_attn
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if hasattr(attn, name):
            setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha, dropout))
    return layer


# ──────────────────────────────────────────────────────────────────────────────
# LoopLM
# ──────────────────────────────────────────────────────────────────────────────

class LoopLM(nn.Module):
    def __init__(self, qwen_model, cfg: LoopLMConfig):
        super().__init__()
        self.cfg = cfg
        m = qwen_model

        self.embed_tokens   = m.model.embed_tokens
        self.rotary_emb     = m.model.rotary_emb
        self.norm           = m.model.norm
        self.lm_head        = m.lm_head

        layers = list(m.model.layers)
        n, P, C = len(layers), cfg.prelude_layers, cfg.coda_layers
        assert P + C < n, f"prelude+coda ({P}+{C}) >= total layers ({n})"

        self.prelude   = nn.ModuleList(layers[:P])
        self.recurrent = nn.ModuleList(layers[P : n - C])
        self.coda      = nn.ModuleList(layers[n - C :])

        hidden_size = m.config.hidden_size
        base_dtype  = next(m.parameters()).dtype

        self.injection = (
            GatedResidualConnectLayer(hidden_size)
            if cfg.use_gated_connect
            else LTIInjection(hidden_size, cfg.lti_init_delta, cfg.lti_init_log_A)
        )

        if cfg.use_lora:
            self.recurrent = nn.ModuleList([
                apply_lora_to_qwen_layer(layer, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
                for layer in self.recurrent
            ])

        # Match dtype to base model
        self.injection.to(base_dtype)
        for layer in self.recurrent:
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(layer.self_attn, name, None)
                if proj is not None and hasattr(proj, "lora_A"):
                    proj.lora_A.to(base_dtype)
                    proj.lora_B.to(base_dtype)

        self._freeze_base()
        self._eos_token_id = -1

        log.info(f"LoopLM: Prelude={P} | Recurrent={len(self.recurrent)} | Coda={C}")
        log.info(f"Trainable: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _freeze_base(self):
        for m in (self.embed_tokens, self.norm, self.lm_head):
            for p in m.parameters():
                p.requires_grad = False
        for layer in list(self.prelude) + list(self.coda):
            for p in layer.parameters():
                p.requires_grad = False

    def _run_layers(self, h, position_ids, layers, cache_position=None, position_embeddings=None):
        for layer in layers:
            out = layer(
                h,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
            h = out[0] if isinstance(out, tuple) else out
        return h

    def forward(self, input_ids, attention_mask=None, labels=None, n_loops=None):
        if n_loops is None:
            n_loops = self.cfg.train_loops

        B, S   = input_ids.shape
        device = input_ids.device
        hidden = self.embed_tokens(input_ids)

        position_ids       = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        cache_position     = torch.arange(S, device=device)
        position_embeddings = self.rotary_emb(hidden, position_ids)

        e = self._run_layers(hidden, position_ids, self.prelude,
                             cache_position, position_embeddings)

        h = e.clone()
        for _ in range(n_loops):
            if isinstance(self.injection, GatedResidualConnectLayer):
                h_prev = h
                h_out  = self._run_layers(h, position_ids, self.recurrent,
                                          cache_position, position_embeddings)
                h      = self.injection(h_out, h_prev)
            else:
                h = self._run_layers(self.injection(h, e), position_ids, self.recurrent,
                                     cache_position, position_embeddings)

        out    = self._run_layers(h, position_ids, self.coda, cache_position, position_embeddings)
        out    = self.norm(out)
        logits = self.lm_head(out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, n_loops=None,
                 temperature=1.0, top_p=0.9):
        if n_loops is None:
            n_loops = self.cfg.max_loops
        self.eval()
        gen = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(gen, n_loops=n_loops)["logits"][:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            sp[torch.cumsum(sp, -1) - sp > top_p] = 0.0
            sp /= sp.sum(-1, keepdim=True)
            tok = si.gather(1, torch.multinomial(sp, 1))
            gen = torch.cat([gen, tok], dim=-1)
            if tok.item() == self._eos_token_id:
                break
        return gen

    def set_eos_token_id(self, eos_id):
        self._eos_token_id = eos_id


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  —  streaming from Nemotron-Post-Training-Dataset-v2
# ──────────────────────────────────────────────────────────────────────────────

def _mask_non_assistant(labels, input_ids, tokenizer):
    """Set labels=-100 for non-assistant tokens (system/user)."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_start is None or im_end is None:
        return labels
    ids, in_asst = input_ids.tolist(), False
    for i, tok in enumerate(ids):
        if tok == im_start:
            in_asst = "assistant" in tokenizer.decode(ids[i : i + 4])
        if not in_asst:
            labels[i] = -100
        if tok == im_end:
            in_asst = False
    return labels


class NemotronStreamDataset(IterableDataset):
    """
    Streams one split of Nemotron-Post-Training-Dataset-v2.

    Key fix: the dataset has NO named config ('SFT' does not exist).
    Load with:  load_dataset(dataset_id, split=split_name, streaming=True)
    """

    def __init__(self, cfg: LoopLMConfig, tokenizer, split: str):
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.split     = split

    def __iter__(self) -> Iterator[dict]:
        tok = self.tokenizer
        cfg = self.cfg

        ds = load_dataset(
            cfg.dataset_id,
            split=self.split,     # ← correct: no config_name arg
            streaming=True,
            trust_remote_code=False,
        )

        count = 0
        for sample in ds:
            if cfg.max_samples_per_split and count >= cfg.max_samples_per_split:
                break

            messages = sample.get("messages", [])
            if not messages:
                continue

            try:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                text = "".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                    for m in messages
                )

            enc = tok(text, truncation=True, max_length=cfg.max_seq_len, return_tensors="pt")
            ids = enc["input_ids"][0]
            if len(ids) < 8:
                continue

            labels = _mask_non_assistant(ids.clone(), ids, tok)
            count += 1
            yield {"input_ids": ids, "labels": labels}


class NemotronCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        iids, labs = [], []
        for x in batch:
            p = max_len - x["input_ids"].shape[0]
            iids.append(F.pad(x["input_ids"], (0, p), value=self.pad_id))
            labs.append(F.pad(x["labels"],    (0, p), value=-100))
        return {"input_ids": torch.stack(iids), "labels": torch.stack(labs)}


def build_dataloaders(cfg: LoopLMConfig, tokenizer):
    collator    = NemotronCollator(tokenizer.pad_token_id)
    all_examples = []

    for split in cfg.dataset_splits:
        log.info(f"  Streaming '{split}' (≤{cfg.max_samples_per_split} samples)...")
        before = len(all_examples)
        for item in NemotronStreamDataset(cfg, tokenizer, split):
            all_examples.append(item)
        log.info(f"  → {len(all_examples) - before} examples from '{split}'")

    log.info(f"Dataset total: {len(all_examples)} examples")

    n_train    = int(0.9 * len(all_examples))
    train_data = all_examples[:n_train]
    val_data   = all_examples[n_train:]

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        return rank, dist.get_world_size()
    return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def loop_schedule(step, total, cfg):
    if not cfg.loop_curriculum:
        return cfg.train_loops
    ramp = int(0.3 * total)
    if step >= ramp:
        return cfg.train_loops
    return max(1, round(1 + step / max(ramp, 1) * (cfg.train_loops - 1)))


def train(cfg: LoopLMConfig = CFG):
    rank, world_size = setup_distributed()
    is_main = rank == 0

    torch.manual_seed(cfg.seed + rank)
    device   = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.bf16 and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32

    if is_main:
        log.info(f"Loading {cfg.base_model_id}")

    tok  = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    qwen  = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=dtype, trust_remote_code=True)
    model = LoopLM(qwen, cfg).to(device)
    model.set_eos_token_id(tok.eos_token_id)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    inner = model.module if world_size > 1 else model

    train_loader, val_loader = build_dataloaders(cfg, tok)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    total_steps = (len(train_loader) * cfg.num_epochs) // cfg.grad_accum_steps
    scheduler   = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    scaler      = torch.cuda.amp.GradScaler(enabled=(not use_bf16 and torch.cuda.is_available()))

    if is_main:
        log.info(f"{total_steps} optimizer steps | {cfg.num_epochs} epochs | "
                 f"bs={cfg.batch_size} accum={cfg.grad_accum_steps}")
        os.makedirs(cfg.output_dir, exist_ok=True)

    gstep, running_loss = 0, 0.0
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        model.train()
        for bidx, batch in enumerate(train_loader):
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            nloops = loop_schedule(gstep, total_steps, cfg)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                loss = inner(ids, labels=labels, n_loops=nloops)["loss"] / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg.grad_accum_steps

            if (bidx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                gstep += 1

                if is_main and gstep % cfg.log_every_steps == 0:
                    inj  = inner.injection
                    info = (f"ρ(A)={inj.get_A().max().item():.4f}" if hasattr(inj, "get_A")
                            else f"gate={torch.sigmoid(inj.gate_proj.bias).mean().item():.3f}")
                    log.info(f"Ep{epoch+1} {gstep}/{total_steps} "
                             f"loss={running_loss/cfg.log_every_steps:.4f} "
                             f"loops={nloops} {info} lr={scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

                if is_main and gstep % cfg.save_every_steps == 0:
                    _save(inner, tok, cfg, gstep)

        if is_main:
            vl = _eval(inner, val_loader, device, dtype, cfg)
            log.info(f"Epoch {epoch+1} done | val_loss={vl:.4f}")

    if is_main:
        _save(inner, tok, cfg, gstep, final=True)
        log.info("Done.")

    cleanup_distributed()


def _eval(model, loader, device, dtype, cfg):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                loss = model(batch["input_ids"].to(device),
                             labels=batch["labels"].to(device),
                             n_loops=cfg.train_loops)["loss"]
            if loss is not None:
                total += loss.item(); n += 1
            if n >= 50: break
    model.train()
    return total / max(n, 1)


def _save(model, tok, cfg, step, final=False):
    path = os.path.join(cfg.output_dir, "final" if final else f"step_{step}")
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "looplm_state_dict.pt"))
    tok.save_pretrained(path)
    log.info(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────────────────

def sanity_check():
    log.info("Sanity check...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    tok  = AutoTokenizer.from_pretrained(CFG.base_model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    qwen  = AutoModelForCausalLM.from_pretrained(CFG.base_model_id, torch_dtype=dtype, trust_remote_code=True)
    model = LoopLM(qwen, CFG).to(device)
    model.set_eos_token_id(tok.eos_token_id)

    ids = tok("Explain why the sky is blue.", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model(ids, n_loops=4)
    log.info(f"Forward OK | logits: {out['logits'].shape}")

    inj = model.injection
    if hasattr(inj, "get_A"):
        A = inj.get_A()
        assert A.max().item() < 1.0, "LTI stability violated!"
        log.info(f"ρ(A) max = {A.max().item():.6f} ✓")
    else:
        with torch.no_grad():
            d = inj.gate_proj.in_features // 2
            g = torch.sigmoid(inj.gate_proj(torch.zeros(1, 4, d*2, device=device, dtype=dtype)))
        log.info(f"GatedResidual gate ∈ [{g.min():.3f}, {g.max():.3f}] (expect ~0.5) ✓")

    gen = model.generate(ids, max_new_tokens=20, n_loops=4)
    log.info(f"Generated: {tok.decode(gen[0], skip_special_tokens=True)}")
    log.info("Sanity check PASSED ✓")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sanity",         action="store_true")
    p.add_argument("--prelude_layers", type=int,   default=CFG.prelude_layers)
    p.add_argument("--coda_layers",    type=int,   default=CFG.coda_layers)
    p.add_argument("--train_loops",    type=int,   default=CFG.train_loops)
    p.add_argument("--max_loops",      type=int,   default=CFG.max_loops)
    p.add_argument("--splits",         nargs="+",  default=CFG.dataset_splits)
    p.add_argument("--max_samples",    type=int,   default=CFG.max_samples_per_split)
    p.add_argument("--epochs",         type=int,   default=CFG.num_epochs)
    p.add_argument("--lr",             type=float, default=CFG.learning_rate)
    p.add_argument("--batch_size",     type=int,   default=CFG.batch_size)
    p.add_argument("--lora_rank",      type=int,   default=CFG.lora_rank)
    p.add_argument("--output_dir",     type=str,   default=CFG.output_dir)
    p.add_argument("--no_gated",       action="store_true", help="Use LTI instead of GatedResidual")
    args = p.parse_args()

    CFG.prelude_layers        = args.prelude_layers
    CFG.coda_layers           = args.coda_layers
    CFG.train_loops           = args.train_loops
    CFG.max_loops             = args.max_loops
    CFG.dataset_splits        = args.splits
    CFG.max_samples_per_split = args.max_samples
    CFG.num_epochs            = args.epochs
    CFG.learning_rate         = args.lr
    CFG.batch_size            = args.batch_size
    CFG.lora_rank             = args.lora_rank
    CFG.output_dir            = args.output_dir
    if args.no_gated:
        CFG.use_gated_connect = False

    if args.sanity:
        sanity_check()
    else:
        train(CFG)
