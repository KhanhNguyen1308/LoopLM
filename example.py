"""
LoopLM example: building a Recurrent-Depth Transformer from Qwen3-1.7B-Base.

This script shows three ways to use LoopLM:
  1. Load from pretrained Qwen3 weights
  2. Build from scratch (random init)
  3. Inspect injection statistics during a mock training step
"""

import torch
from transformers.optimization import Adafactor
from loop_lm import LoopLMConfig, LoopLMForCausalLM


# ─── Config ───────────────────────────────────────────────────────────────────
# Default config matches Qwen3-1.7B-Base (28 layers split 4/20/4).
# Adjust splits to taste:
#   - More prelude/coda → richer single-pass representations
#   - More recurrent   → more loopable depth; adjust max_loop_iters accordingly
cfg = LoopLMConfig(
    prelude_layers=4,
    recurrent_layers=20,
    coda_layers=4,
    max_loop_iters=8,
    use_loop_embeddings=False,  # set True to give each loop step a distinct bias
)

print("=== Option 1: Build from scratch (no pretrained download) ===")
model = LoopLMForCausalLM.from_config(cfg)
print(f"Total parameters : {model.num_parameters():,}")
print(f"Trainable params : {model.num_parameters(only_trainable=True):,}")
print(f"Config           : {cfg}")

# ─── Mock forward pass ────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32
model  = model.to(device=device, dtype=dtype)

B, T = 2, 16  # batch=2, seq_len=16
input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
labels    = input_ids.clone()

# Inference-only passes: no_grad avoids building computation graphs
with torch.no_grad():
    # T=1 loop → approximates original Qwen3 (alpha≈0 at init)
    out_1 = model(input_ids, n_loops=1)
    print(f"\n[n_loops=1] logits shape : {out_1['logits'].shape}")

    # T=8 loops → deeper reasoning
    out_8 = model(input_ids, n_loops=8)
    print(f"[n_loops=8] logits shape : {out_8['logits'].shape}")

    # Loss computation
    out_loss = model(input_ids, n_loops=4, labels=labels)
    print(f"[n_loops=4] loss         : {out_loss['loss'].item():.4f}")

# Injection statistics (before training: alpha ≈ 0)
stats = model.injection_stats()
print(f"\nInjection stats (pre-training): {stats}")
# Expected: mean_alpha ≈ 4.5e-5 (sigmoid(-10))

# Free inference activations before training
del out_1, out_8, out_loss
torch.cuda.empty_cache()

# ─── Mock training step ───────────────────────────────────────────────────────
print("\n=== Mock training step ===")
# Adafactor uses factored second moments — optimizer state is ~hundreds of MB
# instead of AdamW's 2 × model_size (~6.9 GB), fitting comfortably on 16 GB.
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
)
model.gradient_checkpointing_enable()
model.train()
out = model(input_ids, n_loops=4, labels=labels)
out["loss"].backward()
del out
torch.cuda.empty_cache()
optimizer.step()
optimizer.zero_grad()

stats_post = model.injection_stats()
print(f"Injection stats (post-step)   : {stats_post}")
print(f"Loss after one step           : {out['loss'].item():.4f}")

# ─── Generation ───────────────────────────────────────────────────────────────
print("\n=== Greedy generation ===")
model.eval()
prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
generated = model.generate(prompt, max_new_tokens=8, n_loops=4)
print(f"Prompt shape    : {prompt.shape}     → {prompt[0].tolist()}")
print(f"Generated shape : {generated.shape}  → {generated[0].tolist()}")

# ─── Two-stage fine-tuning demo ───────────────────────────────────────────────
print("\n=== Two-stage fine-tuning: freeze base, train injection only ===")
model.freeze_base()
print(f"Trainable (injection only) : {model.num_parameters(only_trainable=True):,}")
model.unfreeze_all()
print(f"Trainable (all unfrozen)   : {model.num_parameters(only_trainable=True):,}")

# ─── Loading from pretrained (commented out — requires download) ───────────────
# Uncomment to load real Qwen3 weights:
#
# model_pretrained = LoopLMForCausalLM.from_pretrained(
#     "Qwen/Qwen3-1.7B-Base",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# print(f"Loaded pretrained LoopLM: {model_pretrained.num_parameters():,} params")
