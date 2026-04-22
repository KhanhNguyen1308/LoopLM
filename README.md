# LoopLM

**Qwen3-1.7B biến thành Recurrent-Depth Transformer** — thêm iterative latent reasoning mà không thay đổi trọng số gốc.

```
Input → [Prelude: 4 layers, 1 lần]
      → [Recurrent: 20 layers, lặp T lần] ← chỉ phần này "suy nghĩ sâu"
      → [Coda: 4 layers, 1 lần]
      → Output logits
```

---

## Ý tưởng cốt lõi

Transformer thông thường chạy mỗi layer đúng **một lần**. LoopLM cho phép một tập layers **chạy lại nhiều lần** trên cùng input — mỗi vòng lặp là một bước "suy nghĩ" thêm trong không gian latent, không tạo ra token trung gian. Đây là looped transformer / recurrent-depth transformer.

Kết quả lý thuyết (Saunshi et al., 2025): T vòng lặp ≈ T bước chain-of-thought ngầm. Mô hình 770M tham số với đủ loops có thể đạt chất lượng của transformer 1.3B thông thường.

---

## Kiến trúc chi tiết

### Phân chia layers

Qwen3-1.7B có 28 layers tổng. LoopLM chia thành 3 phần:

| Phần | Layers | Chạy | Vai trò |
|---|---|---|---|
| **Prelude** | 0–3 (4 layers) | 1 lần | Encode input → vector `e` |
| **Recurrent** | 4–23 (20 layers) | T lần | "Suy nghĩ" lặp lại |
| **Coda** | 24–27 (4 layers) | 1 lần | Decode → logits |

Tỷ lệ: 71% layers (20/28) nằm trong recurrent block.

### Recurrent update rule

Mỗi vòng lặp `t`:

```
h_prev = h
h_out  = TransformerLayers(h, position_embeddings)   # chạy 20 layers
h      = ConnectLayer(h_out, h_prev)                  # blend kết quả
```

`h` sau đó được đưa vào vòng lặp tiếp theo. `e` (output của Prelude) được giữ cố định — input signal không bao giờ mất đi.

### Rotary Embeddings

Qwen3-1.7B dùng **partial RoPE**: chỉ rotate một phần nhỏ của head dimension (`qk_rope_head_dim < head_dim`). `position_embeddings` được tính một lần từ `model.rotary_emb` rồi **dùng chung** cho tất cả layers và tất cả vòng lặp — vì vị trí token không thay đổi qua các loops.

```python
position_embeddings = self.rotary_emb(hidden, position_ids)  # (cos, sin)
# Dùng lại cho prelude, mọi recurrent loop, và coda
```

---

## Connect Layer

Đây là thành phần then chốt quyết định tính ổn định qua nhiều vòng lặp.

### Vấn đề

Nếu inject thẳng output của recurrent block vào input vòng tiếp theo mà không có cơ chế kiểm soát, hidden state `h` sẽ **drift** (phân kỳ) theo số loops tăng dần.

### GatedResidualConnectLayer (default)

Lấy cảm hứng từ [ClawMeBao/looped-lm](https://github.com/ClawMeBao/looped-lm) — kết quả thực nghiệm:

| Connect type | PPL @ n_iter=1 | PPL @ n_iter=4 |
|---|---|---|
| MLP đơn giản | 59 | **9860** ❌ |
| **GatedResidual** | 98 | **~98** ✅ |

Công thức:

```
gate   = sigmoid(W_g · [h_out ; h_prev])   # ∈ (0,1)^D per dim
output = gate * transform(h_out) + (1 - gate) * h_prev
```

Gate học cách blend: nếu vòng này có thông tin tốt → gate cao, dùng nhiều `h_out`. Nếu đã ổn định → gate thấp, giữ nguyên `h_prev`. Tính chất này tự động tạo ra **depth-invariance**.

Tham số: ~6M (0.3% của toàn model).

```python
class GatedResidualConnectLayer(nn.Module):
    def __init__(self, hidden_size, expand=2, dropout=0.05):
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expand),
            nn.SiLU(),
            nn.Linear(hidden_size * expand, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)
        # gate_proj.bias init = 0  →  gate ≈ 0.5 ban đầu
```

### LTIInjection (fallback, `--no_gated`)

Dựa trên lý thuyết Linear Time-Invariant systems:

```
h_{t+1} = A·h_t + B·e
```

Ổn định được đảm bảo **bằng cấu trúc tham số** (không phải bằng regularization):

```
A_discrete = exp(Δt · diag(-exp(log_A)))
```

Vì `-exp(log_A) < 0` và `Δt > 0`, mọi phần tử của `A_discrete` nằm trong `(0, 1)` → spectral radius `ρ(A) < 1` → hệ ổn định. Dùng flag `--no_gated` để so sánh với GatedResidual.

---

## LoRA trên Recurrent Block

Chỉ recurrent block được thêm LoRA adapters. Prelude và Coda được frozen hoàn toàn.

```
q_proj(x)  =  W_base · x   (frozen)
            + lora_B · lora_A · dropout(x) · (alpha/rank)   (trainable)
```

Cấu hình mặc định: rank=16, alpha=32, scale=2.0. Số tham số trainable:
- GatedResidualConnectLayer: ~6M
- LoRA (20 layers × 4 projections × 2 matrices): ~13M
- **Tổng: ~19M / 1700M = 1.1% của model**

---

## Training

### Dataset: Nemotron-Post-Training-Dataset-v2

```python
ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2",
                  split="math",          # hoặc: code, stem, chat, multilingual_*
                  streaming=True)        # không download 98GB
```

Schema mỗi sample:
```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "reasoning": "on/off",
  "category": "math/code/..."
}
```

Loss chỉ được tính trên **assistant tokens** — user/system tokens bị mask thành `-100`.

### Loop Curriculum

Training bắt đầu với ít loops, tăng dần:

```
Bước 0–30% tổng: loops tăng tuyến tính từ 1 → train_loops
Bước 30–100%:    loops cố định ở train_loops
```

Giúp model học cách "suy nghĩ từng bước" trước khi xử lý full depth.

### Mixed Precision

```python
dtype = torch.bfloat16  # MI50/A100/H100 — native support, không cần GradScaler
# hoặc
dtype = torch.float16   # GPU cũ — dùng GradScaler để tránh underflow
```

Gradient checkpointing (`--grad_checkpointing`) giảm VRAM ~40% bằng cách recompute activations thay vì lưu — hữu ích với MI50 16GB.

---

## Cài đặt

```bash
# ROCm (AMD MI50)
pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/rocm6.4

# Hoặc CUDA
pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu126

pip install transformers>=4.51.0 datasets>=2.21.0 \
    accelerate peft trl sentencepiece
```

**Biến môi trường cho AMD MI50:**
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
```

---

## Sử dụng

```bash
# Kiểm tra model load và forward pass
python qwen3_looplm_train.py --sanity

# Train nhỏ để test pipeline
python qwen3_looplm_train.py \
    --splits math \
    --max_samples 500 \
    --train_loops 2 \
    --batch_size 1 \
    --epochs 1

# Train đầy đủ
python qwen3_looplm_train.py \
    --splits math code stem \
    --max_samples 5000 \
    --train_loops 4 \
    --batch_size 2 \
    --epochs 3 \
    --streaming                   # stream thay vì download hết
    --grad_checkpointing          # giảm VRAM

# Multi-GPU
torchrun --nproc_per_node=4 qwen3_looplm_train.py \
    --splits math code stem \
    --max_samples 20000

# Dùng LTI thay vì GatedResidual
python qwen3_looplm_train.py --no_gated --sanity
```

### Tất cả arguments

| Flag | Default | Mô tả |
|---|---|---|
| `--sanity` | False | Chỉ chạy smoke test, không train |
| `--splits` | math code stem | Splits của Nemotron-v2 |
| `--max_samples` | 5000 | Số samples mỗi split |
| `--max_seq_len` | 2048 | Max sequence length |
| `--streaming` | False | Stream dataset (không download hết) |
| `--train_loops` | 4 | Số loops khi train |
| `--max_loops` | 8 | Số loops tối đa khi inference |
| `--prelude_layers` | 4 | Số layers Prelude |
| `--coda_layers` | 4 | Số layers Coda |
| `--batch_size` | 2 | Batch size mỗi GPU |
| `--epochs` | 3 | Số epochs |
| `--lr` | 2e-5 | Learning rate |
| `--lora_rank` | 16 | LoRA rank |
| `--grad_checkpointing` | False | Gradient checkpointing (tiết kiệm VRAM) |
| `--no_gated` | False | Dùng LTI thay vì GatedResidual |
| `--output_dir` | ./qwen3_looplm_checkpoint | Thư mục lưu checkpoint |

---

## Checkpoint

Checkpoint lưu state dict của toàn bộ LoopLM (bao gồm frozen weights để load lại không cần base model):

```python
# Lưu
torch.save(model.state_dict(), "looplm_state_dict.pt")
tokenizer.save_pretrained("checkpoint/")

# Load lại
qwen     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
model    = LoopLM(qwen, cfg)
state    = torch.load("looplm_state_dict.pt", map_location="cpu")
model.load_state_dict(state)

# Inference với nhiều loops hơn lúc train
output = model.generate(input_ids, max_new_tokens=256, n_loops=8)
```

Lưu ý: `n_loops` lúc inference có thể **lớn hơn** lúc train. Nhờ GatedResidual depth-invariance, model không bị drift khi tăng loops.


## Cấu trúc file

```
qwen3_looplm_train.py
│
├── LoopLMConfig          # Tất cả hyperparameters
├── LTIInjection          # Connect layer ổn định theo LTI (fallback)
├── GatedResidualConnectLayer  # Connect layer chính (depth-invariant)
├── LoRALinear            # LoRA wrapper cho nn.Linear
├── apply_lora_to_qwen_layer   # Áp dụng LoRA lên 1 decoder layer
│
├── LoopLM                # Model chính
│   ├── __init__          # Tách layers, khởi tạo injection + LoRA
│   ├── _freeze_base      # Freeze prelude, coda, embed, norm, lm_head
│   ├── _run_layers       # Chạy tuần tự qua list layers (+ grad checkpointing)
│   ├── forward           # Prelude → loop → coda → logits + loss
│   └── generate          # Autoregressive generation với top-p sampling
│
├── load_nemotron_v2      # Load + tokenize dataset (streaming hoặc offline)
├── mask_non_assistant    # Mask non-assistant tokens thành -100
├── NemotronCollator      # Dynamic padding cho DataLoader
│
├── train                 # Training loop chính (DDP support)
├── _evaluate             # Quick validation (50 batches)
├── _save_checkpoint      # Lưu state_dict + tokenizer
│
└── sanity_check          # Smoke test: forward + generate không cần dataset
```

---

## Tham khảo

- Saunshi et al. (2025) — *Reasoning with Latent Thoughts: On the Power of Looped Transformers* — chứng minh T loops ≈ T bước CoT ngầm
- Parcae (Prairie et al., 2026) — *Scaling Laws for Stable Looped Language Models* — LTI stability constraint và scaling laws
- ClawMeBao/looped-lm — GatedResidual connect layer, Phase 0 PPL experiments
- OpenMythos (kyegomez) — Lý thuyết kiến trúc Claude Mythos như Looped Transformer