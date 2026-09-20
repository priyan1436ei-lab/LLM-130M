# 🧠 Privan-130M

### From-Scratch Decoder-Only Language Model in PyTorch

**Privan-130M** is a from-scratch implementation of a GPT-style, decoder-only Transformer language model built with PyTorch.

The project focuses on understanding and implementing the complete LLM stack — from **tokenization and dataset preparation to Transformer training, KV-cache inference, LoRA fine-tuning, distributed training, evaluation, and API serving**.

> **Current model size:** ~124.4M trainable parameters
> **Model class:** ~130M-parameter decoder-only Transformer

---

## 🎬 Visual Architecture

```text
                         ┌────────────────────┐
                         │      INPUT TEXT    │
                         │  "The future of AI"│
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │      TOKENIZER     │
                         │   BPE + ByteLevel  │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │     TOKEN IDs      │
                         │ [1842, 492, 312...]│
                         └──────────┬─────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │     TOKEN + POSITION        │
                    │        EMBEDDINGS            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │          TRANSFORMER BLOCK × 12         │
              │                                        │
              │  ┌──────────────┐   ┌──────────────┐  │
              │  │ LayerNorm    │   │ LayerNorm    │  │
              │  └──────┬───────┘   └──────┬───────┘  │
              │         ▼                  ▼           │
              │  ┌──────────────┐   ┌──────────────┐  │
              │  │ Causal Self  │   │     MLP      │  │
              │  │  Attention   │   │ 768 → 3072   │  │
              │  │    Q K V     │   │ 3072 → 768   │  │
              │  └──────┬───────┘   └──────┬───────┘  │
              │         └────────┬─────────┘           │
              │                  ▼                     │
              │          Residual Connections          │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │  Final LayerNorm│
                         └────────┬───────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │     LM HEAD    │
                         │ 768 → 50,257   │
                         └────────┬───────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │     LOGITS     │
                         └────────┬───────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      SAMPLING           │
                    │ Temperature / Top-K     │
                    │ Top-P / Repetition      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │ GENERATED TEXT │
                         └────────────────┘
```

---

## ⚡ KV Cache Visualization

Privan-130M supports KV caching for autoregressive generation.

Instead of recomputing previous keys and values at every generation step:

```text
Token 1
   │
   ▼
 Q ───┐
 K ───┼──► KV CACHE
 V ───┘


Token 2
   │
   ▼
New Q
New K ─────► Append to Cache
New V ─────► Append to Cache
   │
   ▼
Attention over cached K/V
   │
   ▼
Next Token
```

This allows previously computed attention states to be reused during generation.

---

# 🏗️ Model Architecture

Privan-130M uses a decoder-only Transformer architecture.

| Component                  |         Configuration |
| -------------------------- | --------------------: |
| Approx. Model Class        |                  130M |
| Exact Trainable Parameters |           124,439,808 |
| Vocabulary Size            |                50,257 |
| Context Length             |                 1,024 |
| Transformer Layers         |                    12 |
| Attention Heads            |                    12 |
| Head Dimension             |                    64 |
| Embedding Dimension        |                   768 |
| MLP Dimension              |                 3,072 |
| Activation                 |                  GELU |
| Normalization              |         Pre-LayerNorm |
| Position Encoding          |               Learned |
| Attention                  | Causal Self-Attention |
| LM Head                    |           Weight Tied |
| KV Cache                   |                     ✅ |
| LoRA                       |                     ✅ |
| DDP                        |                     ✅ |
| Mixed Precision            |                     ✅ |

---

# 🔢 Parameter Count

The model contains:

```text
124,439,808
```

unique trainable parameters with token embedding and LM-head weight tying.

Therefore the project is more precisely described as:

> **A ~124.4M-parameter decoder-only Transformer in the ~130M model class.**

The `130M` name is used as a model-class approximation rather than an exact parameter count.

---

# 🧩 Core Components

## 1. Token Embeddings

Each token ID is converted into a dense vector:

```text
Token ID
   ↓
Embedding Table
   ↓
768-dimensional vector
```

The model uses a vocabulary of **50,257 tokens**.

---

## 2. Positional Embeddings

Privan-130M uses learned positional embeddings.

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer Input
```

The maximum supported context length is **1,024 tokens**.

---

## 3. Multi-Head Causal Self-Attention

Each Transformer block contains 12 attention heads.

```text
Input
  │
  ▼
Fused QKV Projection
  │
  ├──── Q
  ├──── K
  └──── V
       │
       ▼
Multi-Head Attention
       │
       ▼
Output Projection
```

Causal masking ensures that a token cannot attend to future tokens.

```text
Token 1 → Token 1
Token 2 → Token 1, Token 2
Token 3 → Token 1, Token 2, Token 3
Token 4 → Token 1, Token 2, Token 3, Token 4
```

---

# 🧠 Transformer Block

Each block follows a Pre-LayerNorm architecture:

```text
                 Input
                   │
                   ▼
              LayerNorm
                   │
                   ▼
          Causal Self-Attention
                   │
                   ▼
             Residual Add
                   │
                   ▼
              LayerNorm
                   │
                   ▼
                 MLP
                   │
                   ▼
             Residual Add
                   │
                   ▼
                Output
```

There are **12 Transformer blocks**.

---

# 🔥 MLP

The feed-forward network expands the hidden dimension:

```text
768
 │
 ▼
3072
 │
 ▼
GELU
 │
 ▼
768
```

The model uses approximate GELU activation.

---

# 🔗 Weight Tying

The token embedding matrix and language-model output matrix share the same weights.

```text
             ┌─────────────────┐
             │ Token Embedding  │
             └────────┬────────┘
                      │
                      │ Shared Weights
                      │
             ┌────────▼────────┐
             │    LM Head      │
             └─────────────────┘
```

This reduces the number of unique parameters and follows a common language-model design.

---

# 📚 Training Pipeline

```text
Raw Documents
      │
      ▼
Text Cleaning
      │
      ▼
Document Deduplication
      │
      ▼
BPE Tokenization
      │
      ▼
Binary Token Storage
      │
      ▼
Context Window Creation
      │
      ▼
Micro-Batching
      │
      ▼
Gradient Accumulation
      │
      ▼
Mixed Precision
      │
      ▼
Gradient Clipping
      │
      ▼
AdamW Optimizer
      │
      ▼
Warmup + Cosine LR
      │
      ▼
Checkpointing
      │
      ▼
Validation / Perplexity
```

---

# ⚙️ Training Configuration

Default production configuration:

```yaml
vocab_size: 50257
context_length: 1024
embedding_dim: 768
num_layers: 12
num_heads: 12
mlp_dim: 3072

learning_rate: 3e-4
min_learning_rate: 3e-5

beta1: 0.9
beta2: 0.95
weight_decay: 0.1

warmup_steps: 2000
max_steps: 100000

batch_size: 8
gradient_accumulation_steps: 16

precision: bf16
```

At:

```text
8 × 16 × 1024
```

the effective token batch is approximately:

```text
131,072 tokens / optimizer step / GPU
```

before accounting for distributed world size.

---

# 🧮 Optimizer

The project uses **AdamW** with decoupled weight decay.

Parameters are separated into:

```text
Weight Decay
├── Matrix parameters
└── Higher-dimensional weights

No Weight Decay
├── Bias
├── LayerNorm
└── 1D parameters
```

---

# 📈 Learning Rate Schedule

Training uses:

```text
        Warmup
          /
         /
        /──────────────
       /                \
      /                  \
     /                    \
    /                      \
   /                        \
  /                          \
Start                         Min LR
```

The scheduler combines:

* Linear warmup
* Cosine decay
* Configurable minimum learning rate

---

# 🧪 Mixed Precision

The training system supports:

```text
FP32
BF16
FP16
```

Mixed precision reduces GPU memory consumption and can improve training throughput on compatible hardware.

---

# 🚀 Distributed Training

Privan-130M includes PyTorch Distributed Data Parallel support.

```text
                 Training Dataset
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           GPU 0     GPU 1     GPU 2
             │         │         │
             └─────────┼─────────┘
                       ▼
                Gradient Sync
                       │
                       ▼
                Global Update
```

---

# 🎯 LoRA Fine-Tuning

Low-Rank Adaptation is supported for parameter-efficient fine-tuning.

Conceptually:

```text
Original Linear Layer
        +
      LoRA
   ┌───────────┐
   │ A × B     │
   │ rank = r  │
   └───────────┘
        │
        ▼
Adapted Output
```

The base model weights can remain frozen while only the LoRA parameters are trained.

Default LoRA targets include:

```text
qkv_proj
out_proj
```

---

# 💾 Checkpointing

Training checkpoints can contain:

```text
Model State
Optimizer State
Scheduler State
Training Step
Configuration
Scaler State
```

This allows training to resume without restarting the complete run.

---

# 📊 Evaluation

The training framework supports:

* Validation loss
* Perplexity
* Training loss tracking
* TensorBoard logging
* Checkpoint evaluation

Example:

```text
Training Loss
      │
      │\
      │ \
      │  \
      │   \
      │    \____
      │
      └──────────────► Steps
```

---

# ✍️ Text Generation

Generation supports:

### Temperature

Controls sampling randomness.

```text
Low Temperature
      ↓
More deterministic

High Temperature
      ↓
More diverse
```

### Top-K

Restricts sampling to the K highest-probability tokens.

### Top-P

Samples from the smallest probability mass whose cumulative probability reaches the selected threshold.

### Repetition Penalty

Reduces repetitive generation.

---

# 🔄 Generation Pipeline

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
Transformer
  │
  ▼
KV Cache
  │
  ▼
Logits
  │
  ▼
Sampling
  │
  ▼
Next Token
  │
  └──────────────┐
                 │
                 ▼
             Repeat
                 │
                 ▼
          Generated Text
```

---

# 🌐 FastAPI Inference

The project provides an API layer for model inference.

Example architecture:

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Tokenizer
  │
  ▼
Privan-130M
  │
  ▼
Generation Engine
  │
  ▼
Response
```

Example endpoint structure:

```http
POST /generate
```

Request:

```json
{
  "prompt": "Artificial intelligence is",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95
}
```

---

# 📁 Project Structure

```text
LLM-130M/
│
├── src/
│   └── llm/
│       ├── model/
│       │   ├── embeddings.py
│       │   ├── attention.py
│       │   ├── mlp.py
│       │   ├── transformer_block.py
│       │   ├── transformer.py
│       │   ├── lm.py
│       │   └── lora.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── optimizer.py
│       │   ├── scheduler.py
│       │   └── checkpoint.py
│       │
│       ├── generation/
│       │   ├── generate.py
│       │   └── sampler.py
│       │
│       ├── dataset.py
│       └── tokenizer.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── pretrain.py
│   └── ...
│
├── tests/
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_generation.py
│   └── test_overfit.py
│
├── configs/
│
├── data/
│
├── checkpoints/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 🧪 Testing

The project includes tests for:

```text
✓ Attention shapes
✓ Causal masking
✓ SDPA / manual attention parity
✓ KV-cache attention
✓ Transformer blocks
✓ Weight tying
✓ Parameter count
✓ Optimizer grouping
✓ Learning-rate scheduler
✓ Checkpoint save/load
✓ Generation
✓ Sampling
✓ Repetition penalty
✓ Tiny-model overfitting
```

Run:

```bash
pytest -q
```

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/priyan1436ei-lab/LLM-130M.git
cd LLM-130M
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Data Preparation

Prepare the training data:

```bash
python scripts/prepare_data.py
```

The pipeline performs:

```text
Input Documents
      ↓
Cleaning
      ↓
Normalization
      ↓
Deduplication
      ↓
Train / Validation / Test Split
      ↓
Tokenization
      ↓
Binary Token Dataset
```

---

# 🏋️ Training

Start pretraining using the configured training script:

```bash
python scripts/pretrain.py
```

For distributed training:

```bash
torchrun --nproc_per_node=4 scripts/pretrain.py
```

Adjust the number of processes according to the available GPUs.

---

# 💬 Inference

A typical generation workflow:

```python
from llm.model.lm import CausalLM

model = CausalLM(...)
model.eval()

output = model.generate(
    prompt="The future of artificial intelligence",
    max_new_tokens=100
)

print(output)
```

---

# 📈 Performance

Benchmark results should always be reported together with the hardware and software configuration.

Recommended benchmark metadata:

```text
GPU:
CUDA:
PyTorch:
Precision:
Batch Size:
Context Length:
Prompt Length:
Generated Tokens:
Warmup Runs:
Measured Runs:
Average Latency:
Tokens / Second:
```

This makes performance comparisons reproducible.

---

# 🔬 Engineering Design Goals

Privan-130M was designed around several principles:

### 1. From Scratch

Core Transformer components are implemented directly in PyTorch rather than relying on a high-level pretrained language-model implementation.

### 2. Modularity

Model, training, dataset, generation, and serving components are separated.

### 3. Reproducibility

Configuration-driven training, checkpointing, evaluation, and tests are included.

### 4. Efficient Inference

KV caching and optimized attention paths are supported.

### 5. Parameter-Efficient Fine-Tuning

LoRA support enables adaptation without updating every base-model parameter.

### 6. Scalable Training

The training stack supports gradient accumulation, mixed precision, and distributed training.

---

# ⚠️ Current Project Status

> **Important:** The repository currently represents a substantial from-scratch LLM implementation and training framework. The checked-in demonstration dataset is not sufficient evidence of large-scale pretraining.

Current status:

```text
Transformer Architecture     ✅
Causal Attention             ✅
KV Cache                     ✅
Weight Tying                 ✅
LoRA                         ✅
AdamW                        ✅
LR Scheduler                 ✅
Mixed Precision              ✅
DDP Support                  ✅
Checkpointing                ✅
FastAPI Serving              ✅
Generation                   ✅
Testing                      ✅
Large-Scale Pretraining      🚧
Production Corpus            🚧
Full Pretrained Checkpoint   🚧
```

The model should therefore currently be described as:

> **A from-scratch ~124.4M-parameter decoder-only Transformer implementation and training framework.**

A claim that it is a fully pretrained foundation model should only be made after documenting the actual large-scale training run.

---

# 🚧 Roadmap

## Phase 1 — Core Architecture

* [x] Decoder-only Transformer
* [x] Multi-head causal attention
* [x] Pre-LayerNorm
* [x] GELU MLP
* [x] Learned positional embeddings
* [x] Weight tying

## Phase 2 — Training Infrastructure

* [x] AdamW
* [x] Gradient accumulation
* [x] Gradient clipping
* [x] Mixed precision
* [x] Checkpointing
* [x] TensorBoard
* [x] DDP

## Phase 3 — Efficient Inference

* [x] KV cache
* [x] Top-K sampling
* [x] Top-P sampling
* [x] Temperature
* [x] Repetition penalty
* [x] Streaming generation

## Phase 4 — Fine-Tuning

* [x] LoRA
* [ ] Instruction tuning
* [ ] Chat fine-tuning
* [ ] Preference optimization

## Phase 5 — Large-Scale Pretraining

* [ ] Production-scale corpus
* [ ] Large-scale tokenization
* [ ] Data quality filtering
* [ ] Training run
* [ ] Loss curves
* [ ] Perplexity evaluation
* [ ] Standard benchmarks
* [ ] Reproducible compute report

## Phase 6 — Release

* [ ] Hugging Face model release
* [ ] Model card
* [ ] Dataset documentation
* [ ] License documentation
* [ ] Quantized model
* [ ] Production inference deployment

---

# 🧭 Research & Learning Focus

This project is intended to provide practical understanding of:

```text
Tokenization
     ↓
Language Modeling
     ↓
Transformer Architecture
     ↓
Self-Attention
     ↓
Causal Masking
     ↓
Optimization
     ↓
Distributed Training
     ↓
Efficient Inference
     ↓
Fine-Tuning
     ↓
Model Serving
```

It is both a learning project and an engineering foundation for experimenting with small-to-medium language models.

---

# 👨‍💻 Author

**Priyan**

B.Tech Information Technology
Prathyusha Engineering College

Areas of interest:

* Artificial Intelligence
* Machine Learning
* Large Language Models
* Generative AI
* Full-Stack Development
* AI Engineering
* Research & Experimentation

---

# ⭐ Repository

**GitHub**

`https://github.com/priyan1436ei-lab/LLM-130M`

If you find the project useful, consider giving the repository a ⭐.

---

# 📜 License

See the repository license for the current licensing terms.

---

## ⚡ Privan-130M

> **Build the Transformer. Understand the Model. Train the Intelligence.**
