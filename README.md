# Privan-130M: A ~130M Parameter Decoder-Only Foundation LLM from Scratch in PyTorch

Privan-130M is a complete, modular, from-scratch implementation of a ~130M parameter causal decoder-only transformer language model built entirely in pure PyTorch. 

This is not a wrapper, nor does it rely on remote APIs or pre-packaged GPT models. Every single component—from subword tokenization, positional embeddings, causal multi-head self-attention with KV-caching, Pre-LayerNorm transformer blocks, and decoupled AdamW optimization to distributed training (DDP), LoRA fine-tuning, and FastAPI inference—is implemented cleanly from foundational mathematical principles.

---

## Architecture Specification

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Name** | Privan-130M | Foundation Decoder-Only Transformer |
| **Vocabulary Size ($V$)** | 50,257 | Byte-Level BPE subword vocabulary |
| **Context Length ($T$)** | 1,024 tokens | Maximum sequence length |
| **Embedding Dimension ($D$)** | 768 | Hidden representation dimension |
| **Transformer Layers ($L$)** | 12 | Number of stacked decoder blocks |
| **Attention Heads ($H$)** | 12 | Number of parallel attention heads |
| **Head Dimension ($d_{\text{head}}$)** | 64 | $768 / 12 = 64$ |
| **MLP Inner Dimension** | 3,072 | $4 \times 768$ expansion dimension |
| **Activation** | GELU | Gaussian Error Linear Unit (approximate='tanh') |
| **Normalization** | Pre-LayerNorm | Normalized residual stream ($\epsilon = 10^{-5}$) |
| **Weight Tying** | Enabled | `lm_head.weight` tied to `token_embedding.weight` |
| **Total Trainable Parameters** | **124,439,808 (~124.4M)** | Tied weights (163,037,184 un-tied) |

---

## Directory Structure

```
privan-130m/
├── README.md                 # Technical documentation & theory
├── LICENSE                   # Apache 2.0 license
├── requirements.txt          # Minimal production dependencies
├── pyproject.toml            # Package configuration
├── MODEL_CARD.md             # Model card & ethical guidelines
├── configs/
│   ├── 130m.yaml             # Primary ~130M model training configuration
│   ├── tiny.yaml             # 2-layer miniature model for rapid testing
│   └── debug.yaml            # Debugging configuration for instant verification
├── data/
│   ├── raw/                  # Raw text, jsonl, or parquet datasets
│   ├── processed/            # Binary memory-mapped tokenized chunks (.bin)
│   ├── tokenizer/            # Trained Byte-Level BPE tokenizer files
│   └── README.md             # Ethical data guidelines & storage documentation
├── checkpoints/              # Model weights (latest.pt, best.pt, step_*.pt)
├── logs/                     # Text logs and TensorBoard experiment event files
├── src/
│   └── llm/
│       ├── config.py         # Type-safe dataclass configurations
│       ├── tokenizer.py      # Byte-Level BPE Tokenizer wrapper
│       ├── dataset.py        # Cleaning, chunking, and memory-mapped Dataset
│       ├── model/
│       │   ├── embeddings.py       # Token & Learned Positional Embeddings
│       │   ├── attention.py        # Causal Multi-Head Self-Attention with KV Cache
│       │   ├── mlp.py              # Feed-Forward Network with GELU
│       │   ├── transformer_block.py# Pre-LN Transformer Decoder Block
│       │   ├── transformer.py      # Transformer Backbone
│       │   ├── lm.py               # CausalLM with LM Head & Weight Tying
│       │   └── lora.py             # Low-Rank Adaptation (LoRA) module
│       ├── training/
│       │   ├── optimizer.py        # AdamW with Decoupled Weight Decay
│       │   ├── scheduler.py        # Linear Warmup + Cosine Annealing
│       │   ├── checkpoint.py       # Full state checkpointing & NaN recovery
│       │   ├── distributed.py      # Multi-GPU DDP orchestrator
│       │   └── trainer.py          # Complete training & validation engine
│       ├── generation/
│       │   ├── sampler.py          # Greedy, Temp, Top-K, Top-P, Repetition penalty
│       │   └── generate.py         # Autoregressive generation & streaming
│       ├── evaluation/
│       │   └── evaluate.py         # Cross-entropy loss & perplexity evaluation
│       └── utils/
│           ├── device.py           # Hardware inspection & precision selector
│           ├── seed.py             # Global determinism & reproducibility
│           └── logging.py          # Structured logging setup
├── scripts/
│   ├── train_tokenizer.py    # Train BPE tokenizer on raw text
│   ├── prepare_data.py       # Clean, tokenize, and chunk into .bin files
│   ├── count_parameters.py   # Programmatic parameter counter
│   ├── pretrain.py           # Pretraining engine (single-GPU & DDP)
│   ├── finetune.py           # Instruction fine-tuning (with LoRA & response masking)
│   ├── evaluate.py           # Checkpoint evaluation script
│   ├── generate.py           # CLI generation script with streaming
│   ├── export_model.py       # Export to PyTorch .pt, Safetensors, HuggingFace format
│   ├── data_report.py        # Data quality & statistics report
│   └── benchmark.py          # Latency, throughput, and KV-cache benchmark
├── inference/
│   ├── app.py                # Production FastAPI REST server
│   └── cli.py                # Interactive terminal shell
├── tests/
│   ├── test_tokenizer.py     # Tokenizer unit tests
│   ├── test_attention.py     # Causal masking & KV-cache parity tests
│   ├── test_model.py         # Shapes, weight tying, and initialization tests
│   ├── test_dataset.py       # Data packing & memmap tests
│   ├── test_generation.py    # Sampling algorithms & streaming tests
│   ├── test_training.py      # AdamW weight decay, scheduler, and checkpoints
│   └── test_overfit.py       # Single-batch overfitting verification test
└── notebooks/
    ├── tokenizer.ipynb       # Tokenizer exploration notebook
    ├── architecture.ipynb    # Architecture inspection notebook
    └── training.ipynb        # Training curves and optimization notebook
```

---

## Theoretical Foundations & Engineering Principles

### 1. What is an LLM?
A Large Language Model is a deep parametric function $f_\theta$ trained over extensive text corpora to approximate the joint probability distribution of natural language tokens:
$$P(x_1, x_2, \dots, x_N) = \prod_{t=1}^N P(x_t \mid x_1, \dots, x_{t-1})$$

### 2. What is a Transformer?
Introduced by Vaswani et al. (2017), the Transformer eliminates recurrent and convolutional structures in favor of parallel attention mechanisms. In a decoder-only transformer, representations are built by iteratively refining hidden states through stacked self-attention and non-linear feed-forward transformations.

### 3. What is Causal Language Modeling?
Causal Language Modeling (CLM) trains the network autoregressively: given the history of previous tokens $x_{<t} = (x_1, \dots, x_{t-1})$, the model predicts the probability distribution over the vocabulary for token $x_t$. The training loss is the negative log-likelihood (Cross-Entropy):
$$\mathcal{L}(\theta) = - \frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$
The perplexity (PPL) is the exponentiated loss:
$$\text{PPL} = \exp(\mathcal{L})$$
representing the effective branching factor of the model's uncertainty.

### 4. What is Self-Attention & Why Multiple Heads?
Given an input sequence $X \in \mathbb{R}^{B \times T \times D}$, query ($Q$), key ($K$), and value ($V$) representations are projected linearly:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
Scaled dot-product attention computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}}\right) V$$
Dividing by $\sqrt{d_{\text{head}}}$ prevents dot products from growing excessively large in high dimensions, which would drive the softmax into saturated regions with vanishingly small gradients.

**Multi-Head Attention** splits the embedding dimension $D$ across $H$ heads ($d_{\text{head}} = D / H$). This allows the model to jointly attend to information from different representation subspaces at different positions simultaneously (e.g. syntax, semantic co-occurrence, long-range entity coreference).

### 5. What is Positional Encoding?
Transformers are permutation-invariant by design; without positional information, shuffling tokens in the input yields identical attention patterns. Privan-130M uses **learned positional embeddings**: a parameter matrix $W_{\text{pos}} \in \mathbb{R}^{\text{context\_length} \times D}$ that is indexed by discrete position indices $t \in [0, \text{context\_length}-1]$ and directly added element-wise to the token embeddings.

### 6. Why Pre-LayerNorm & Residual Connections?
- **Residual Connections** ($x \leftarrow x + f(x)$) create identity gradient highways, allowing error signals to backpropagate across dozens of layers without vanishing.
- **Pre-LayerNorm** applies normalization $\text{LN}(x)$ to the input of each sub-layer *before* the multi-head attention and feed-forward blocks:
  $$x^{(l)}_{1} = x^{(l-1)} + \text{Attention}(\text{LN}_1(x^{(l-1)}))$$
  $$x^{(l)} = x^{(l)}_1 + \text{MLP}(\text{LN}_2(x^{(l)}_1))$$
  Unlike Post-LayerNorm (used in early Transformers), Pre-LayerNorm keeps the residual stream un-normalized and mathematically clean, preventing gradient explosion at initialization and making training robust without delicate learning rate tuning.

### 7. Why GELU?
The Gaussian Error Linear Unit (GELU) weights inputs by their value rather than gating strictly by their sign like ReLU:
$$\text{GELU}(x) = x \Phi(x) = x P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
Privan-130M uses the fast tanh approximation:
$$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$
GELU provides smooth non-zero curvature across the entire real number line, preventing "dead neurons" and accelerating convergence in language modeling.

### 8. Why AdamW & Decoupled Weight Decay?
Standard L2 regularization in Adam adds the weight decay term directly to the gradient, which inadvertently scales regularization by the moving average of squared gradients ($v_t$). AdamW (Loshchilov & Hutter, 2017) decouples weight decay by subtracting it directly from the parameter after the adaptive step:
$$\theta_{t+1} = \theta_t - \gamma_t \lambda \theta_t - \gamma_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
In Privan-130M, weight decay ($0.1$) is strictly applied to 2D weight matrices (linear layers and embeddings), while 1D tensors (biases and LayerNorm scale/shift parameters) have weight decay set to $0.0$.

### 9. Why Warmup and Cosine Decay?
- **Linear Warmup**: At step 0, adaptive optimizer states ($\hat{m}_t, \hat{v}_t$) have high variance. Linearly ramping the learning rate from 0 to $3 \times 10^{-4}$ over the first 2,000 steps prevents destabilizing early updates.
- **Cosine Annealing**: Decaying the learning rate smoothly toward $\eta_{\min} = 3 \times 10^{-5}$ according to:
  $$\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\pi \frac{t - t_{\text{warmup}}}{T_{\max} - t_{\text{warmup}}}\right)\right)$$
  allows the optimizer to escape sharp local minima early on and settle into flat, generalizable basins toward the end of training.

### 10. Why Gradient Accumulation?
Training foundation models with large effective batch sizes (e.g. 128 sequences = 131,072 tokens per update) often exceeds available GPU memory. Gradient accumulation splits the effective batch into micro-batches (e.g. batch size 8), accumulates gradients over 16 micro-steps, and performs a single optimizer step, producing mathematically equivalent gradient updates while fitting comfortably within modest VRAM.

### 11. Why KV Cache?
During autoregressive generation, tokens are predicted one by one. In naive generation, passing the growing sequence of length $t$ repeatedly recomputes Key and Value projections for all previous $t-1$ tokens, leading to quadratic $\mathcal{O}(T^2)$ time complexity.

With **KV Caching**, previous keys and values for each layer are stored in memory. At step $t$, only the new token $x_t$ is projected into $Q_t, K_t, V_t$. $K_t$ and $V_t$ are appended to the cache, reducing per-token generation complexity from $\mathcal{O}(T)$ to $\mathcal{O}(1)$ attention computation.

---

## Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/your-username/privan-130m.git
cd privan-130m
pip install -r requirements.txt
```

### 2. Parameter Count Verification
Run the programmatic parameter counting script:
```bash
python scripts/count_parameters.py --config configs/130m.yaml
```

### 3. Run Complete Test Suite
Verify model architecture, causal masking, KV-cache, and single-batch memorization:
```bash
pytest -v
```

### 4. Train Tokenizer
Train the Byte-Level BPE tokenizer on raw text:
```bash
python scripts/train_tokenizer.py --input data/raw --output data/tokenizer --vocab-size 50257
```

### 5. Prepare & Tokenize Dataset
Clean, split (98% train, 1% val, 1% test), and compile memory-mapped binary chunks:
```bash
python scripts/prepare_data.py --input data/raw --tokenizer-dir data/tokenizer --output-dir data/processed
```
Inspect dataset quality and token counts:
```bash
python scripts/data_report.py
```

### 6. Pretraining
Train locally on GPU or CPU:
```bash
python scripts/pretrain.py --config configs/130m.yaml
```
Resume from a previous checkpoint:
```bash
python scripts/pretrain.py --config configs/130m.yaml --resume checkpoints/latest.pt
```

### 7. Multi-GPU Distributed Pretraining (DDP)
Launch multi-GPU distributed data parallel training with `torchrun`:
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py --config configs/130m.yaml
```

### 8. Evaluation
Compute validation loss and perplexity:
```bash
python scripts/evaluate.py --checkpoint checkpoints/latest.pt --config configs/130m.yaml --data data/processed/val.bin
```

### 9. Autoregressive Text Generation
Generate text using Top-P nucleus sampling and KV-caching:
```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Artificial intelligence is" --max-new-tokens 150 --stream
```

### 10. Instruction Fine-Tuning (with LoRA)
Fine-tune on instruction-response pairs with response-only loss masking:
```bash
python scripts/finetune.py --config configs/130m.yaml --checkpoint checkpoints/latest.pt --data data/raw/instructions.jsonl --use-lora --lora-rank 8
```

### 11. Model Export
Export checkpoints to `.pt`, `safetensors`, and Hugging Face format:
```bash
python scripts/export_model.py --checkpoint checkpoints/latest.pt --config configs/130m.yaml --output-dir export
```

### 12. Inference REST API Server
Start the high-performance FastAPI inference server:
```bash
python inference/app.py
```
Or interact via terminal CLI:
```bash
python inference/cli.py --checkpoint checkpoints/latest.pt
```

---

## Benchmarking & Performance Profiling

Run the comprehensive performance benchmark suite:
```bash
python scripts/benchmark.py --config configs/debug.yaml
```
Measures:
- Forward pass latency (ms)
- Backward pass latency (ms)
- Training throughput (tokens/sec)
- Autoregressive generation throughput with vs. without KV cache (tokens/sec)
- Peak memory consumption

---

## License
Privan-130M is released under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.
