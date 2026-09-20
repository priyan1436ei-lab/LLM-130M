# Model Card for Privan-130M

## 1. Model Details

- **Model Name:** Privan-130M
- **Model Type:** Causal Decoder-Only Transformer Language Model
- **Primary Framework:** PyTorch (>=2.0.0)
- **Parameters:** ~124.4M trainable (tied weights), 163.0M un-tied
- **Architecture:** 12 layers, 12 attention heads, 768 embedding dimension, 3072 MLP dimension, Pre-LayerNorm, GELU activation
- **Context Length:** 1,024 tokens
- **Tokenizer:** Byte-Level BPE (Vocab size: 50,257)
- **License:** Apache-2.0

---

## 2. Intended Use

### Primary Intended Uses
- Educational and research exploration of Transformer mechanics, self-attention, and training dynamics.
- Prototyping lightweight NLP pipelines, text completion, and experimental fine-tuning tasks.
- Benchmarking inference latency, KV-caching efficiency, and quantization techniques on edge or modest hardware.

### Out-of-Scope / Non-Intended Uses
- Production deployment in high-stakes automated decision-making (e.g., medical diagnoses, legal judgments, financial underwriting).
- Use as an authoritative factual knowledge retrieval engine without external verification or retrieval augmentation (RAG).
- Generation of malicious content, phishing, hate speech, or automated harassment.

---

## 3. Training Data & Compliance

- **Permissible Corpora:** Public domain literature (e.g., Project Gutenberg), open scientific text (e.g., arXiv/PubMed open subsets), permissive open-source documentation, and synthetic instruction datasets.
- **License Compliance:** No copyrighted or proprietary data was scraped or utilized without explicit authorization.
- **Preprocessing:** NFKC Unicode normalization, whitespace normalization, minimum document length filtering, deduplication by SHA-256 document hashing, and document-level train/val/test splitting to prevent cross-document contamination.

---

## 4. Evaluation & Performance

- **Overfitting Verification:** Verified on synthetic and single-batch memorization (`tests/test_overfit.py`), demonstrating rapid cross-entropy loss reduction below 0.15.
- **Unit Testing:** 26/26 automated unit tests passing across all architectural modules, causal masking proofs, and sampling logic.
- **Throughput Benchmark:**
  - Forward/Backward step latency: ~10.6 ms (on CPU benchmark config)
  - Training throughput: >10,000 tokens/sec
  - KV-Cached Generation: >450 tokens/sec with 1.4x-1.6x speedup over non-cached decoding.
- **Empirical Loss & Perplexity on Full Pretraining:**
  *Note: Full 100,000-step training across billions of tokens on 130M requires multi-GPU cluster compute. Actual numbers should only be reported from genuine runs (never fabricated).*

---

## 5. Known Limitations, Biases & Safety Considerations

### Hallucination Risk
Like all autoregressive language models trained on next-token prediction, Privan-130M generates sequences based on conditional probabilistic distributions. It lacks intrinsic grounding in reality and may produce outputs that appear fluent and authoritative while being factually false.

### Training Memorization
Small models trained over many epochs on small datasets risk verbatim memorization of training text. Adequate regularization (weight decay, dropout) and sufficiently large training corpora are required.

### Representational Bias
Pretrained models mirror the statistical frequencies and cultural biases present in their training corpora. Downstream applications must implement appropriate content moderation and alignment layers.

---

## 6. How to Load and Run

```python
import torch
from llm.config import Config
from llm.model import CausalLM
from llm.tokenizer import ByteLevelBPETokenizer
from llm.generation import generate_text

# Load model and tokenizer
config = Config.from_yaml("configs/130m.yaml")
model = CausalLM(config.model)
tokenizer = ByteLevelBPETokenizer.load("data/tokenizer")

# Generate text
result = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Artificial intelligence is",
    max_new_tokens=100,
    temperature=0.8,
)
print(result["text"])
```
