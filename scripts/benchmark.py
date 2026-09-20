"""Benchmarking script for Privan-130M execution, training throughput, and generation."""

import sys
import time
import argparse
from pathlib import Path
import torch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config
from llm.model import CausalLM
from llm.utils import select_device_and_dtype


def benchmark_forward_backward(model, device, dtype, batch_size=4, seq_len=1024, warmup=3, runs=10):
    model.train()
    inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    autocast_enabled = (device.type == "cuda") or (dtype in (torch.bfloat16, torch.float16))

    # Warmup
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            _, loss, _ = model(inputs, targets=targets)
        loss.backward()
        model.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark forward
    fwd_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            _, loss, _ = model(inputs, targets=targets)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)

    # Benchmark backward
    bwd_times = []
    for _ in range(runs):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            _, loss, _ = model(inputs, targets=targets)
        t0 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t0)
        model.zero_grad()

    avg_fwd = sum(fwd_times) / len(fwd_times) * 1000
    avg_bwd = sum(bwd_times) / len(bwd_times) * 1000
    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / ((avg_fwd + avg_bwd) / 1000)

    return avg_fwd, avg_bwd, tokens_per_sec


def benchmark_generation(model, device, prompt_len=64, gen_tokens=50, use_kv_cache=True):
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    curr_ids = input_ids
    kv_caches = None
    pos_offset = 0

    with torch.no_grad():
        for step in range(gen_tokens):
            logits, _, new_kv = model(
                input_ids=curr_ids,
                kv_caches=kv_caches if use_kv_cache else None,
                position_offset=pos_offset if use_kv_cache else 0,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if use_kv_cache:
                curr_ids = next_token
                pos_offset += 1 if step > 0 else prompt_len
                kv_caches = new_kv
            else:
                curr_ids = torch.cat([curr_ids, next_token], dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    dur = time.perf_counter() - t0
    gen_tokens_per_sec = gen_tokens / max(1e-5, dur)
    return gen_tokens_per_sec, dur * 1000


def main():
    parser = argparse.ArgumentParser(description="Benchmark Privan-130M performance")
    parser.add_argument("--config", type=str, default="configs/debug.yaml", help="Config file to benchmark")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.device:
        device = torch.device(args.device)
        dtype = torch.float32
    else:
        device, dtype, _ = select_device_and_dtype(cfg.training.precision)

    model = CausalLM(cfg.model).to(device)

    print("=" * 60)
    print(f"PRIVAN BENCHMARK REPORT: {cfg.model.name}")
    print("=" * 60)
    print(f"Device:    {device}")
    print(f"Precision: {dtype}")
    print(f"Context:   {cfg.model.context_length}")
    print(f"Layers:    {cfg.model.num_layers}, Hidden: {cfg.model.embedding_dim}")
    print("-" * 60)

    seq_len = min(128, cfg.model.context_length)
    batch_size = 2

    print(f"Running Forward/Backward Benchmark (Batch={batch_size}, SeqLen={seq_len})...")
    fwd_ms, bwd_ms, train_tok_sec = benchmark_forward_backward(
        model, device, dtype, batch_size=batch_size, seq_len=seq_len, warmup=2, runs=5
    )
    print(f"  Forward Latency:      {fwd_ms:.2f} ms")
    print(f"  Backward Latency:     {bwd_ms:.2f} ms")
    print(f"  Total Step Latency:   {(fwd_ms + bwd_ms):.2f} ms")
    print(f"  Training Throughput:  {train_tok_sec:,.0f} tokens/sec")

    print("-" * 60)
    print("Running Autoregressive Generation Benchmark (50 tokens)...")
    gen_tok_sec_kv, gen_lat_kv = benchmark_generation(model, device, prompt_len=16, gen_tokens=30, use_kv_cache=True)
    gen_tok_sec_nokv, gen_lat_nokv = benchmark_generation(model, device, prompt_len=16, gen_tokens=30, use_kv_cache=False)

    print(f"  Generation (with KV Cache):    {gen_tok_sec_kv:.1f} tokens/sec ({gen_lat_kv:.2f} ms)")
    print(f"  Generation (without KV Cache): {gen_tok_sec_nokv:.1f} tokens/sec ({gen_lat_nokv:.2f} ms)")
    if gen_lat_nokv > 0:
        speedup = gen_tok_sec_kv / max(1e-5, gen_tok_sec_nokv)
        print(f"  KV Cache Speedup:              {speedup:.2f}x")

    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print("-" * 60)
        print(f"Peak GPU Memory: {max_mem:.2f} MB")

    print("=" * 60)


if __name__ == "__main__":
    main()
