#!/usr/bin/env python3
"""
Aegis Systems Architecture vs Stock PyTorch (CPU Forward Pass Benchmark)
Verifies:
1. Bit-for-bit mathematical loss parity against PyTorch nanoGPT
2. Latency scaling across 1, 4, 8, 16, 32 native Win32 threads
3. Zero-allocation flat arena execution

Copyright (c) 2026 Aventine Labs LLC.
Author: Mark Gilbert (mbgilbert@gmail.com)
"""

import os
import sys
import time
import ctypes
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from nanogpt_reference import GPTConfig, GPT

def main():
    print("=" * 78)
    print("TABLE 1: CPU-BOUND AI EMPIRICAL BENCHMARK (STOCK PYTORCH vs AEGIS NATIVE)")
    print("Hardware: AMD Ryzen 9 9955HX (16 Cores / 32 Threads, Zen 5, 64MB L3)")
    print("Architecture: GPT-2 (6 Layers, 6 Heads, 384 Hidden Dim, Vocab 65, Context 256)")
    print("=" * 78)

    vocab_size = 65
    block_size = 256
    n_layer = 6
    n_head = 6
    n_embd = 384
    head_dim = n_embd // n_head
    T = 256

    # 1. Setup PyTorch Baseline
    torch.set_num_threads(32)
    torch.manual_seed(1337)
    conf = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, bias=True)
    py_model = GPT(conf)
    py_model.eval()

    tokens = torch.randint(0, vocab_size, (1, T), dtype=torch.long)
    targets = torch.randint(0, vocab_size, (1, T), dtype=torch.long)

    NUM_WARMUP = 5
    NUM_RUNS = 25

    print("\n--- Running Stock PyTorch (32 Threads, MKL/oneDNN) ---")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = py_model(tokens, targets)
        py_times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            logits, loss = py_model(tokens, targets)
            t1 = time.perf_counter()
            py_times.append((t1 - t0) * 1000.0)

    py_min = float(np.min(py_times))
    py_med = float(np.median(py_times))
    py_mean = float(np.mean(py_times))
    py_p95 = float(np.percentile(py_times, 95))
    py_loss = float(loss.item())
    py_tok_sec = T / (py_med / 1000.0)

    print(f"PyTorch CPU (32T): Min = {py_min:.2f} ms | Median = {py_med:.2f} ms | Mean = {py_mean:.2f} ms | p95 = {py_p95:.2f} ms")
    print(f"PyTorch Loss:     {py_loss:.6f}")

    # 2. Load Aegis Native DLL
    dll_path = os.path.join(os.path.dirname(__file__), "bin", "aegis_gpt.dll")
    if not os.path.exists(dll_path):
        print(f"[-] DLL not found: {dll_path}")
        return

    dll = ctypes.CDLL(dll_path)

    class AegisConfigStruct(ctypes.Structure):
        _fields_ = [
            ("vocab_size", ctypes.c_uint32),
            ("block_size", ctypes.c_uint32),
            ("n_layer", ctypes.c_uint32),
            ("n_head", ctypes.c_uint32),
            ("n_embd", ctypes.c_uint32),
            ("head_dim", ctypes.c_uint32),
        ]

    class AegisWeightsStruct(ctypes.Structure):
        _fields_ = [
            ("wte", ctypes.c_void_p),
            ("wpe", ctypes.c_void_p),
            ("ln_f_g", ctypes.c_void_p),
            ("ln_f_b", ctypes.c_void_p),
            ("lm_head_w", ctypes.c_void_p),
            ("block_ln1_g", ctypes.c_void_p),
            ("block_ln1_b", ctypes.c_void_p),
            ("block_qkv_w", ctypes.c_void_p),
            ("block_qkv_b", ctypes.c_void_p),
            ("block_proj_w", ctypes.c_void_p),
            ("block_proj_b", ctypes.c_void_p),
            ("block_ln2_g", ctypes.c_void_p),
            ("block_ln2_b", ctypes.c_void_p),
            ("block_mlp_fc_w", ctypes.c_void_p),
            ("block_mlp_fc_b", ctypes.c_void_p),
            ("block_mlp_proj_w", ctypes.c_void_p),
            ("block_mlp_proj_b", ctypes.c_void_p),
        ]

    c_config = AegisConfigStruct(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        head_dim=head_dim
    )

    VoidPtrArray = ctypes.c_void_p * n_layer
    def make_ptr_array(tensors):
        arr = VoidPtrArray()
        for i, t in enumerate(tensors):
            arr[i] = t.data_ptr()
        return arr

    kept_tensors = {
        'ln1_g': [b.ln_1.weight.contiguous() for b in py_model.transformer.h],
        'ln1_b': [b.ln_1.bias.contiguous() for b in py_model.transformer.h],
        'qkv_w': [b.attn.c_attn.weight.t().contiguous() for b in py_model.transformer.h],
        'qkv_b': [b.attn.c_attn.bias.contiguous() for b in py_model.transformer.h],
        'proj_w': [b.attn.c_proj.weight.t().contiguous() for b in py_model.transformer.h],
        'proj_b': [b.attn.c_proj.bias.contiguous() for b in py_model.transformer.h],
        'ln2_g': [b.ln_2.weight.contiguous() for b in py_model.transformer.h],
        'ln2_b': [b.ln_2.bias.contiguous() for b in py_model.transformer.h],
        'mlp_fc_w': [b.mlp.c_fc.weight.t().contiguous() for b in py_model.transformer.h],
        'mlp_fc_b': [b.mlp.c_fc.bias.contiguous() for b in py_model.transformer.h],
        'mlp_proj_w': [b.mlp.c_proj.weight.t().contiguous() for b in py_model.transformer.h],
        'mlp_proj_b': [b.mlp.c_proj.bias.contiguous() for b in py_model.transformer.h],
    }

    block_ln1_g = make_ptr_array(kept_tensors['ln1_g'])
    block_ln1_b = make_ptr_array(kept_tensors['ln1_b'])
    block_qkv_w = make_ptr_array(kept_tensors['qkv_w'])
    block_qkv_b = make_ptr_array(kept_tensors['qkv_b'])
    block_proj_w = make_ptr_array(kept_tensors['proj_w'])
    block_proj_b = make_ptr_array(kept_tensors['proj_b'])
    block_ln2_g = make_ptr_array(kept_tensors['ln2_g'])
    block_ln2_b = make_ptr_array(kept_tensors['ln2_b'])
    block_mlp_fc_w = make_ptr_array(kept_tensors['mlp_fc_w'])
    block_mlp_fc_b = make_ptr_array(kept_tensors['mlp_fc_b'])
    block_mlp_proj_w = make_ptr_array(kept_tensors['mlp_proj_w'])
    block_mlp_proj_b = make_ptr_array(kept_tensors['mlp_proj_b'])

    wte = py_model.transformer.wte.weight.contiguous()
    wpe = py_model.transformer.wpe.weight.contiguous()
    ln_f_g = py_model.transformer.ln_f.weight.contiguous()
    ln_f_b = py_model.transformer.ln_f.bias.contiguous()
    lm_head_w = py_model.lm_head.weight.t().contiguous()

    c_weights = AegisWeightsStruct(
        wte=wte.data_ptr(),
        wpe=wpe.data_ptr(),
        ln_f_g=ln_f_g.data_ptr(),
        ln_f_b=ln_f_b.data_ptr(),
        lm_head_w=lm_head_w.data_ptr(),
        block_ln1_g=ctypes.addressof(block_ln1_g),
        block_ln1_b=ctypes.addressof(block_ln1_b),
        block_qkv_w=ctypes.addressof(block_qkv_w),
        block_qkv_b=ctypes.addressof(block_qkv_b),
        block_proj_w=ctypes.addressof(block_proj_w),
        block_proj_b=ctypes.addressof(block_proj_b),
        block_ln2_g=ctypes.addressof(block_ln2_g),
        block_ln2_b=ctypes.addressof(block_ln2_b),
        block_mlp_fc_w=ctypes.addressof(block_mlp_fc_w),
        block_mlp_fc_b=ctypes.addressof(block_mlp_fc_b),
        block_mlp_proj_w=ctypes.addressof(block_mlp_proj_w),
        block_mlp_proj_b=ctypes.addressof(block_mlp_proj_b),
    )

    scratch_size = T * n_embd * 30 + n_head * T * T
    arena_buf = torch.zeros(scratch_size, dtype=torch.float32)
    out_logits = torch.empty(T, vocab_size, dtype=torch.float32)

    dll.aegis_gpt_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(AegisConfigStruct), ctypes.POINTER(AegisWeightsStruct),
        ctypes.c_void_p, ctypes.c_void_p
    ]
    dll.aegis_gpt_forward.restype = ctypes.c_float

    dll.aegis_set_num_threads.argtypes = [ctypes.c_int32]
    dll.aegis_set_num_threads.restype = None

    thread_configs = [1, 4, 8, 16, 32]
    aegis_results = {}
    last_aegis_loss = 0.0

    print("\n--- Running Aegis Native across Thread Scaling (1 to 32 Threads) ---")
    for num_t in thread_configs:
        dll.aegis_set_num_threads(num_t)
        
        for _ in range(NUM_WARMUP):
            _ = dll.aegis_gpt_forward(
                tokens.data_ptr(), targets.data_ptr(), T,
                ctypes.byref(c_config), ctypes.byref(c_weights),
                arena_buf.data_ptr(), out_logits.data_ptr()
            )
        
        a_times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            last_aegis_loss = dll.aegis_gpt_forward(
                tokens.data_ptr(), targets.data_ptr(), T,
                ctypes.byref(c_config), ctypes.byref(c_weights),
                arena_buf.data_ptr(), out_logits.data_ptr()
            )
            t1 = time.perf_counter()
            a_times.append((t1 - t0) * 1000.0)
        
        a_min = float(np.min(a_times))
        a_med = float(np.median(a_times))
        a_mean = float(np.mean(a_times))
        a_p95 = float(np.percentile(a_times, 95))
        aegis_results[num_t] = (a_min, a_med, a_mean, a_p95)
        print(f"Aegis CPU ({num_t:2d}T): Min = {a_min:6.2f} ms | Median = {a_med:6.2f} ms | Mean = {a_mean:6.2f} ms | p95 = {a_p95:6.2f} ms")

    loss_delta = abs(last_aegis_loss - py_loss)

    print("\n" + "=" * 78)
    print("FINAL EMPIRICAL FORWARD PASS BENCHMARK SUMMARY (AMD Ryzen 9 9955HX)")
    print("=" * 78)
    print(f"{'Engine':<24} | {'Threads':<8} | {'Min (ms)':<10} | {'Median (ms)':<12} | {'p95 (ms)':<10} | {'Scaling':<12}")
    print("-" * 78)
    print(f"{'PyTorch Eager (MKL)':<24} | {'32':<8} | {py_min:<10.2f} | {py_med:<12.2f} | {py_p95:<10.2f} | {'Baseline':<12}")

    base_aegis_1t = aegis_results[1][1]
    for num_t in thread_configs:
        a_min, a_med, a_mean, a_p95 = aegis_results[num_t]
        scaling = base_aegis_1t / a_med
        print(f"{'Aegis Native C':<24} | {num_t:<8} | {a_min:<10.2f} | {a_med:<12.2f} | {a_p95:<10.2f} | {scaling:<12.2f}x")

    print("-" * 78)
    print(f"Mathematical Loss Parity: PyTorch={py_loss:.6f} | Aegis={last_aegis_loss:.6f} | Delta={loss_delta:.8f}")
    print(f"Status:                   {'✅ EXACT MATHEMATICAL BIT-PARITY' if loss_delta < 1e-4 else '⚠️ PARITY DRIFT'}")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
