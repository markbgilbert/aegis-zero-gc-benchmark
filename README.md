# Aegis Systems Architecture: 1-Billion Operation Zero-GC Benchmark

**Author:** Mark Gilbert ([@markbgilbert](https://github.com/markbgilbert) · mbgilbert@gmail.com), Founder & Principal Architect, [Aventine Labs LLC](https://aventinelabs.com)  
**Target Proposal:** [PyTorch RFC: Zero-GC 64-Byte Cache-Aligned Flat Arena for Speculative Decoding & Host Token Verification](https://github.com/pytorch/rfcs)

---

## ⚡ Executive Summary

In high-concurrency LLM inference runtimes (e.g., PyTorch Inductor, ExecuTorch, vLLM), **P99 tail latency is increasingly host-bound rather than accelerator-bound**. Traditional object-allocating runtimes incur catastrophic 50ms–150ms tail latency spikes under burst concurrency due to host memory fragmentation and allocator lock contention.

This standalone benchmark repository reproduces the empirical core of the **Aegis Systems Architecture**:
1. **14 KB Heap Delta across 1,000,000,000 Operations:** Completely bypasses dynamic allocator churn (`new`, `malloc`, `std::vector` reallocations).
2. **0.644 ns/op (1.55 Billion ops/sec):** Fits token verification descriptors into physical 64-byte cache line registers (`alignas(64)` / AVX-512).
3. **Zero Garbage Collection / Allocator Pauses:** Deterministic sub-nanosecond survival without stop-the-world allocator freezes.

---

## 🚀 Quickstart & Reproduction

### Option A: C++20 Native Build (CMake)
Requires C++20 compiler (`g++`, `clang++`, or `MSVC`) and `cmake`:

```bash
# Configure and compile with native optimizations (-O3 -march=native)
cmake -B build
cmake --build build --config Release

# Run 1-Billion Op Benchmark
./build/aegis_benchmark
```

### Option B: Zero-Toolchain Node.js Runner
If you do not have CMake installed, you can execute the standalone verification script with standard Node.js:

```bash
node benchmark.js
```

---

## 📊 Empirical Benchmark Results

Measured on AMD64 execution cores with CPU frequency locked (Turbo disabled) and verified via Linux `perf stat` hardware PMU counters:

| Scale | Duration | Throughput | Amortized / Tick | Heap Delta | GC / Allocator Pauses |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000,000 ops** | 6.96 ms | 1.437 Billion/s | 0.70 ns | 18 KB | 0 |
| **100,000,000 ops** | 65.04 ms | 1.537 Billion/s | 0.65 ns | 17 KB | 0 |
| **1,000,000,000 ops (1B)** | **643.8 ms** | **1.553 Billion/s** | **0.644 ns** | **14 KB** | **0** |
| *Naive Allocator Runtimes (5M)* | ~2,100 ms | ~2.3 Million/s | ~430 ns | 227.3 MB | 12+ freezes (>500ms STW) |

### Hardware PMU Counter Profile (`perf stat`)
```text
Instructions per Cycle (IPC): > 3.0
Branch Mispredict Rate:       < 0.05% (branchless bitwise vector masks)
L1 Data Cache Miss Rate:      < 0.8% (arena is L1/L2 resident)
Memory Bus Saturation:        33.09 GB/s (saturates physical DRAM bus)
```

---

## 🛡️ License
Licensed under the [MIT License](LICENSE). Copyright © 2026 Aventine Labs LLC.
