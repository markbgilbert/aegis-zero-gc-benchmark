# Test 8R: Native BF16 Unaudited 1.0-Hour Endurance Soak

## 1. Test Overview

Test 8R executed a continuous 3,600-second bare-metal training soak on the physical NVIDIA GeForce RTX 5060 Laptop GPU. The run utilized native C++ / CUDA kernels with deterministic IEEE-754 round-to-nearest-even without in-band telemetry stream auditing, providing the un-instrumented physical baseline for audit tax evaluation.

```text
Arm: Native Bare-Metal Aegis BF16 (Unaudited, Arm B)
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (50W TGP)
Dataset: Shakespeare Character Corpus (B=16, T=1024)
Duration: 3,600.06 seconds (60.0 minutes)
```

---

## 2. Empirical Telemetry Summary

| Metric Dimension | Measured Value | Operational Context |
| :--- | :--- | :--- |
| **Duration Target** | 3,600.00 s | Pre-committed duration bracket |
| **Actual Soak Duration** | 3,600.06 s | 100.0% soak completion |
| **Steps Completed** | 50,753 steps | +6,124 steps (+13.72%) vs PyTorch |
| **Tokens Trained** | 831,537,152 tokens | +100,335,616 tokens vs PyTorch |
| **Average Throughput** | 230,978.6 tokens/sec | Native kernel throughput |
| **Step Wall Latency (p50)** | 70.95 ms | -9.53 ms (-11.84%) vs PyTorch |
| **Step Wall Latency (p99)** | 71.73 ms | Tightly bounded tail latency |
| **GPU Compute Time (p50)** | 70.93 ms | Pure on-device kernel time |
| **Host CPU Extract (p50)** | 21.78 us | 37.5x faster than Python DataLoader |
| **Host CPU Extract (p99)** | 39.98 us | Sub-40 us tail extraction |
| **Step 0 Untrained Loss** | 4.293573 nats | Exact physical baseline |
| **Final Loss (Raw)** | 0.109028 nats | Converged terminal loss |
| **Final Loss (Smoothed)** | 0.106509 nats | Convergence parity with PyTorch |
| **Initial Host RSS Memory** | 819.01 MB VmRSS | Flat arena initialization |
| **Final Host RSS Memory** | 834.09 MB VmRSS | Operating system resident memory |
| **Net VmRSS Memory Drift** | +15.08 MB / hour | Driver page table quantization |
| **Net VmData Heap Drift** | 0.000 MB / hour | Zero heap growth across 831M tokens |
| **Minor Page Faults** | 53 faults | 22,933x reduction vs PyTorch |
| **Page Fault Rate** | 0.0147 faults/sec | Page fault generation halted |
| **GPU Operating Temperature** | 54 C to 59 C (steady 55 C)| Controlled thermal envelope |
| **GPU Power Draw** | 49.85 W | Locked at 50W TGP envelope |

---

## 3. Key Architectural Observations

1. **Host Memory Invariance**: Virtual data segment drift (`VmData`) measured exactly **0.000 MB** from step 0 to step 50,753. Pre-allocating contiguous memory pools upfront completely eliminated heap fragmentation.
2. **Page Fault Suppression**: Total minor page faults were suppressed from **1,215,452** in PyTorch to **53** in native execution (a 22,933x reduction). Fault generation plateaued within the first 10 seconds of process startup.
3. **Execution Efficiency**: Native kernel pipelining yielded a **+13.72% higher token rate** (230,978.6 tok/s vs 203,110.3 tok/s), training over 100 million additional tokens within the identical 1-hour window.
