# Test 10R: Native BF16 Audited 1.0-Hour Endurance Soak

## 1. Test Overview

Test 10R executed an uninterrupted 3,600-second bare-metal training soak on the physical NVIDIA GeForce RTX 5060 Laptop GPU with continuous in-band FNV-1a stream integrity tracking enabled (`--audit 1`). This test served as the final audited endurance evaluation, demonstrating long-term memory stability, zero-cost continuous stream auditing, and terminal SHA-256 cryptographic provenance sealing.

```text
Arm: Native Bare-Metal Aegis BF16 (Audited, In-Band FNV-1a)
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (50W TGP)
Dataset: Shakespeare Character Corpus (B=16, T=1024)
Duration: 3,600.04 seconds (60.0 minutes)
```

---

## 2. Empirical Telemetry Summary

| Metric Dimension | Measured Value | Operational Context |
| :--- | :--- | :--- |
| **Duration Target** | 3,600.00 s | Pre-committed duration bracket |
| **Actual Soak Duration** | 3,600.04 s | 100.0% soak completion |
| **Measured Steps** | 50,937 steps | Saturated training run |
| **Measured Tokens Trained** | 834,551,808 tokens | Saturated corpus coverage |
| **Average Token Rate** | 231,817.1 tokens/sec | +0.36% vs Test 8R (Unaudited) |
| **Step Wall Latency (p50)** | 70.62 ms | -0.33 ms lower than Test 8R |
| **Step Wall Latency (p99)** | 71.38 ms | -0.35 ms lower tail than Test 8R |
| **Pipeline Interval (p50)** | 70.60 ms | GPU kernel dispatch pacing |
| **Host CPU Extract (p50)** | 22.04 us | Fast AVX2 batch extraction |
| **Host CPU Extract (p99)** | 41.20 us | Stable extraction tail |
| **Global Step 0 Baseline** | 4.293573 nats | Exact root baseline loss |
| **Post-Warmup Step 0 Loss** | 2.742700 nats | Global Step 50 loss |
| **Final Loss (Raw)** | 0.105857 nats | Continuous convergence |
| **Final Loss (Smoothed)** | 0.106768 nats | Closely matched to Test 8R |
| **Initial Host RSS Memory** | 819.45 MB VmRSS | Flat arena initialization |
| **Final Host RSS Memory** | 834.57 MB VmRSS | Stable operating system footprint |
| **Net VmRSS Memory Drift** | +15.12 MB / hour | Driver page table quantization |
| **Net VmData Heap Drift** | 0.000 MB / hour | Flat heap across 834M tokens |
| **Minor Page Faults** | 48 faults | Fault generation plateaued |
| **Page Fault Rate** | 0.0133 faults/sec | Page fault generation halted |
| **GPU Operating Temperature** | 52 C to 56 C (steady 54 C)| Controlled thermal envelope |
| **GPU Power Draw** | 49.88 W | Saturated 50W TGP envelope |
| **Layer 1 Stream Digest** | FNV-1a: `0xf5b0b764` | Terminal receipt digest |
| **Sampled Stream Digest** | FNV-1a: `0x46e84c88` | Step 50,930 CSV sample digest |
| **Bit-Flip Detection** | Detected immediately | Altered digest to `0x2d176717` |
| **Layer 2 Artifact Seal** | SHA-256 (Pass) | Cryptographic disk verification |

---

## 3. Key Architectural Observations

1. **Zero Measurable Audit Tax**: Across 3,600 seconds of execution, the audited harness (Test 10R) achieved 231,817 tok/s compared to 230,978 tok/s in the unaudited baseline (Test 8R). This delta (+0.36%) falls within normal physical run-to-run variance, confirming that computing an in-band 32-bit FNV-1a rolling digest introduces no observable performance degradation.
2. **Long-Term Memory Flatlining**: Host virtual memory heap (`VmData`) remained at exactly 600.441 MB across all 50,937 steps, yielding $\Delta \text{VmData} = 0.000\text{ MB}$. Process minor faults plateaued at 48 during startup, exhibiting zero recurring page reclaims during 1 hour of sustained execution.
3. **Loss Convergence Parity**: Final smoothed loss converged to 0.106768 nats, reproducing Test 8R's convergence trajectory (0.106509 nats) within $\pm 0.00026$ nats.
4. **Reconciled Stream Telemetry**: The periodic CSV sampling engine (emitting 1 sample every 10 steps in duration mode) recorded 5,095 sampled rows ending at step 50,930 (digest `0x46e84c88`), while the terminal receipt captured step 50,936 (digest `0xf5b0b764`). Both states were reconciled through the sampling window. Single-bit falsification testing verified that mutating any loss value diverts the running digest, causing the independent validator to fail closed.
