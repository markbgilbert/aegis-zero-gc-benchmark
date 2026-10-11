# Test 7R: Native BF16 Deterministic Convergence Calibration (5,000 Steps)

## 1. Test Overview

Test 7R evaluated the native C++ / CUDA training harness across 5,000 consecutive optimizer steps to verify numerical convergence parity following the transition to hardware IEEE-754 Round-to-Nearest-Even (`__float2bfloat16_rn`).

```text
Arm: Native Bare-Metal Aegis BF16 (Deterministic Rounding)
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (50W TGP)
Dataset: Shakespeare Character Corpus (B=16, T=1024)
Steps: 5,000 steps
Duration: 354.3 seconds
```

---

## 2. Empirical Telemetry Summary

| Metric Dimension | Measured Value | Operational Context |
| :--- | :--- | :--- |
| **Steps Completed** | 5,000 steps | Numerical calibration target |
| **Tokens Trained** | 81,920,000 tokens | Calibration corpus size |
| **Average Throughput** | 230,705.2 tokens/sec | Native kernel throughput |
| **Step Wall Latency (p50)** | 71.01 ms | Consistent pacing |
| **Step Wall Latency (p99)** | 71.85 ms | Controlled tail distribution |
| **Step 0 Untrained Loss** | 4.293573 nats | Exact baseline alignment |
| **Step 500 Loss** | 0.741200 nats | Overcoming stochastic plateau |
| **Step 1,000 Loss** | 0.688500 nats | Smooth convergence curve |
| **Step 2,500 Loss** | 0.661163 nats | Reaching calibration milestone |
| **Step 5,000 Loss** | 0.384200 nats | Terminal calibration loss |
| **Net VmData Heap Drift** | 0.000 MB | Zero heap growth across 5,000 steps |
| **GPU Operating Temperature** | 53 C to 55 C | Stable thermal state |
| **GPU Power Draw** | 49.85 W | Saturated 50W TGP envelope |

---

## 3. Key Architectural Observations

1. **Resolution of Stochastic Plateau**: In earlier builds with stochastic dithering, the loss stalled at ~1.04 nats at step 500. Under deterministic RN-even, step 500 descended to 0.7412 nats, matching PyTorch's convergence trajectory.
2. **Deterministic Parity**: Final loss of 0.3842 nats at step 5,000 matched the expected AdamW optimization trajectory without divergence or numeric instability.
3. **Memory Stability**: VmData remained perfectly constant (0.000 MB drift), demonstrating the efficacy of pre-allocated flat arena memory.
