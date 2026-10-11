# Test 9R: Audited Native BF16 Calibration (Step 0 Corrected, 2,500 Steps)

## 1. Test Overview

Test 9R executed a 2,500-step audited benchmark to validate continuous in-band FNV-1a stream integrity tracking following the explicit correction of the Step 0 semantic baseline. The run resolved earlier telemetry ambiguity by explicitly logging the untrained baseline at Global Step 0, capturing the 50 warmup steps, and seeding the FNV-1a hash chain with the true untrained loss (4.293573 nats).

```text
Arm: Native Bare-Metal Aegis BF16 (Audited, In-Band FNV-1a)
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (50W TGP)
Dataset: Shakespeare Character Corpus (B=16, T=1024)
Steps: 2,500 steps
Duration: 177.15 seconds
```

---

## 2. Empirical Telemetry Summary

| Metric Dimension | Measured Value | Operational Context |
| :--- | :--- | :--- |
| **Steps Completed** | 2,500 steps | Intermediate calibration run |
| **Tokens Trained** | 40,960,000 tokens | Saturated execution |
| **Average Throughput** | 231,219.0 tokens/sec | Native kernel throughput |
| **Step Wall Latency (p50)** | 70.87 ms | Consistent pacing |
| **Step Wall Latency (p99)** | 71.30 ms | Tightly bounded tail |
| **Global Step 0 Baseline** | 4.293573 nats | Physical untrained baseline loss |
| **Warmup Steps** | 50 steps | Optimizer warmup interval |
| **Measured Step 0 Loss** | 2.742700 nats | Global Step 50 loss |
| **Terminal Step Loss** | 0.661163 nats | Measured step 2,499 loss |
| **Terminal Stream Digest** | `0xe3d94ce1` | In-band 32-bit FNV-1a digest |
| **Digest Reproduction** | Verified 100% | Independent validator match |
| **Falsification Bit-Flip** | Detected immediately | Validator failed closed |
| **Net VmData Heap Drift** | 0.000 MB | Zero heap growth across 2,500 steps |
| **GPU Operating Temperature** | 53 C to 55 C | Stable thermal state |
| **GPU Power Draw** | 49.88 W | Saturated 50W TGP envelope |

---

## 3. Key Architectural Observations

1. **Resolution of Step 0 Telemetry Ambiguity**: Global step semantics are made explicit. Global step 0 records the 4.293573 nats untrained state, steps 1 to 50 execute the warmup, and measured step 0 records the 2.742700 nats post-warmup state. The FNV-1a stream chain is seeded with the exact 4.293573 baseline.
2. **Audit Overhead Evaluation**: Throughput was measured at 231,219 tok/s (p50: 70.87 ms) compared to 230,705 tok/s (p50: 71.01 ms) in Test 7R. The audit instrumentation introduced no observable performance slowdown (+0.22% within normal run-to-run noise).
3. **Falsification Verification**: Inverting a single bit in a logged loss float immediately altered the resulting digest, proving that the independent verification script actively detects tampering rather than checking a static assertion.
