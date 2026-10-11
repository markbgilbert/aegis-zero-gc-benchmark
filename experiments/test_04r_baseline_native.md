# Test 4R: Stock PyTorch BF16 1.0-Hour Endurance Baseline

## 1. Test Overview

Test 4R provides the reference PyTorch 2.6.0 baseline against which native execution and memory dynamics are evaluated. The run executed a 3,600-second sustained training soak on bare-metal hardware using standard PyTorch mixed-precision (`torch.amp.autocast`) with FlashAttention-2.

```text
Arm: Stock PyTorch BF16 (Arm A)
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (50W TGP)
Dataset: Shakespeare Character Corpus (B=16, T=1024)
Duration: 3,600.02 seconds (60.0 minutes)
```

---

## 2. Empirical Telemetry Summary

| Metric Dimension | Measured Value | Operational Context |
| :--- | :--- | :--- |
| **Duration Target** | 3,600.00 s | Pre-committed duration bracket |
| **Actual Soak Duration** | 3,600.02 s | 100.0% soak completion |
| **Steps Completed** | 44,629 steps | Dynamic allocator paced execution |
| **Tokens Trained** | 731,201,536 tokens | Saturated training run |
| **Average Throughput** | 203,110.3 tokens/sec | Reference PyTorch BF16 baseline |
| **Step Wall Latency (p50)** | 80.48 ms | Host plus device loop latency |
| **Step Wall Latency (p99)** | 81.14 ms | Tail latency under allocator churn |
| **Host CPU DataLoader (p50)** | 817.60 us | Python DataLoader extraction |
| **Host CPU DataLoader (p99)** | 1,211.30 us | Python tail extraction overhead |
| **Step 0 Untrained Loss** | 4.293591 nats | Untrained baseline reference |
| **Final Loss (Raw)** | 0.110200 nats | Final step loss |
| **Final Loss (Smoothed)** | 0.109900 nats | Exponential moving average |
| **Initial Host RSS Memory** | 1,581.15 MB VmRSS | Dynamic allocator startup footprint |
| **Final Host RSS Memory** | 1,672.59 MB VmRSS | Operating system resident memory |
| **Net VmRSS Memory Drift** | +91.44 MB / hour | Dynamic allocator fragmentation |
| **Net VmData Heap Drift** | +1.78 MB / hour | Process heap growth |
| **Minor Page Faults** | 1,215,452 faults | Dynamic memory mapping churn |
| **Page Fault Rate** | 337.62 faults/sec | Active MMU intervention |
| **GPU Operating Temperature** | 52 C to 54 C | Stable thermal state |
| **GPU Power Draw** | 49.88 W | Saturated 50W TGP envelope |

---

## 3. Key Architectural Observations

1. **Memory Allocator Churn**: PyTorch's dynamic caching allocator relies on recurrent virtual memory mapping and splitting. Over 1 hour, this mechanism generated **1,215,452 minor page faults** (337.6 faults/sec) and +91.44 MB of resident memory drift.
2. **DataLoader Bottleneck**: Python DataLoader batch extraction consumed 817.6 us (p50) and 1,211.3 us (p99), introducing intermittent micro-stalls into the kernel launch queue.
3. **Loss Reference**: Convergence to 0.1099 nats establishes the empirical loss baseline for 124M parameter models over 3,600 seconds on this dataset.
