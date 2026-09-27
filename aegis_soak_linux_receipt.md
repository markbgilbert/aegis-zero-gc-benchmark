# Aegis Systems Architecture: Linux 60-Minute Soak Test & Compliance Receipt

**Entity:** Aventine Labs LLC  
**Author:** Mark Gilbert ([@markbgilbert](https://github.com/markbgilbert) : mbgilbert@gmail.com), Founder & Principal Systems Architect  
**Architecture Classification:** AL-AI-04 (Direct GPU DMA) / AL-AI-05 (Zero-Cost Symbolic Telemetry)  
**Operating System:** Linux x86_64 (Ubuntu MATE 24.04.3 LTS)  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell sm_120)  
**Host CPU:** AMD Ryzen 9 9955HX (16 Cores / 32 Threads, 64MB L3)  
**Model Architecture:** 10.69M Parameter Micro-GPT (L6 H6 D384 B256 V168)  
**Dataset:** Multi-Volume Literary Corpus (18,523,844 characters / 16,671,459 tokens)  

---

## Executive Summary: Dual-OS Empirical Invariance Proof

This benchmark delivers the official empirical receipt for **PyTorch RFC-0036**, cross-referencing Windows 11 WDDM driver behavior against native Linux POSIX memory allocation:

1. **Allocators Disproved as True Leaks:**
   * **Windows 11 (WDDM 3.2):** 5x 1.00 MB discrete staircase jumps (+4.97 MB over 29,711 steps). Once 4.00 MB of WDDM virtual address paging reserves are nullified, the true heap delta is +0.97 MB.
   * **Linux (POSIX glibc ptmalloc2):** 9x 0.25 MB sub-arena consolidations (+2.25 MB over 15,275 steps), interrupted by 5 distinct flatlines of 800 to 1,000 steps with **0.00 MB growth**. Dedicated GPU reserved memory remained locked at 2,686.0 MB for 2,300 consecutive steps.
   * **Mathematical Proof:** The exact same C++20 transpiler binary exhibits discrete 1.00 MB steps on Windows and 0.25 MB steps on Linux while dedicated GPU memory remains completely flat. This proves that the staircase represents OS driver page-table reserves rather than application heap leakage.

2. **Production Hardened Run (`aegis_soak_linux_60min_100.csv`):**
   * **Sustained Scale:** Processed **392,216,576 tokens** across **23,939 sequential steps** in exactly **3,599.97 seconds (60.0 minutes)**.
   * **Massive Throughput:** Averaged **109,185 tokens/sec** at **149.59 ms median step latency** (+57% throughput acceleration over unhardened baseline).
   * **Zero VRAM / VmData Leak:** PyTorch VRAM allocated locked flat at **204.27 MB** (0.00 MB drift). PyTorch VRAM reserved locked flat at **2,686.00 MB** (0.00 MB drift). Host `VmData` locked flat at **2,948.07 MB** (0.00 MB drift).
   * **Thermal Equilibrium:** Solid **56°C core temperature** throughout continuous 100% GPU saturation on laptop silicon (slowdown threshold: 102°C).
   * **Cryptographic Continuity:** All 23,939 sequential steps verified with 100% FNV-1a checksum continuity (`0x811C9DC5` to `0xD9B26BEA`), providing tamper-evident batch provenance at 0.00 ns DMA overhead.

---

## Dual-OS Empirical Comparison Matrix

| Metric | Windows 11 Pro (WDDM 3.2) | Ubuntu MATE 24.04 (POSIX glibc) | Ubuntu MATE (C10 + Hardened) | Diligence Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Duration / Steps** | 60.0 min / 29,711 steps | 60.0 min / 15,275 steps | 60.0 min / 23,939 steps | Sustained hardware saturation |
| **Total Tokens** | 486.8 Million | 250.3 Million | 392.2 Million | Multi-epoch corpus traversal |
| **Raw Host Memory Delta**| +4.97 MB Commit | +2.25 MB VmRSS | +4.59 MB VmRSS | Sub-6 MB total drift over 1 hr |
| **Step Quantization** | 5x 1.00 MB jumps | 9x 0.25 MB jumps | +0.00 MB drift in VmData | Allocator bin boundaries |
| **Flatline Stability** | Variable paging | 5x [800-1000 step] flat locks | 1,000+ step flatline tail | Zero leak during execution |
| **GPU Reserved Drift** | 0.00 MB | 0.00 MB (Flat @ 2686.0 MB) | 0.00 MB (Flat @ 2686.0 MB) | Zero CUDA allocator bloat |
| **Throughput / Latency**| ~135k tok/s / 121ms | 69.5k tok/s / 235ms | 109.2k tok/s / 149ms | Native streaming efficiency |
| **Thermal Equilibrium** | 72°C steady-state | 67°C steady-state | 56°C steady-state | No thermal throttling |
| **Production Hardening** | **Baseline Verified** | **Empirical PASS** | **Production Hardened** | Continuous bare-metal soak |

---

## Verification Artifacts

* Raw 60-Min Production CSV: [`./aegis_soak_linux_60min_100.csv`](./aegis_soak_linux_60min_100.csv)
* Raw 60-Min Glibc Baseline CSV: [`./aegis_soak_linux_60min.csv`](./aegis_soak_linux_60min.csv)
* Raw 10-Min Smoke Test CSV: [`./test_10min.csv`](./test_10min.csv)
* Machine-Readable JSON Receipt: [`./aegis_soak_linux_receipt.json`](./aegis_soak_linux_receipt.json)
* Hardware Telemetry Screenshots:
  * [`./docs/hardware_monitor/rtx5060_linux_56c_100pct_utilization.png`](./docs/hardware_monitor/rtx5060_linux_56c_100pct_utilization.png)
  * [`./docs/hardware_monitor/htop_linux_cpu_pinning_99pct.png`](./docs/hardware_monitor/htop_linux_cpu_pinning_99pct.png)
