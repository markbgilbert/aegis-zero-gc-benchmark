# Aegis Systems Architecture: Linux 60-Minute Soak Test & Compliance Receipt

**Entity:** Aventine Labs LLC  
**Author:** Mark Gilbert ([@markbgilbert](https://github.com/markbgilbert) : mbgilbert@gmail.com), Founder & Principal Systems Architect  
**Architecture Classification:** AL-AI-04 (Direct GPU DMA) / AL-AI-05 (Zero-Cost Symbolic Telemetry)  
**Operating System:** Linux x86_64 (Ubuntu MATE 24.04.3 LTS)  
**Target Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU  
**Model Architecture:** 10.69M Parameter Micro-GPT (L6 H6 D384 B256 V168)  

---

## Executive Summary (Linux Native Verification)

This benchmark provides the raw Linux baseline for **PyTorch RFC-0036**, eliminating Windows WDDM driver abstractions:

1. **Zero Memory Leakage (Linux VmRSS):** Over 60.0 minutes and 15,276 iterations, process resident memory (`VmRSS`) remained completely flat with a net drift of **+4.25 MB**.
2. **Mass Throughput:** Processed **250,281,984 tokens** at an average rate of **69,522 tokens/sec**.
3. **Cryptographic Chain:** All 15,276 in-band FNV-1a telemetry records verified with 100% chain continuity.

Full CSV telemetry: [`./aegis_soak_linux_60min.csv`](./aegis_soak_linux_60min.csv)
