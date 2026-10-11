# Measurement Methodology and Physical Verification Protocols

## 1. Physical Hardware and Execution Environment

All benchmarks and endurance soaks in this study were executed on bare-metal physical silicon under strictly controlled environmental parameters:

* **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell sm_120 architecture, 8,188 MiB GDDR7, 128-bit bus width).
* **Host Processor**: AMD Ryzen 9 8945HS with Radeon 780M Graphics (8 physical cores, 16 threads, up to 5.2 GHz).
* **Host System Memory**: 32 GB DDR5 RAM.
* **Operating System**: Linux kernel 6.8.0-52-generic (Ubuntu 24.04 LTS x86_64).
* **NVIDIA Driver**: 570.86.16 (Data Center / Production Branch).
* **CUDA Toolkit**: 12.8 (Compute Capability 12.0).
* **PyTorch Baseline**: PyTorch 2.6.0+cu128 with FlashAttention-2 integration.
* **Compiler Harness**: Native C++ / CUDA kernels compiled via `nvcc` with `-O3 --use_fast_math -arch=sm_120`.

```text
Host System: AMD Ryzen 9 8945HS | 32 GB DDR5 | Ubuntu 24.04 LTS
Accelerator: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR7, Blackwell sm_120)
Driver/CUDA: 570.86.16 / CUDA 12.8 / PyTorch 2.6.0+cu128
```

---

## 2. Strict Pre-Run Thermal Gate

To eliminate thermal throttling variance and prevent cold-start or thermal-soak skew between comparative runs, every benchmark script enforces a mandatory pre-run thermal check:

1. The harness queries `nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits`.
2. The GPU die temperature must register strictly **at or below 43 C**.
3. The board power draw must register strictly **at or below 5.00 W** (quiescent idle state).
4. If either condition fails, the execution harness blocks and enters a 5-second polling loop until the thermal dissipation threshold is satisfied.

This protocol guarantees that every run begins from an identical thermal baseline and that power-budget headroom remains consistent across runs.

---

## 3. Explicit Metric Separation: The Two-Table Rule

Physical timing measurements must never blur asynchronous hardware kernel dispatch with synchronous host process lifecycle overhead. We enforce strict separation between two primary timing metrics:

### Table 1: End-to-End Training Pipeline Throughput
* **Definition**: The complete forward pass, backward pass, optimizer step, and parameter update pipeline executed on device.
* **Instrumentation**: Measured using synchronized `cudaEventRecord` pairs (`start_event`, `stop_event`) placed immediately before input tensor ingestion and immediately following the optimizer parameter step.
* **Synchronization**: `cudaEventSynchronize(stop_event)` is executed before computing elapsed time to ensure that GPU execution completes prior to timestamp reading.
* **Token Rate Formula**:
  $$\text{Tokens/Sec} = \frac{\text{Batch Size} \times \text{Sequence Length}}{\text{Pipeline Latency (seconds)}}$$
  For our standardized configuration ($B=16, T=1024$):
  $$\text{Tokens/Sec} = \frac{16{,}384}{\text{Pipeline Latency (s)}}$$

### Table 2: Step Wall Clock Interval and Process Lifecycle
* **Definition**: The total elapsed time per step as measured by the host CPU, including host loop overhead, CUDA kernel launch queuing, file system telemetry serialization, and process status checks.
* **Instrumentation**: High-resolution host clock timestamps (`std::chrono::high_resolution_clock` or Python `time.perf_counter()`).
* **Distinction**: Pipeline latency quantifies raw device execution throughput. Step wall clock captures real-world host consumption. Both metrics are reported side-by-side in every receipt.

### Isolation of Feeder and I/O Bandwidth
Raw token streaming bandwidth from pinned host memory or disk (e.g. 3.75 billion tokens/sec through an asynchronous ring buffer) must **never** be conflated with model training throughput. Feeder performance is audited as an ingestion sub-metric, completely segregated from end-to-end backpropagation token throughput.

---

## 4. Host Memory Accounting and Virtual Memory Churn

To verify host memory stability and measure allocator fragmentation over multi-hour runs, the harness captures operating system memory counters directly from the Linux `/proc` pseudo-filesystem on every measurement interval:

### Virtual Memory Counters (`/proc/self/status`)
* `VmRSS`: Resident Set Size (physical memory occupied by the process).
* `VmData`: Virtual memory allocated to the data segment (heap).
* `VmPeak`: Peak virtual memory footprint across the process lifetime.
* **VmData Drift Metric**: Calculated as:
  $$\Delta \text{VmData} = \text{VmData}_{\text{terminal}} - \text{VmData}_{\text{step 0}}$$
  A stable memory manager exhibits $\Delta \text{VmData} = 0.000\text{ MB}$, proving zero memory leakage or heap growth.

### Operating System Page Fault Counters (`/proc/self/stat`)
* `minflt` (Field 10): Minor page faults (page reclaims that did not require physical disk I/O).
* `majflt` (Field 12): Major page faults (page faults requiring disk read).
* **Fault Accounting**: PyTorch's dynamic caching allocator repeatedly requests, frees, and maps virtual memory blocks via `mmap`/`brk`, generating over **1,200,000 minor page faults** during a 1-hour training run. In contrast, a flat arena allocator pre-allocates contiguous device and host buffers upfront, yielding **48 minor faults** over an identical 1-hour run, after which fault generation ceases entirely.

---

## 5. Hardware PMU and Thermal Telemetry

Hardware operating conditions are continuously monitored via asynchronous background polling:

* A dedicated thread queries `nvidia-smi` at 1,000 ms intervals.
* Logged parameters include instantaneous core temperature (C), total board power draw (W), and allocated GPU VRAM (MB).
* These readings verify that:
  1. The GPU sustained its target power cap (50 W sustained during computation).
  2. Operating temperatures remained within the stable operating window (52 C to 56 C across 1-hour continuous runs).
  3. No throttling events or thermal power drops occurred during endurance testing.

---

## 6. Two-Layer Stream and Artifact Evidence Architecture

To maintain strict auditability without sacrificing execution speed, we establish a two-layer verification architecture. The separation between stream integrity checking and cryptographic artifact sealing is absolute:

```text
+--------------------------------------------------------------------------+
| Layer 1: In-Band Stream Integrity (Microsecond Latency, Non-Cryptographic)|
| - Algorithm: 32-bit FNV-1a hash over (step, loss, lr, norm) tuples       |
| - Purpose: Tamper-evident change detection and continuous stream state   |
| - Performance Overhead: 0.00% measurable throughput impact               |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| Layer 2: Out-of-Band Cryptographic Sealing (Post-Run Provenance Seal)   |
| - Algorithm: SHA-256 digest over complete telemetry CSV and JSON receipt |
| - Purpose: Immutable forensic validation and tamper detection            |
| - Execution: Computed once at run completion                             |
+--------------------------------------------------------------------------+
```

### Layer 1: In-Band Stream Integrity (FNV-1a)
* **Designation**: Stream Integrity Digest (Non-Cryptographic).
* **Implementation**: A 32-bit Fowler-Noll-Vo (FNV-1a) algorithm running in-band within the step iteration.
* **Hash Computation**:
  Each measured step computes:
  $$\text{hash}_{k} = \text{FNV1a}(\text{hash}_{k-1}, \langle \text{step}, \text{loss}, \text{learning\_rate}, \text{grad\_norm} \rangle)$$
* **Seeding**: The hash chain is seeded with the exact physical baseline loss recorded at Global Step 0 (4.293573 nats).
* **Validation**: An independent post-run validator replays the logged CSV data, recalculates the running FNV-1a digest, and verifies exact match with the terminal hash stored in the run receipt. Any altered float bit immediately diverges the running digest.

### Layer 2: Out-of-Band Cryptographic Sealing (SHA-256)
* **Designation**: Cryptographic Artifact Seal.
* **Implementation**: Standard NIST FIPS 180-4 SHA-256 cryptographic hash.
* **Execution**: Computed across the raw on-disk telemetry log file (`.csv`) and the run receipt (`.json`) immediately upon process completion.
* **Purpose**: Guarantees artifact immutability for archival storage and peer reproduction.

---

## 7. Pre-Committed Pass / Fail Verification Gates

Before launching comparative endurance runs (Test 8R vs Test 10R), acceptance criteria were formally pre-committed to prevent post-hoc rationalization:

1. **Audit Overhead Ceiling**: Training throughput degradation in the audited build must not exceed **1.00%** relative to the non-audited baseline.
2. **Loss Parity Tolerance**: Smoothed terminal loss must agree within **$\pm 0.005$ nats** across identical step counts.
3. **Memory Stability Gate**: Virtual data segment drift ($\Delta \text{VmData}$) must measure exactly **$0.000\text{ MB}$** over 3,600 seconds of continuous execution.
4. **Digest Verification Gate**: The independent stream validator must achieve **100% bit-exact terminal hash match** against the execution receipt, and must fail closed upon single-bit tamper injection.
