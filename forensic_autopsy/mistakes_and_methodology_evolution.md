# Forensic Autopsy: Methodological Mistakes, Retractions, and Scientific Evolution

## 1. Epistemic Charter

Scientific progress requires uncompromised transparency. Engineering research frequently buries flawed initial prototypes, failed convergence curves, and telemetry ambiguities, publishing only polished post-hoc successes.

This project rejects that practice. This forensic autopsy documents every significant error, conceptual flaw, and measurement bug encountered during the development of our native C++ / CUDA training harness and audit verification pipeline. We document what went wrong, how the flaw was detected, the exact technical remediation applied, and why preserving these failures makes our physical silicon findings scientifically unassailable.

---

## 2. Mistake 1: Hardcoded Stubs in Early Prototype Code

### The Error
During initial exploratory prototyping, several helper scripts and mock pipelines contained synthetic sleep statements, simulated mathematical decays, and hardcoded placeholders to test user interface layouts and pipeline wiring before native CUDA kernels were fully compiled.

### Detection
The project instituted automated static scanning (`scripts/hallucination_scan.js`) and cross-functional code auditing. Reviewers flagged that placeholder stubs could be mistakenly conflated with physical measurements.

### Technical Remediation
1. The codebase was subjected to a complete architectural purge. All placeholder stubs, mock pipelines, and synthetic delays were systematically removed.
2. The execution pipeline was rebuilt strictly around native C++ / CUDA kernels compiled via `nvcc` with Compute Capability 12.0 (`sm_120`).
3. Hardware execution gates were established: every test script must verify active GPU presence, query device hardware properties, and poll the Linux `/proc` filesystem before recording any metrics.
4. If a physical GPU is not present or if an unhandled kernel error occurs, the test harness halts with an immediate exit code rather than falling back to simulation.

---

## 3. Mistake 2: Conflating Feeder Ingestion Bandwidth with Training Throughput

### The Error
Early data-loading optimizations produced an asynchronous memory-mapped token ring buffer capable of streaming 3,750,000,000 tokens/second out of host pinned RAM into device buffers. Early marketing copy erroneously juxtaposed this 3.75B tok/s figure alongside model training throughput, creating the misleading impression that the model was performing backpropagation at billions of tokens per second.

### Detection
Peer engineering review identified that pure I/O memory streaming does not perform matrix multiplications, attention mechanisms, or gradient computations. Comparing I/O ingestion bandwidth directly to end-to-end training throughput violated standard ML benchmarking conventions.

### Technical Remediation
1. The project instituted the **Two-Table Separation Rule**.
2. Training throughput is strictly defined as the complete forward pass, backward pass, optimizer step, and parameter update. It is measured via synchronized `cudaEventRecord` pairs on physical silicon, yielding the authentic ~231,000 tokens/second throughput.
3. Feeder / I/O bandwidth is quarantined to a dedicated data-pipeline section, explicitly labeled as host-to-device memory ingestion bandwidth with zero neural computation.

---

## 4. Mistake 3: Stochastic Rounding Divergence (The ~1.04 nats Plateau)

### The Error
During the development of custom mixed-precision matrix multiplication routines, an experimental stochastic rounding (bit-dithering) routine was implemented for `fp32` downcasting to `bf16`. When evaluated over 500 steps, the native harness diverged from PyTorch: while PyTorch descended to ~0.74 nats, the native harness hit an artificial loss ceiling at ~1.04 nats.

```text
Step 500 Comparison:
- PyTorch Baseline:              0.748 nats (Converging)
- Native (Stochastic Dithering): 1.043 nats (Plateaued / Divergent)
```

### Detection
Loss curve comparison revealed that gradient variance was artificially high in late-stage training. Mathematical inspection of the AdamW second moment accumulator ($v_t$) proved that random bit perturbations were being rectified ($g_t^2 \ge 0$), inflating the denominator and stalling convergence along narrow loss valleys.

### Technical Remediation
1. The stochastic bit-dithering implementation was completely excised from the kernel codebase.
2. The casting logic was replaced with native hardware-accelerated IEEE-754 Round-to-Nearest-Even (`__float2bfloat16_rn`).
3. Physical re-verification (Test 7R) confirmed that deterministic rounding restored exact convergence parity with PyTorch: 0.741 nats at step 500, 0.384 nats at step 5,000, and 0.1058 nats across a 1-hour soak.
4. The flawed 1.04 nats stochastic curve is intentionally preserved and rendered in the staged Silicon Visualizer as an educational autopsy view.

---

## 5. Mistake 4: Telemetry Step 0 Semantic Ambiguity

### The Error
In the initial execution of Test 9R (2,500-step audited native run), the logged CSV output began at `step=0` with a recorded loss of `2.742700` nats. However, the physical baseline of the untrained model was known to be `4.293573` nats. This caused confusion during external review, as reviewers questioned how an untrained model could register a loss of 2.74 nats on its very first step.

### Detection
Adversarial AI reviews (Claude and ChatGPT) flagged the semantic ambiguity: did the model start at 2.74 nats (suggesting pre-trained initialization or a corrupted baseline), or was there an undocumented warmup period?

### Root Cause
Investigation revealed that the training loop executed a 50-step optimizer warmup routine prior to starting the measurement loop. The measurement loop initialized its internal index at `step=0`, which actually corresponded to Global Training Step 50.

### Technical Remediation
The telemetry schema and audit receipt format were formally updated prior to launching the 1-hour soak (Test 10R):
1. The baseline untrained loss is explicitly captured at Global Step 0 before any optimizer updates: `step_0_untrained_loss: 4.293573`.
2. The 50 warmup steps are explicitly logged: `warmup_steps: 50`.
3. The measurement loop index is explicitly mapped to global step numbers: `measured_step_0` maps to Global Step 50 (`loss: 2.742700`), and `measured_step_50937` maps to Global Step 50987.
4. The running FNV-1a stream digest is seeded with the exact physical baseline loss (`4.293573`), binding the entire global training trajectory into the stream verification chain.

---

## 6. Mistake 5: Regulatory Over-Claiming and Marketing Hyperbole

### The Error
Early drafts of documentation, RFC proposals, and marketing summaries included speculative claims associating the in-band telemetry stream with formal regulatory certifications, specifically:
* Claiming "FIPS 140-3 Level 3 Boundary Enforcement".
* Claiming compliance with "EU AI Act High-Risk Auditability Articles 12 & 15".
* Claiming "SEC Rule 17a-4 and FINRA Regulatory Immutable Storage".

### Detection
Legal and compliance reviews flagged that:
1. FNV-1a is a lightweight, non-cryptographic checksum designed for fast stream integrity and change detection. It is not an approved cryptographic hash under FIPS 140-3.
2. Formal regulatory compliance (EU AI Act, SEC, FINRA) requires certified physical hardware security modules (HSMs), independent laboratory audits, and organizational policies, none of which can be claimed by an open-source training script.
3. Using compliance buzzwords undermined the genuine technical achievement: achieving zero-cost in-band stream tamper detection.

### Technical Remediation
1. All references to FIPS 140-3, EU AI Act compliance, and SEC/FINRA regulatory certifications were completely retracted and purged from technical documents.
2. The project adopted the strict **Two-Layer Terminology Invariant**:
   * **Layer 1**: In-Band Stream Integrity (FNV-1a 32-bit checksum). Strictly designated as non-cryptographic, lightweight change detection.
   * **Layer 2**: Cryptographic Artifact Sealing (SHA-256). Computed out-of-band across completed log files at run termination.
3. All RFCs and research papers were reframed around empirical systems engineering: host virtual memory stability, flat arena allocation, and zero-cost stream observability.

---

## 7. Comparative Summary of Evolution

| Phase | Methodology Flaw | Detection Vector | Applied Resolution | Final Physical Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Placeholder stubs in exploratory scripts | Automated static scanner (`hallucination_scan.js`) | Rebuilt 100% on bare-metal C++ / CUDA kernels | Verified on RTX 5060 |
| **Phase 2** | Feeder memory streaming conflated with training | Peer architecture review | Enforced Two-Table Separation Rule | ~231k tok/s training segregated from 3.75B tok/s I/O |
| **Phase 3** | Stochastic bit-dithering loss plateau (~1.04 nats) | Mathematical AdamW variance analysis | Switched to IEEE-754 Round-to-Nearest-Even | Bit-exact parity down to 0.1058 nats |
| **Phase 4** | Telemetry Step 0 semantic ambiguity | Adversarial AI review (Claude / ChatGPT) | Explicit global step mapping and baseline seeding | Baseline 4.2936 nats bound into audit chain |
| **Phase 5** | Speculative regulatory compliance claims | Legal and compliance audit | Purged all regulatory buzzwords; locked Two-Layer model | Clean engineering scope in PyTorch RFC-0036 |
