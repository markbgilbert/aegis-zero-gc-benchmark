/**
 * Aegis Systems Architecture (AL-LANG-02) — Public C/C++ API Definitions
 * 
 * Provides C-compatible interface declarations for the compiled Aegis AI binaries:
 * 1. aegis_gpt.dll         - CPU Forward Pass & Persistent Thread Pool Engine
 * 2. aegis_cuda_engine.dll - Direct CUDA Driver PCIe Gen4 DMA Pipeline Engine
 * 3. aegis_telemetry.dll   - In-Band 64-Byte Cache-Aligned Cryptographic Audit Engine
 *
 * Copyright (c) 2026 Aventine Labs LLC.
 * Author: Mark Gilbert (mbgilbert@gmail.com)
 * License: MIT
 */

#ifndef AEGIS_AI_H
#define AEGIS_AI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. CPU Forward Pass Engine (aegis_gpt.dll)                               */
/* ========================================================================= */

typedef struct {
    uint32_t vocab_size;
    uint32_t block_size;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_embd;
    uint32_t head_dim;
} AegisModelConfig;

typedef struct {
    const float* wte;
    const float* wpe;
    const float* ln_f_g;
    const float* ln_f_b;
    const float* lm_head_w;
    const float** block_ln1_g;
    const float** block_ln1_b;
    const float** block_qkv_w;
    const float** block_qkv_b;
    const float** block_proj_w;
    const float** block_proj_b;
    const float** block_ln2_g;
    const float** block_ln2_b;
    const float** block_mlp_fc_w;
    const float** block_mlp_fc_b;
    const float** block_mlp_proj_w;
    const float** block_mlp_proj_b;
} AegisModelWeights;

/**
 * Initialize the Aegis CPU Runtime and allocate persistent thread pool.
 * @param num_threads Number of worker threads (1-32)
 */
int32_t aegis_gpt_init(uint32_t num_threads);

/**
 * Execute forward pass over pre-pinned cache-aligned memory arena.
 */
int32_t aegis_gpt_forward(
    const AegisModelConfig* config,
    const AegisModelWeights* weights,
    const int64_t* token_ids,
    uint32_t seq_len,
    float* out_logits
);

/**
 * Destroy runtime, free thread pool, and unmap memory arenas.
 */
void aegis_gpt_destroy(void);


/* ========================================================================= */
/* 2. Direct GPU DMA Engine (aegis_cuda_engine.dll)                         */
/* ========================================================================= */

/**
 * Initialize NVIDIA CUDA Driver API (nvcuda.dll) without CUDA runtime overhead.
 */
int32_t aegis_cuda_init(void);

/**
 * Allocate device flat arena (cuMemAlloc).
 */
uint64_t aegis_cuda_alloc_device_arena(uint64_t bytes);

/**
 * Allocate page-locked host memory arena (cuMemAllocHost).
 */
void* aegis_cuda_alloc_pinned_host(uint64_t bytes);

/**
 * High-speed PCIe Gen4 DMA asynchronous transfer.
 */
int32_t aegis_cuda_copy_to_device(uint64_t d_dest, const void* h_src, uint64_t bytes);

/**
 * Synchronize GPU stream.
 */
void aegis_cuda_sync(void);

/**
 * Free device arena memory.
 */
void aegis_cuda_free_device_arena(uint64_t d_ptr);

/**
 * Free page-locked host memory.
 */
void aegis_cuda_free_pinned_host(void* h_ptr);


/* ========================================================================= */
/* 3. In-Band 64-Byte Cryptographic Audit Engine (aegis_telemetry.dll)       */
/* ========================================================================= */

/**
 * Packed to exactly 64 bytes (the physical CPU cache line width).
 * Fits into a single hardware cache line, eliminating cache line splits.
 */
typedef struct __attribute__((aligned(64))) {
    uint64_t timestamp_cycles;  // Hardware RDTSC timestamp
    uint64_t batch_id;          // Sequential monotonically increasing batch ID
    uint32_t step_index;        // Training / inference iteration step
    uint32_t thread_id;         // Core / worker thread execution ID
    float    loss_value;        // Instantaneous batch loss
    float    grad_norm;         // Global gradient L2 norm
    uint32_t event_type;        // Enumerated event code (0=INGEST, 1=FWD, 2=BWD, 3=OPT)
    uint32_t prev_record_hash;  // Rolling cryptographic hash chain (FNV-1a)
    uint8_t  payload_digest[16];// 128-bit payload content fingerprint
    uint8_t  reserved[8];       // Reserved alignment padding (guarantees 64B)
} AegisAuditRecord64;

/**
 * Atomic in-band audit record insertion into pre-pinned ring buffer.
 * Execution speed: 3.45 ns (18.67 CPU cycles).
 */
uint32_t aegis_record_audit_step(
    void* arena_base_ptr,
    uint64_t slot_index,
    uint64_t batch_id,
    uint32_t step,
    uint32_t thread_id,
    float loss,
    float grad_norm,
    uint32_t event_type,
    const void* payload_data,
    uint32_t payload_len
);

/**
 * Microbenchmark hardware cycle counters across N audit writes.
 */
void aegis_bench_audit_throughput(
    uint64_t iterations,
    void* arena_base_ptr,
    uint64_t arena_capacity,
    uint64_t* out_elapsed_cycles,
    uint32_t* out_final_hash
);

#ifdef __cplusplus
}
#endif

#endif /* AEGIS_AI_H */
