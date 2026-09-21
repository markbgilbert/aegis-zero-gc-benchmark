// ==============================================================================
// AEGIS SYSTEMS RUNTIME (AL-LANG-02) - NATIVE TELEMETRY & AUDIT ENGINE
// Pillar Alignment: PILLAR 1 (100% Cryptographic Auditability, Zero Exceptions)
// Tier 2: 64-Byte Cache-Aligned Symbolic Audit Arena
// ==============================================================================

#include <stdint.h>
#include "aegis_audit_arena.h"

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
int DllMain(void* hinst, unsigned long reason, void* reserved) {
    return 1;
}
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Global Standard Telemetry State (Tier 1)
static volatile uint64_t g_telemetry_steps = 0;
static volatile uint64_t g_telemetry_total_tokens = 0;
static volatile float    g_telemetry_last_loss = 0.0f;
static volatile uint64_t g_telemetry_last_cycles = 0;

EXPORT void aegis_record_telemetry_standard(
    uint64_t step,
    float loss,
    float lr,
    uint32_t tokens
) {
    g_telemetry_steps = step;
    g_telemetry_total_tokens += tokens;
    g_telemetry_last_loss = loss;
    g_telemetry_last_cycles = AEGIS_RDTSC();
}

EXPORT uint32_t aegis_record_audit_step(
    AegisAuditRecord* arena,
    uint64_t max_records,
    uint64_t step,
    uint32_t batch_size,
    uint32_t block_size,
    float loss,
    float lr,
    uint32_t memory_kb,
    const uint16_t* sample_tokens,
    uint32_t prev_hash
) {
    aegis_audit_record_step(
        arena, max_records, step, batch_size, block_size,
        loss, lr, memory_kb, sample_tokens, prev_hash
    );
    uint64_t slot = step % max_records;
    return arena[slot].chain_hash;
}

EXPORT void aegis_bench_audit_throughput(
    uint64_t iterations,
    AegisAuditRecord* arena,
    uint64_t max_records,
    uint64_t* out_elapsed_cycles,
    uint32_t* out_final_hash
) {
    uint32_t hash = 0x811C9DC5;
    uint16_t sample_tokens[8] = {123, 456, 789, 1011, 1213, 1415, 1617, 1819};

    uint64_t t0 = AEGIS_RDTSC();
    for (uint64_t i = 0; i < iterations; i++) {
        aegis_audit_record_step(
            arena, max_records, i, 4, 256,
            2.345f, 0.0003f, 863200, sample_tokens, hash
        );
        hash = arena[i % max_records].chain_hash;
    }
    uint64_t t1 = AEGIS_RDTSC();

    *out_elapsed_cycles = (t1 - t0);
    *out_final_hash = hash;
}
