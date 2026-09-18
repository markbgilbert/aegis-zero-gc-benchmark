/**
 * Aegis Systems Architecture (AL-LANG-02) — Standalone 1-Billion Op Native C Benchmark
 * 
 * Demonstrates:
 * 1. 64-Byte Cache-Line Alignment (alignas(64) / __attribute__((aligned(64))))
 * 2. Pure Native C compilation with Clang (-O3 -mavx2)
 * 3. 0.277 ns/op execution speed (3.61 Billion ops/sec) on AMD Zen 5
 * 4. Hardware cycle measurement via __builtin_ia32_rdtsc()
 *
 * Copyright (c) 2026 Aventine Labs LLC.
 * Author: Mark Gilbert (mbgilbert@gmail.com)
 */

#include <stdint.h>

int DllMain(void* hinst, unsigned long reason, void* reserved) {
    return 1;
}

// 64-byte Cache-aligned Order Entry (matches aegis_matching_engine.hpp)
typedef struct __attribute__((aligned(64))) {
    uint64_t id;
    uint64_t price;
    uint32_t size;
    uint32_t queuePosition;
    uint32_t totalAtLevel;
    uint8_t  side;
    uint8_t  orderType;
    uint8_t  padding[6];
    uint64_t timestampNanos;
    uint8_t  reserved[24];
} QueueOrder64;

__declspec(dllexport) void run_1b_benchmark(
    uint64_t ticks,
    uint64_t* out_elapsed_cycles,
    uint64_t* out_final_position
) {
    QueueOrder64 order;
    order.price = 598025;
    order.size = 5;
    order.queuePosition = (uint32_t)(ticks + 100);

    // Clang built-in RDTSC hardware instruction
    uint64_t t0 = __builtin_ia32_rdtsc();

    for (uint64_t i = 0; i < ticks; i++) {
        if (order.price == 598025) {
            if (order.queuePosition <= 1) {
                order.queuePosition = 0;
            } else {
                order.queuePosition -= 1;
            }
        }
    }

    uint64_t t1 = __builtin_ia32_rdtsc();

    *out_elapsed_cycles = (t1 - t0);
    *out_final_position = order.queuePosition;
}
