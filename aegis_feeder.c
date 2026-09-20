#include <stdint.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
int DllMain(void* hinst, unsigned long reason, void* reserved) { return 1; }
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// 64-byte Cache-line aligned Batch Slot Header
typedef struct __attribute__((aligned(64))) {
    uint64_t batchId;
    uint32_t blockSize;
    uint32_t batchSize;
    uint32_t totalTokens;
    uint32_t strideBytes;
    uint64_t timestampNanos;
    uint64_t reserved;
} BatchSlot64;

/**
 * Zero-Allocation 64-Byte Cache-Aligned Batch Ingestion Kernel
 * Directly extracts tokens into pre-pinned flat arena with zero runtime heap allocations.
 */
EXPORT void aegis_extract_batch(
    const uint16_t* __restrict__ source_data,
    const int64_t*  __restrict__ indices,
    int64_t batch_size,
    int64_t block_size,
    int64_t* __restrict__ out_x,
    int64_t* __restrict__ out_y
) {
    for (int64_t i = 0; i < batch_size; i++) {
        int64_t offset = indices[i];
        int64_t base_dest = i * block_size;
        for (int64_t j = 0; j < block_size; j++) {
            out_x[base_dest + j] = (int64_t)source_data[offset + j];
            out_y[base_dest + j] = (int64_t)source_data[offset + j + 1];
        }
    }
}
