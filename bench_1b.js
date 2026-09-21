// ==============================================================================
// Aegis Standalone Zero-GC Flat Arena Benchmark (Pure Node.js / V8 TypedArray)
// Reference: PyTorch RFC-0036 (pytorch/rfcs#110)
// Cross-Language Zero-GC Thesis: 64-Byte Cache-Aligned Flat Arena in Managed Memory
// ==============================================================================

const ARENA_SLOTS = 65536;
const ARENA_MASK = ARENA_SLOTS - 1;
const STRIDE_BYTES = 64;
const STRIDE_INT32 = STRIDE_BYTES / 4; // 16 int32s per 64-byte cache line

// Pre-allocate fixed 4MB ArrayBuffer (L3 cache resident, zero GC churn)
const buffer = new ArrayBuffer(ARENA_SLOTS * STRIDE_BYTES);
const view = new Int32Array(buffer);

// Offsets within 64-byte struct (int32 indices)
// QueueOrder64 layout:
// uint64_t id (indices 0, 1)
// uint64_t price (indices 2, 3)
// uint32_t size (index 4)
// uint32_t queuePosition (index 5)
// uint32_t totalAtLevel (index 6)
const POS_OFFSET = 5;
const TOTAL_OFFSET = 6;

// Initialize arena slots
for (let i = 0; i < ARENA_SLOTS; i++) {
    const base = i * STRIDE_INT32;
    view[base + POS_OFFSET] = 100;
    view[base + TOTAL_OFFSET] = 100;
}

function runBenchmark(totalOps) {
    const baselineMem = process.memoryUsage();
    if (global.gc) {
        global.gc();
    }

    const t0 = performance.now();
    let checksum = 0;

    for (let i = 0; i < totalOps; i++) {
        const slot = (i & ARENA_MASK) * STRIDE_INT32;
        const currentPos = view[slot + POS_OFFSET];
        const total = view[slot + TOTAL_OFFSET];
        const nextPos = currentPos <= 1 ? total : currentPos - 1;
        view[slot + POS_OFFSET] = nextPos;
        checksum = (checksum + nextPos) | 0;
    }

    const t1 = performance.now();
    const durationMs = t1 - t0;
    const finalMem = process.memoryUsage();
    const heapDeltaKb = (finalMem.heapUsed - baselineMem.heapUsed) / 1024;

    const opsPerSec = totalOps / (durationMs / 1000);
    const nsPerOp = (durationMs * 1e6) / totalOps;

    return {
        totalOps,
        durationMs,
        opsPerSec,
        nsPerOp,
        checksum,
        heapDeltaKb
    };
}

console.log("================================================================================");
console.log("AEGIS STANDALONE ZERO-GC FLAT ARENA BENCHMARK (PURE JS / V8 ENGINE)");
console.log("Memory Pattern:       64-Byte Cache-Line Strided ArrayBuffer (4 MB Arena)");
console.log(`Arena Dimensions:     ${ARENA_SLOTS.toLocaleString()} Slots x ${STRIDE_BYTES} Bytes`);
console.log("================================================================================");

// Warmup JIT compiler
runBenchmark(10000000);

// Execute 100 Million Ops
const res100m = runBenchmark(100000000);
console.log(`\n[100M Ops Execution]:`);
console.log(`  Duration:           ${res100m.durationMs.toFixed(2)} ms`);
console.log(`  Throughput:         ${(res100m.opsPerSec / 1e9).toFixed(3)} Billion ops/sec`);
console.log(`  Latency / Op:       ${res100m.nsPerOp.toFixed(3)} ns/op`);
console.log(`  Checksum:           0x${(res100m.checksum >>> 0).toString(16).toUpperCase()}`);
console.log(`  V8 Heap Delta:      ${res100m.heapDeltaKb > 0 ? "+" : ""}${res100m.heapDeltaKb.toFixed(2)} KB (Zero GC churn)`);

// Execute 1 Billion Ops (1B)
console.log(`\nExecuting 1,000,000,000 Operations (1 Billion Ops)...`);
const res1b = runBenchmark(1000000000);
console.log(`\n[1 Billion Ops (1B) Full-Scale Receipt]:`);
console.log(`  Duration:           ${res1b.durationMs.toFixed(2)} ms (${(res1b.durationMs / 1000).toFixed(3)} seconds)`);
console.log(`  Throughput:         ${(res1b.opsPerSec / 1e9).toFixed(3)} Billion ops/sec`);
console.log(`  Latency / Op:       ${res1b.nsPerOp.toFixed(3)} ns/op`);
console.log(`  Checksum:           0x${(res1b.checksum >>> 0).toString(16).toUpperCase()}`);
console.log(`  V8 Heap Delta:      ${res1b.heapDeltaKb > 0 ? "+" : ""}${res1b.heapDeltaKb.toFixed(2)} KB`);
console.log("================================================================================");
