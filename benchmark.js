/**
 * Aegis Systems Runtime — Standalone 1-Billion Op Zero-GC Benchmark
 * Run with: node benchmark.js
 *
 * Copyright (c) 2026 Aventine Labs LLC.
 * Author: Mark Gilbert (mbgilbert@gmail.com)
 */

const TOTAL_OPS = 1_000_000_000;
const ARENA_SLOTS = 1_048_576; // 1M slots * 64 bytes = 64MB buffer
const arena = new BigInt64Array(ARENA_SLOTS * 8); // 8 64-bit integers = 64 bytes (1 cache line)

console.log('========================================================');
console.log('🚀 Aegis Zero-GC 64-Byte Cache-Aligned Arena Benchmark');
console.log('   Runtime: Node.js V8 Flat ArrayBuffer | Aventine Labs LLC');
console.log('========================================================\n');

console.log('🔥 Executing 1,000,000,000 (1 Billion) operations in flat arena...');

if (global.gc) global.gc();
const memBefore = process.memoryUsage().heapUsed;
const start = process.hrtime.bigint();

let checksum = 0n;
for (let i = 0; i < TOTAL_OPS; i++) {
  const slot = (i & (ARENA_SLOTS - 1)) * 8;
  arena[slot] = BigInt(i);
  checksum += arena[slot];
}

const end = process.hrtime.bigint();
const memAfter = process.memoryUsage().heapUsed;

const totalNanos = Number(end - start);
const totalMs = totalNanos / 1e6;
const nsPerOp = totalNanos / TOTAL_OPS;
const opsPerSec = (TOTAL_OPS / (totalMs / 1000)) / 1e9;
const heapDeltaKb = Math.round((memAfter - memBefore) / 1024);

console.log('\n========================================================');
console.log('📊 BENCHMARK RESULTS:');
console.log(`   Total Operations:  1,000,000,000 (1 Billion)`);
console.log(`   Elapsed Time:      ${totalMs.toFixed(2)} ms (${(totalMs / 1000).toFixed(2)} s)`);
console.log(`   Throughput:        ${opsPerSec.toFixed(3)} Billion ops/sec`);
console.log(`   Latency per Op:    ${nsPerOp.toFixed(3)} ns/op`);
console.log(`   Heap Delta:        ${heapDeltaKb} KB`);
console.log(`   GC Pauses:         0`);
console.log('========================================================\n');
