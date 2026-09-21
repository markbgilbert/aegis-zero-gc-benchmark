#!/usr/bin/env python3
"""
Aegis Benchmark Telemetry Verification Script
Zero-dependency verifier using standard library csv module.
Verifies integrity, monotonicity, and latency bounds of smoke and soak test CSV telemetry.
"""

import sys
import os
import csv

def verify_csv(file_path, min_rows=100):
    if not os.path.exists(file_path):
        print(f"[FAIL] Missing target telemetry file: {file_path}")
        return False
    
    print(f"--- Verifying Telemetry: {file_path} ---")
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        
        # 1. Required columns check
        required_cols = [
            "step", "elapsed_s", "loss", "feeder_us", 
            "train_ms", "tokens_per_sec", "vram_alloc_mb", "vram_reserved_mb"
        ]
        for col in required_cols:
            if col not in fieldnames:
                print(f"[FAIL] Missing required column: {col}")
                return False
                
        rows = 0
        last_step = -1
        last_elapsed = -1.0
        feeder_latencies = []
        tokens_per_sec_list = []
        
        for row in reader:
            rows += 1
            step = int(row["step"])
            elapsed = float(row["elapsed_s"])
            feeder_us = float(row["feeder_us"])
            tok_sec = float(row["tokens_per_sec"])
            
            # Monotonicity check
            if step <= last_step:
                print(f"[FAIL] Step non-monotonic at row {rows}: {step} <= {last_step}")
                return False
            if elapsed < last_elapsed:
                print(f"[FAIL] Elapsed time decreased at row {rows}: {elapsed} < {last_elapsed}")
                return False
                
            last_step = step
            last_elapsed = elapsed
            
            # Exclude first 5 steps warm-up
            if rows > 5:
                feeder_latencies.append(feeder_us)
                tokens_per_sec_list.append(tok_sec)
                
    print(f"Row count: {rows} (threshold >= {min_rows})")
    if rows < min_rows:
        print(f"[FAIL] Expected at least {min_rows} rows, got {rows}")
        return False
        
    avg_feeder_us = sum(feeder_latencies) / len(feeder_latencies) if feeder_latencies else 0.0
    sorted_latencies = sorted(feeder_latencies)
    p99_idx = int(len(sorted_latencies) * 0.99)
    p99_feeder_us = sorted_latencies[p99_idx] if sorted_latencies else 0.0
    
    print(f"Feeder Latency: Mean = {avg_feeder_us:.2f} us | p99 = {p99_feeder_us:.2f} us")
    
    # Assert host feeder is under 150 us on average (Aegis target is ~7-50 us)
    if avg_feeder_us > 150.0:
        print(f"[FAIL] Average feeder latency exceeded threshold (150 us): {avg_feeder_us:.2f} us")
        return False
        
    avg_tok_sec = sum(tokens_per_sec_list) / len(tokens_per_sec_list) if tokens_per_sec_list else 0.0
    print(f"Throughput: Mean = {avg_tok_sec:,.0f} tok/s")
    if avg_tok_sec < 10000:
        print(f"[FAIL] Throughput below valid bounds: {avg_tok_sec:,.0f} tok/s")
        return False
        
    print(f"[PASS] Telemetry verification succeeded for {file_path}")
    return True

def main():
    target_files = ["test_10min.csv", "aegis_soak_linux_60min_100.csv"]
    verified_any = False
    
    for f in target_files:
        if os.path.exists(f):
            if not verify_csv(f):
                sys.exit(1)
            verified_any = True
            
    if not verified_any:
        print("[FAIL] No telemetry CSV files found to verify.")
        sys.exit(1)
        
    print("ALL TELEMETRY INTEGRITY CHECKS PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()
