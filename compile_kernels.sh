#!/bin/bash
set -e

echo "================================================================================"
echo "AEGIS NATIVE LINUX COMPILATION (GCC -O3 -mavx2)"
echo "================================================================================"

echo "[+] Compiling aegis_feeder.so (Zero-Allocation Flat Arena Feeder)..."
gcc -O3 -mavx2 -shared -fPIC aegis_feeder.c -o aegis_feeder.so

echo "[+] Compiling aegis_telemetry.so (64-Byte Cache-Aligned FNV-1a Audit Engine)..."
gcc -O3 -mavx2 -shared -fPIC aegis_telemetry_engine.c -o aegis_telemetry.so

echo "[✓] Successfully generated native Linux shared libraries:"
ls -lh aegis_feeder.so aegis_telemetry.so
echo "================================================================================"
