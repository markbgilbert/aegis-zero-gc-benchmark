#!/bin/bash
# ==============================================================================
# Aventine Labs LLC - Aegis Linux AI 10-Minute Smoke Harness (Jemalloc Hardened)
# Fast Validation Suite: Checks for 0.00 MB VRAM delta and +0.25 MB flatline
# ==============================================================================

set -eo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

INVOKE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="$INVOKE_DIR"
RESULTS_DIR="$WORK_DIR/results"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}${BOLD}AVENTINE LABS: AEGIS 10-MINUTE LINUX SMOKE HARNESS (JEMALLOC HARDENED)${NC}"
echo -e "${BLUE}================================================================================${NC}"

# Check & install libjemalloc
if ! dpkg -s libjemalloc2 &> /dev/null; then
    echo -e "${YELLOW}[+] Installing libjemalloc2 and dev headers...${NC}"
    sudo apt update -qq
    sudo apt install -y libjemalloc2 libjemalloc-dev 2>/dev/null || true
fi

# Build C kernels if needed
if [ ! -f "$WORK_DIR/aegis_feeder.so" ] || [ ! -f "$WORK_DIR/aegis_telemetry.so" ]; then
    echo -e "${YELLOW}[+] Compiling native C kernels...${NC}"
    gcc -O3 -mavx2 -shared -fPIC "$WORK_DIR/aegis_feeder.c" -o "$WORK_DIR/aegis_feeder.so"
    gcc -O3 -mavx2 -shared -fPIC "$WORK_DIR/aegis_telemetry_engine.c" -o "$WORK_DIR/aegis_telemetry.so"
    chmod 755 "$WORK_DIR/aegis_feeder.so" "$WORK_DIR/aegis_telemetry.so"
fi

# Allocator Hardening
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TOP_PAD_=65536

if [ -f "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2" ]; then
    echo -e "${GREEN}[✓] Hardening allocator: Pre-loading libjemalloc.so.2${NC}"
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
elif [ -f "/usr/lib/libjemalloc.so.2" ]; then
    echo -e "${GREEN}[✓] Hardening allocator: Pre-loading libjemalloc.so.2${NC}"
    export LD_PRELOAD="/usr/lib/libjemalloc.so.2"
else
    echo -e "${YELLOW}[!] Note: libjemalloc.so.2 not found, using hardened glibc ptmalloc${NC}"
fi

# Python environment activation
VENV_DIR="/tmp/aegis_soak_env"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo -e "\n${BLUE}================================================================================${NC}"
echo -e "${GREEN}${BOLD}Launching 10-Minute Smoke Validation...${NC}"
echo -e "Streaming output to: $RESULTS_DIR/test_10min.csv"
echo -e "${BLUE}================================================================================${NC}\n"

python3 "$WORK_DIR/soak_test_aegis_linux.py" --duration_minutes 10.0 --dataset soak_corpus --csv test_10min.csv

echo -e "\n${GREEN}================================================================================${NC}"
echo -e "${GREEN}${BOLD}[✓] 10-MINUTE SMOKE VALIDATION COMPLETED!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo -e "If test_10min.csv shows flatline VmRSS and 0.00 MB VRAM delta, run ./run_linux_soak.sh for full 60-min production soak receipt."
