#!/bin/bash
# ==============================================================================
# Aventine Labs LLC - Aegis Linux AI Soak Harness (Intelligent 1-Click Execution)
# PyTorch RFC-0036 Empirical Reproduction Suite
# Hardware Target: NVIDIA GeForce RTX 50-Series Laptop GPU (Ubuntu MATE 24.04.3 LTS)
# ==============================================================================

set -eo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

INVOKE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}${BOLD}AVENTINE LABS: AEGIS 60-MINUTE LINUX REPRODUCTION HARNESS (RFC-0036)${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "Launched from:    $INVOKE_DIR"
echo -e "Kernel Version:   $(uname -r)"

# ------------------------------------------------------------------------------
# 0. Emergency Sync Trap Handler (Ensures USB receives data even if interrupted)
# ------------------------------------------------------------------------------
WORK_DIR="$INVOKE_DIR"
RESULTS_DIR="$WORK_DIR/results"

sync_results_to_usb() {
    local exit_code=$?
    echo ""
    echo -e "${YELLOW}[+] Running automated result synchronization...${NC}"
    
    local usb_dest=""
    if [[ "$INVOKE_DIR" == /media/* ]]; then
        usb_dest="$INVOKE_DIR/results"
    else
        for candidate in /media/*/*/aegis_linux_usb_bundle; do
            if [ -d "$candidate" ]; then
                usb_dest="$candidate/results"
                break
            fi
        done
        if [ -z "$usb_dest" ]; then
            for candidate in /media/*/*; do
                if [ -d "$candidate" ] && [ -w "$candidate" ]; then
                    usb_dest="$candidate/aegis_linux_usb_bundle/results"
                    break
                fi
            done
        fi
    fi

    if [ -d "$RESULTS_DIR" ] && [ "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
        if [ -n "$usb_dest" ]; then
            echo -e "${GREEN}[+] Preserving telemetry to persistent USB: $usb_dest...${NC}"
            mkdir -p "$usb_dest"
            cp -v "$RESULTS_DIR"/* "$usb_dest"/ 2>/dev/null || true
            sync
            echo -e "${GREEN}[✓] Data safely flushed to USB flash drive.${NC}"
        else
            echo -e "${RED}[!] Notice: USB destination not found. Files preserved in RAM at: $RESULTS_DIR${NC}"
        fi
    fi

    if [ $exit_code -ne 0 ]; then
        echo -e "${YELLOW}[!] Process finished with status code $exit_code.${NC}"
    fi
}
trap sync_results_to_usb EXIT INT TERM

# ------------------------------------------------------------------------------
# 1. Environment Staging & Filesystem Diagnostics
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 0/5] Pre-Flight Diagnostics & Staging...${NC}"

# Detect if running from FAT32 / noexec mount (e.g., /media/*)
if [[ "$INVOKE_DIR" == /media/* ]] || ! [ -w "$INVOKE_DIR" ]; then
    WORK_DIR="/tmp/aegis_linux_workdir"
    echo -e "${YELLOW}[+] Removable or read-only/FAT32 media detected.${NC}"
    echo -e "${YELLOW}[+] Staging bundle into high-speed RAM workspace: $WORK_DIR...${NC}"
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
    cp -a "$INVOKE_DIR"/* "$WORK_DIR"/
fi

RESULTS_DIR="$WORK_DIR/results"
mkdir -p "$RESULTS_DIR"
cd "$WORK_DIR"

echo -e "Active Workdir:   $WORK_DIR"
echo -e "Local Results:    $RESULTS_DIR"
echo -e "Available RAM:    $(free -h | awk '/^Mem:/ {print $7}') free of $(free -h | awk '/^Mem:/ {print $2}')"
echo -e "Available /tmp:   $(df -h /tmp | awk 'NR==2 {print $4}')"

# Check Secure Boot Status
SECURE_BOOT_ACTIVE=0
if [ -d /sys/firmware/efi ]; then
    if command -v mokutil &> /dev/null; then
        if mokutil --sb-state 2>/dev/null | grep -qi "enabled"; then
            SECURE_BOOT_ACTIVE=1
        fi
    fi
fi

if [ $SECURE_BOOT_ACTIVE -eq 1 ]; then
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}${BOLD}[!] ATTENTION: UEFI SECURE BOOT IS DETECTED AS ENABLED${NC}"
    echo -e "${YELLOW}Secure Boot enforces kernel module signature verification.${NC}"
    echo -e "${YELLOW}On ephemeral Live USB sessions, MOK enrollments cannot persist across reboots.${NC}"
    echo -e "${YELLOW}If the NVIDIA driver fails to load below, please reboot into BIOS (F2) and${NC}"
    echo -e "${YELLOW}set Secure Boot to DISABLED under Security / Boot settings.${NC}"
    echo -e "${RED}================================================================================${NC}"
fi

# ------------------------------------------------------------------------------
# 2. NVIDIA GPU Driver Initialization & Self-Healing
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 1/5] Verifying NVIDIA GPU & Driver...${NC}"

# Check for nouveau conflict and auto-blacklist
if lsmod | grep -q nouveau; then
    echo -e "${YELLOW}[+] Open-source nouveau module detected. Setting up blacklist...${NC}"
    echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
    echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
    sudo modprobe -r nouveau 2>/dev/null || sudo rmmod nouveau 2>/dev/null || true
fi

# Function to verify if nvidia-smi is currently working
check_nvidia_smi() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

if ! check_nvidia_smi; then
    echo -e "${YELLOW}[!] Working NVIDIA driver not detected. Starting automated driver installation...${NC}"
    
    # 2a. Refresh repository metadata (DO NOT upgrade entire system in RAM!)
    echo -e "[+] Refreshing package index..."
    sudo apt update -qq

    # 2b. Install matching kernel headers
    echo -e "[+] Ensuring matching kernel headers: linux-headers-$(uname -r)..."
    sudo apt install -y linux-headers-$(uname -r)

    # 2c. Identify and install target Blackwell/RTX 50 driver
    TARGET_DRIVER=""
    if apt-cache show nvidia-driver-595-open &> /dev/null; then
        TARGET_DRIVER="nvidia-driver-595-open"
    elif apt-cache show nvidia-driver-580-open &> /dev/null; then
        TARGET_DRIVER="nvidia-driver-580-open"
    elif apt-cache show nvidia-driver-595 &> /dev/null; then
        TARGET_DRIVER="nvidia-driver-595"
    elif apt-cache show nvidia-driver-580 &> /dev/null; then
        TARGET_DRIVER="nvidia-driver-580"
    fi

    if [ -n "$TARGET_DRIVER" ]; then
        echo -e "${GREEN}[+] Installing $TARGET_DRIVER and DKMS utilities...${NC}"
        sudo apt install -y "$TARGET_DRIVER" nvidia-dkms-595-open nvidia-modprobe 2>/dev/null || \
        sudo apt install -y "$TARGET_DRIVER" nvidia-modprobe
    else
        echo -e "${YELLOW}[+] Running ubuntu-drivers auto-install...${NC}"
        sudo ubuntu-drivers install
    fi

    # 2d. Check DKMS status and trigger build if necessary
    if command -v dkms &> /dev/null; then
        echo -e "[+] Checking DKMS module build state..."
        dkms status || true
        sudo dkms autoinstall || true
    fi

    # 2e. Sequential kernel module insertion with error reporting
    echo -e "[+] Loading NVIDIA kernel modules..."
    if ! sudo modprobe nvidia; then
        echo -e "${RED}[!] 'modprobe nvidia' returned non-zero.${NC}"
        echo -e "${YELLOW}[+] Kernel diagnostic log (dmesg tail):${NC}"
        dmesg | grep -iE 'secure|lockdown|nvidia|rejected|signature' | tail -n 15 || true
    fi
    sudo modprobe nvidia_modeset 2>/dev/null || true
    sudo modprobe nvidia_uvm 2>/dev/null || true

    # 2f. Automated Device Node Creation (Fixes missing /dev/nvidia* nodes)
    echo -e "[+] Initializing device character nodes (/dev/nvidia* & /dev/nvidia-uvm)..."
    if command -v nvidia-modprobe &> /dev/null; then
        sudo nvidia-modprobe -u -c 0 2>/dev/null || true
    fi

    # Fallback mknod if nodes are absent but module is resident
    if [ ! -e /dev/nvidia0 ] && lsmod | grep -q nvidia; then
        NV_MAJOR=$(awk '$2=="nvidia-frontend" {print $1}' /proc/devices 2>/dev/null || true)
        if [ -n "$NV_MAJOR" ]; then
            sudo mknod -m 666 /dev/nvidia0 c "$NV_MAJOR" 0 2>/dev/null || true
            sudo mknod -m 666 /dev/nvidiactl c "$NV_MAJOR" 255 2>/dev/null || true
        fi
    fi
    if [ ! -e /dev/nvidia-uvm ] && lsmod | grep -q nvidia_uvm; then
        UVM_MAJOR=$(awk '$2=="nvidia-uvm" {print $1}' /proc/devices 2>/dev/null || true)
        if [ -n "$UVM_MAJOR" ]; then
            sudo mknod -m 666 /dev/nvidia-uvm c "$UVM_MAJOR" 0 2>/dev/null || true
        fi
    fi
    sudo chmod 666 /dev/nvidia* 2>/dev/null || true
fi

# 2g. Verify final driver status
if check_nvidia_smi; then
    echo -e "${GREEN}[✓] NVIDIA Driver & GPU verified successfully!${NC}"
    nvidia-smi
else
    echo -e "\n${RED}================================================================================${NC}"
    echo -e "${RED}${BOLD}[!] CRITICAL: NVIDIA GPU Driver could not communicate with hardware.${NC}"
    if [ $SECURE_BOOT_ACTIVE -eq 1 ]; then
        echo -e "${YELLOW}Root Cause: UEFI Secure Boot is active and blocked the DKMS driver module.${NC}"
        echo -e "${GREEN}Remedy: Restart laptop -> press F2 -> F7 (Advanced) -> Security/Boot ->${NC}"
        echo -e "${GREEN}        Set 'Secure Boot' to DISABLED -> F10 (Save & Exit).${NC}"
    fi
    echo -e "${YELLOW}Kernel log entries:${NC}"
    dmesg | grep -iE 'nvidia|secure|lockdown|nouveau' | tail -n 12 || true
    echo -e "${RED}================================================================================${NC}"
    echo -e "${YELLOW}Continuing execution with available compute devices...${NC}"
fi

# ------------------------------------------------------------------------------
# 3. Verify Native Build Tools (GCC)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 2/5] Checking Build Tools (GCC)...${NC}"
if ! command -v gcc &> /dev/null; then
    echo -e "${YELLOW}[+] Installing build-essential (GCC)...${NC}"
    sudo apt update -qq
    sudo apt install -y build-essential
fi
echo -e "${GREEN}[✓] GCC version: $(gcc --version | head -n 1)${NC}"

# ------------------------------------------------------------------------------
# 4. Compile Native C Kernels (Flat Arena & Telemetry Engine)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 3/5] Compiling Native C Flat Arena & Telemetry Kernels...${NC}"
gcc -O3 -mavx2 -shared -fPIC aegis_feeder.c -o aegis_feeder.so
gcc -O3 -mavx2 -shared -fPIC aegis_telemetry_engine.c -o aegis_telemetry.so
chmod 755 aegis_feeder.so aegis_telemetry.so
echo -e "${GREEN}[✓] Native shared objects compiled successfully:${NC}"
ls -lh aegis_feeder.so aegis_telemetry.so

# ------------------------------------------------------------------------------
# 5. Setup Python Environment & PyTorch with CUDA
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 4/5] Setting up Python Virtual Environment & PyTorch...${NC}"
if ! dpkg -s python3-venv &> /dev/null; then
    sudo apt update -qq
    sudo apt install -y python3-venv python3-pip
fi

VENV_DIR="/tmp/aegis_soak_env"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "[+] Initializing clean virtual environment in RAM: $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# Install PyTorch with CUDA support if not present
if ! python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo -e "[+] Installing PyTorch with CUDA support..."
    pip install torch numpy --index-url https://download.pytorch.org/whl/cu128 || \
    pip install torch numpy --index-url https://download.pytorch.org/whl/cu126
fi

# Print PyTorch environment banner
python3 -c "import torch; print(f'[✓] PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Host CPU\"}')"

# ------------------------------------------------------------------------------
# 6. Launch 60-Minute Prolonged Soak Test (RFC-0036)
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}================================================================================${NC}"
echo -e "${GREEN}${BOLD}[Step 5/5] Launching 60-Minute Aegis Soak Harness (Linux Native)...${NC}"
echo -e "Real-time CSV stream: $RESULTS_DIR/aegis_soak_linux_60min.csv"
echo -e "${BLUE}================================================================================${NC}"

python3 "$WORK_DIR/soak_test_aegis_linux.py" --duration_minutes 60.0 --dataset soak_corpus

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}${BOLD}[✓] 60-MINUTE LINUX SOAK TEST COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo -e "Results are preserved on your USB drive under aegis_linux_usb_bundle/results."
echo -e "You can now safely shut down or reboot back into Windows."
echo -e "================================================================================"
