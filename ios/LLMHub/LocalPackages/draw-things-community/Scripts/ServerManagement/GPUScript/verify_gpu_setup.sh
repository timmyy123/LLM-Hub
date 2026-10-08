#!/bin/bash

# GPU Setup Verification Script
# This script verifies that Docker, CUDA, and NVIDIA Container Toolkit are properly installed
#
# Usage:
#   ./verify_gpu_setup.sh                    # Run locally
#   ./verify_gpu_setup.sh user@hostname      # Run on remote host via SSH

REMOTE_HOST="$1"

# If remote host provided, run via SSH
if [ -n "$REMOTE_HOST" ]; then
    echo "Running verification on remote host: $REMOTE_HOST"
    echo ""
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" 'bash -s' < "$0"
    exit $?
fi

echo ""
echo "🔍 GPU Server Setup Verification"
echo "=================================================="
echo ""

# Exit code tracking
EXIT_CODE=0

# --- Check NVIDIA Driver ---
echo "1️⃣  Checking NVIDIA Driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ NVIDIA Driver is installed"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | while read line; do
        echo "      GPU: $line"
    done
else
    echo "   ❌ NVIDIA Driver not found (nvidia-smi command not available)"
    EXIT_CODE=1
fi
echo ""

# --- Check Docker ---
echo "2️⃣  Checking Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "   ✅ Docker is installed: $DOCKER_VERSION"

    # Check if Docker daemon is running
    if sudo docker info &> /dev/null; then
        echo "   ✅ Docker daemon is running"
    else
        echo "   ⚠️  Docker is installed but daemon is not running"
        EXIT_CODE=1
    fi
else
    echo "   ❌ Docker not found"
    EXIT_CODE=1
fi
echo ""

# --- Check NVIDIA Container Toolkit ---
echo "3️⃣  Checking NVIDIA Container Toolkit..."
if command -v nvidia-ctk &> /dev/null; then
    NVIDIA_CTK_VERSION=$(nvidia-ctk --version 2>&1 | head -n 1)
    echo "   ✅ NVIDIA Container Toolkit is installed: $NVIDIA_CTK_VERSION"
else
    echo "   ❌ NVIDIA Container Toolkit not found"
    EXIT_CODE=1
fi
echo ""

# --- Check Docker GPU Integration ---
echo "4️⃣  Checking Docker GPU Integration..."
if command -v docker &> /dev/null && sudo docker info &> /dev/null; then
    # Check if nvidia runtime is configured
    if sudo docker info 2>/dev/null | grep -q "nvidia"; then
        echo "   ✅ NVIDIA runtime is configured in Docker"
    else
        echo "   ⚠️  NVIDIA runtime may not be configured in Docker"
    fi

    # Try to run a test container (optional, only if nvidia-smi is available)
    if command -v nvidia-smi &> /dev/null; then
        echo "   Testing GPU access from Docker container..."
        if sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "   ✅ Docker can access GPU successfully"
        else
            echo "   ⚠️  Docker GPU test failed (this may be due to network issues or missing nvidia/cuda image)"
        fi
    fi
else
    echo "   ⏭️  Skipping Docker GPU integration check (Docker not available)"
fi
echo ""

# --- Summary ---
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All checks passed! GPU server setup is complete."
    echo ""
    echo "You can now:"
    echo "  • Run nvidia-smi to check GPU status"
    echo "  • Run Docker containers with GPU access using --gpus all flag"
else
    echo "⚠️  Some checks failed. Please review the output above."
    echo ""
    echo "Common fixes:"
    echo "  • If Docker daemon is not running, run: sudo systemctl start docker"
    echo "  • A system reboot may be required after installation"
fi
echo "=================================================="
echo ""

exit $EXIT_CODE
