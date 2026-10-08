#!/bin/bash

# Script to start a single gRPCServerCLI process
# Usage: start_single_grpc.sh <port> <gpu> <address>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <port> <gpu> <address>"
    exit 1
fi

PORT=$1
GPU=$2
ADDRESS=$3

echo "Starting gRPCServerCLI on port $PORT using GPU $GPU at address $ADDRESS"
exec gRPCServerCLI /models \
    -a $ADDRESS \
    -p $PORT \
    -g $GPU \
    --max-crashes-within-time-window 10 \
    --cpu-offload \
    --echo-on-queue \
    --join "{\"host\":\"100.71.37.12\", \"port\":50002, \"servers\": [{\"address\":\"$ADDRESS\", \"port\":$PORT, \"priority\":1}]}" \
    --secondary-models-directory "/loraModels/" \
    --blob-store-access-key 7a2a1994c95022455bf8afc6a61a5e9f \
    --blob-store-secret 03dd45eb16a615636b32827778462b5584d4a1e0fa03a0e450e8b56c5ef006ea \
    --blob-store-endpoint https://cd96f610b0bb2657da157aca332052ec.r2.cloudflarestorage.com \
    --blob-store-bucket draw-things-byom \
    -d f395315025e2f0577c31e8b7fa5c7381 \
    --model-browser \
    --no-tls 