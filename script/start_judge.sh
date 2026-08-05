#!/bin/bash

set -e

MODEL_PATH="/path/to/your/OneEmo/ckpts/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"
HOST="0.0.0.0"
PORT=15555
GPU_ID=1
GPU_UTIL=0.95
MAX_MODEL_LEN=8192

echo "=========================================="
echo " vLLM Server - ${MODEL_NAME}"
echo "=========================================="

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path not found: $MODEL_PATH"
    exit 1
fi

LOCAL_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "Model:       ${MODEL_NAME}"
echo "GPU:         CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "Listen:      ${HOST}:${PORT}"
echo ""
echo "Local:       http://localhost:${PORT}"
echo "Network:     http://${LOCAL_IP}:${PORT}"
echo ""
echo "Client usage:"
echo "  base_url = \"http://${LOCAL_IP}:${PORT}/v1\""
echo ""

if command -v ufw &>/dev/null; then
    if ufw status 2>/dev/null | grep -q "Status: active"; then
        if ! ufw status | grep -q "15555"; then
            echo "[WARN] Firewall is active, port ${PORT} may be blocked."
            echo "  Run: sudo ufw allow ${PORT}/tcp"
            echo ""
        fi
    fi
fi

echo "Starting server..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_MODEL_LEN}
