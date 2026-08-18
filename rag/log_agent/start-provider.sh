#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/dell/huggingface-model-server"
VENV="$ROOT/.rag-venv"
APP_DIR="$ROOT/rag/log_agent"
LOG_FILE="$APP_DIR/provider-integration.log"

NVIDIA_LIBRARY_PATHS=$(
    find \
        "$VENV/lib/python3.12/site-packages/nvidia" \
        -type f \
        \( \
            -name 'libcudart.so.13*' \
            -o -name 'libcublas.so.13*' \
            -o -name 'libcublasLt.so.13*' \
        \) \
        -printf '%h\n' \
        2>/dev/null |
    sort -u |
    paste -sd: -
)

if [ -z "$NVIDIA_LIBRARY_PATHS" ]; then
    echo "CUDA runtime kütüphaneleri bulunamadı."
    exit 1
fi

export LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATHS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/rag:$APP_DIR:${PYTHONPATH:-}"

exec "$VENV/bin/uvicorn" \
    main:app \
    --app-dir "$APP_DIR" \
    --host 10.142.1.136 \
    --port 8000 \
    --workers 1
