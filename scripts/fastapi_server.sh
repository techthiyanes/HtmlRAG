#!/bin/bash

python html4rag/fastapi_server.py \
    --port 80 \
    --host 0.0.0.0 \
    --uvicorn_log_level debug \
    --ckpt_path ../../huggingface/glm-4-9b-chat-1m