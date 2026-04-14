#!/bin/bash

# 本地 vLLM 快速启动脚本
# 用法:
#   ./launch_vllm_local.sh base       # 启动 Qwen/Qwen2.5-7B 在 port 8000
#   ./launch_vllm_local.sh instruct   # 启动 Qwen/Qwen2.5-7B-Instruct 在 port 8001
#   ./launch_vllm_local.sh            # 同时启动两个模型

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

launch_base() {
    echo -e "${GREEN}[*] 启动 Qwen/Qwen2.5-7B 在 port 8000${NC}"
    echo -e "${YELLOW}    API endpoint: http://127.0.0.1:8000/v1${NC}"
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B \
        --served-model-name Qwen/Qwen2.5-7B \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 4096 \
        --dtype auto
}

launch_instruct() {
    echo -e "${GREEN}[*] 启动 Qwen/Qwen2.5-7B-Instruct 在 port 8001${NC}"
    echo -e "${YELLOW}    API endpoint: http://127.0.0.1:8001/v1${NC}"
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --served-model-name Qwen/Qwen2.5-7B-Instruct \
        --host 0.0.0.0 \
        --port 8001 \
        --max-model-len 4096 \
        --dtype auto
}

MODE=${1:-both}

if [ "$MODE" = "base" ]; then
    launch_base
elif [ "$MODE" = "instruct" ]; then
    launch_instruct
elif [ "$MODE" = "both" ]; then
    # 后台启动两个服务
    echo -e "${GREEN}[*] 在后台启动两个 vLLM 服务...${NC}"
    launch_base &
    BASE_PID=$!
    sleep 2
    launch_instruct &
    INSTRUCT_PID=$!
    
    echo -e "${GREEN}[✓] 两个服务已启动${NC}"
    echo -e "${YELLOW}    Base model (port 8000): PID $BASE_PID${NC}"
    echo -e "${YELLOW}    Instruct model (port 8001): PID $INSTRUCT_PID${NC}"
    
    # 等待用户中断
    trap "kill $BASE_PID $INSTRUCT_PID 2>/dev/null; exit 0" INT TERM
    wait
else
    echo "用法: $0 [base|instruct|both]"
    exit 1
fi
