#!/bin/bash

# 本地 vLLM 集成示例脚本
# 演示如何通过本地部署的 vLLM 模型完成生成任务

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}本地 vLLM 生成任务示例${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查是否有检索结果文件
RETRIEVAL_FILE=$(ls -1t ./retrieve_output/retrieval_detailed_*.json 2>/dev/null | head -1)

if [ -z "$RETRIEVAL_FILE" ]; then
    echo -e "${YELLOW}[!] 没有找到检索结果文件${NC}"
    echo -e "${YELLOW}    请先运行: python main.py --mode retrieve --dataset HotpotQA${NC}\n"
    exit 1
fi

echo -e "${GREEN}[✓] 找到检索结果文件: $RETRIEVAL_FILE${NC}\n"

# 示例 1: 使用快捷参数 --use_vllm（推荐）
echo -e "${YELLOW}示例 1: 使用 --use_vllm 快捷参数（推荐）${NC}"
echo -e "${YELLOW}命令:${NC}"
echo -e "python main.py \\"
echo -e "    --mode generate_retrieve \\"
echo -e "    --dataset HotpotQA \\"
echo -e "    --retrieval_input_file $RETRIEVAL_FILE \\"
echo -e "    --generation_output_dir ./generation_output \\"
echo -e "    --generation_output_prefix hotpot_vllm_quick \\"
echo -e "    --use_vllm \"Qwen/Qwen2.5-7B\" \\"
echo -e "    --max_tokens 128 \\"
echo -e "    --temperature 0.1\n"

read -p "是否运行示例 1？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}[*] 运行示例 1...${NC}\n"
    python main.py \
        --mode generate_retrieve \
        --dataset HotpotQA \
        --retrieval_input_file "$RETRIEVAL_FILE" \
        --generation_output_dir ./generation_output \
        --generation_output_prefix hotpot_vllm_quick \
        --use_vllm "Qwen/Qwen2.5-7B" \
        --max_tokens 128 \
        --temperature 0.1
    echo -e "\n${GREEN}[✓] 示例 1 完成${NC}\n"
fi

# 示例 2: 使用显式 API 参数
echo -e "${YELLOW}示例 2: 使用显式 API 参数（需要指定端口）${NC}"
echo -e "${YELLOW}命令:${NC}"
echo -e "python main.py \\"
echo -e "    --mode generate_retrieve \\"
echo -e "    --dataset HotpotQA \\"
echo -e "    --retrieval_input_file $RETRIEVAL_FILE \\"
echo -e "    --generation_output_dir ./generation_output \\"
echo -e "    --generation_output_prefix hotpot_api_explicit \\"
echo -e "    --use_api \\"
echo -e "    --gen_model Qwen/Qwen2.5-7B \\"
echo -e "    --api_base_url http://127.0.0.1:8000/v1 \\"
echo -e "    --api_key EMPTY \\"
echo -e "    --max_tokens 128 \\"
echo -e "    --temperature 0.1\n"

read -p "是否运行示例 2？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}[*] 运行示例 2...${NC}\n"
    python main.py \
        --mode generate_retrieve \
        --dataset HotpotQA \
        --retrieval_input_file "$RETRIEVAL_FILE" \
        --generation_output_dir ./generation_output \
        --generation_output_prefix hotpot_api_explicit \
        --use_api \
        --gen_model Qwen/Qwen2.5-7B \
        --api_base_url http://127.0.0.1:8000/v1 \
        --api_key EMPTY \
        --max_tokens 128 \
        --temperature 0.1
    echo -e "\n${GREEN}[✓] 示例 2 完成${NC}\n"
fi

# 示例 3: 使用 Instruct 模型
echo -e "${YELLOW}示例 3: 使用 Instruct 模型${NC}"
echo -e "${YELLOW}命令:${NC}"
echo -e "python main.py \\"
echo -e "    --mode generate_retrieve \\"
echo -e "    --dataset HotpotQA \\"
echo -e "    --retrieval_input_file $RETRIEVAL_FILE \\"
echo -e "    --generation_output_dir ./generation_output \\"
echo -e "    --generation_output_prefix hotpot_vllm_instruct \\"
echo -e "    --use_vllm \"Qwen/Qwen2.5-7B-Instruct\" \\"
echo -e "    --max_tokens 128 \\"
echo -e "    --temperature 0.1\n"

read -p "是否运行示例 3？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}[*] 运行示例 3...${NC}\n"
    python main.py \
        --mode generate_retrieve \
        --dataset HotpotQA \
        --retrieval_input_file "$RETRIEVAL_FILE" \
        --generation_output_dir ./generation_output \
        --generation_output_prefix hotpot_vllm_instruct \
        --use_vllm "Qwen/Qwen2.5-7B-Instruct" \
        --max_tokens 128 \
        --temperature 0.1
    echo -e "\n${GREEN}[✓] 示例 3 完成${NC}\n"
fi

# 显示结果
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}生成任务完成${NC}"
echo -e "${GREEN}========================================${NC}\n"
echo -e "查看生成结果:"
echo -e "  ${BLUE}详细结果${NC}: ls -lh ./generation_output/*_detailed_*.json"
echo -e "  ${BLUE}摘要统计${NC}: ls -lh ./generation_output/*_summary_*.json"
echo -e "  ${BLUE}预测答案${NC}: ls -lh ./generation_output/*_predictions_*.json"
echo -e "\n"
