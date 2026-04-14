# 本地 vLLM 部署与集成指南

本文档说明如何通过本地部署的 vLLM 服务完成生成任务。

## 快速开始

### 1. 启动 vLLM 服务

使用提供的启动脚本（推荐）：

```bash
# 启动两个模型（后台运行）
bash launch_vllm_local.sh both

# 或分别启动
bash launch_vllm_local.sh base       # Qwen/Qwen2.5-7B 在 port 8000
bash launch_vllm_local.sh instruct   # Qwen/Qwen2.5-7B-Instruct 在 port 8001
```

或手动启动：

```bash
# Base model（使用 /v1/completions）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B \
    --served-model-name Qwen/Qwen2.5-7B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096

# Instruct model（使用 /v1/chat/completions）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 4096
```

### 2. 验证服务状态

```bash
# 检查服务是否运行
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8001/v1/models
```

### 3. 测试集成

```bash
# 测试与 base 模型的连接
python test_vllm_integration.py --model "Qwen/Qwen2.5-7B"

# 测试与 instruct 模型的连接
python test_vllm_integration.py --model "Qwen/Qwen2.5-7B-Instruct"
```

## 三种使用方式

### 方式 1：使用 --use_vllm 快捷参数（推荐）

最简单的方式，自动配置 API endpoint：

```bash
# 使用 base 模型
python main.py \
    --mode generate_retrieve \
    --dataset HotpotQA \
    --retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
    --generation_output_dir ./generation_output \
    --generation_output_prefix hotpot_vllm_base \
    --use_vllm "Qwen/Qwen2.5-7B" \
    --max_tokens 128 \
    --temperature 0.1

# 使用 instruct 模型
python main.py \
    --mode generate_retrieve \
    --dataset HotpotQA \
    --retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
    --generation_output_dir ./generation_output \
    --generation_output_prefix hotpot_vllm_instruct \
    --use_vllm "Qwen/Qwen2.5-7B-Instruct" \
    --max_tokens 128 \
    --temperature 0.1
```

### 方式 2：显式指定 API 参数

如果需要自定义 port 或其他参数：

```bash
# Base 模型（使用 /v1/completions）
python main.py \
    --mode generate_retrieve \
    --dataset HotpotQA \
    --retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
    --generation_output_dir ./generation_output \
    --generation_output_prefix hotpot_api_base \
    --use_api \
    --gen_model Qwen/Qwen2.5-7B \
    --api_base_url http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --max_tokens 128 \
    --temperature 0.1

# Instruct 模型（使用 /v1/chat/completions）
python main.py \
    --mode generate_retrieve \
    --dataset HotpotQA \
    --retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
    --generation_output_dir ./generation_output \
    --generation_output_prefix hotpot_api_instruct \
    --use_api \
    --gen_model Qwen/Qwen2.5-7B-Instruct \
    --api_base_url http://127.0.0.1:8001/v1 \
    --api_key EMPTY \
    --max_tokens 128 \
    --temperature 0.1
```

### 方式 3：本地权重加载（仅限于已下载模型）

不通过 vLLM 服务，直接加载模型权重：

```bash
# Base 模型
python main.py \
    --mode generate_retrieve \
    --dataset HotpotQA \
    --retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
    --generation_output_dir ./generation_output \
    --generation_output_prefix hotpot_local_base \
    --local_gen_model "Qwen/Qwen2.5-7B" \
    --use_4bit \
    --max_tokens 128 \
    --temperature 0.1

# Instruct 模型
python main.py \
    --mode generate_retrieve \
    --dataset HotpotQA \
    --retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
    --generation_output_dir ./generation_output \
    --generation_output_prefix hotpot_local_instruct \
    --local_gen_model "Qwen/Qwen2.5-7B-Instruct" \
    --use_4bit \
    --max_tokens 128 \
    --temperature 0.1
```

## 性能优化

对于 vLLM 服务，可以根据硬件情况添加以下参数：

```bash
# 启用张量并行（多 GPU）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B \
    --served-model-name Qwen/Qwen2.5-7B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000

# 调整显存利用率
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --port 8001
```

## 验证与调试

### 1. 检查 API 响应格式

```bash
# Base 模型 - 使用 completions 接口
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B",
    "prompt": "What is 2 + 2?",
    "max_tokens": 50
  }'

# Instruct 模型 - 使用 chat/completions 接口
curl -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 50
  }'
```

### 2. 验证生成结果

生成完成后，检查输出文件：

```bash
# 查看详细结果（包含生成的答案、EM/F1 等指标）
cat ./generation_output/hotpot_vllm_base_detailed_YYYYMMDD_HHMMSS.json

# 查看摘要（汇总统计）
cat ./generation_output/hotpot_vllm_base_summary_YYYYMMDD_HHMMSS.json
```

### 3. 常见问题排查

| 问题 | 排查 |
|------|------|
| Connection refused | 检查 vLLM 服务是否启动：`ps aux \| grep vllm` |
| Model not found | 确认模型名称是否与 `--served-model-name` 一致 |
| Timeout 超时 | 增加 `max_tokens`，或减少 `--max-model-len` |
| CUDA OOM | 降低 `--gpu-memory-utilization` 或启用张量并行 |
| 生成不连贯 | 调整 `--temperature` 参数（0.1 较低，1.0 较高） |

## 参数说明

### vLLM 服务参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 要加载的模型 ID | 必填 |
| `--served-model-name` | 服务返回的模型名称 | 与 `--model` 相同 |
| `--host` | 监听主机 | 127.0.0.1 |
| `--port` | 监听端口 | 8000 |
| `--max-model-len` | 最大序列长度 | 自动推断 |
| `--gpu-memory-utilization` | GPU 显存利用率（0-1） | 0.9 |
| `--tensor-parallel-size` | GPU 张量并行数 | 1 |
| `--dtype` | 数据类型（auto/float32/float16/bfloat16） | auto |

### main.py 生成参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--use_vllm` | 使用本地 vLLM 快捷模式 | Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct |
| `--use_api` | 使用 API 模式 | - |
| `--gen_model` | 模型名（API 模式） | 任意 |
| `--api_base_url` | API 地址 | URL |
| `--api_key` | API 密钥 | 字符串 |
| `--local_gen_model` | 本地模型 | Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct |
| `--use_local` | 强制使用本地权重 | - |
| `--max_tokens` | 最大生成长度 | 整数 |
| `--temperature` | 生成温度 | 0-2 的浮点数 |

## 结果输出格式

生成后的输出文件保持一致性，包含：

```json
{
  "question": "问题内容",
  "gold_answer": "标准答案",
  "generated_answer": "模型生成的答案",
  "em": 0 或 1,  // 精确匹配
  "f1": 0.0-1.0,  // F1 分数
  "factscore": 0.0-1.0,  // 事实得分（如启用）
  "retrieved_results": [...],  // 检索结果
  "knowledge_used": "使用的知识文本"
}
```

## 并行部署

如需在多个 GPU 上并行部署两个模型：

```bash
# Terminal 1: 启动 base 模型
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B \
    --port 8000

# Terminal 2: 启动 instruct 模型
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8001
```

然后同时运行任务：

```bash
# Terminal 3
python main.py --mode generate_retrieve ... --use_vllm "Qwen/Qwen2.5-7B" ...

# Terminal 4
python main.py --mode generate_retrieve ... --use_vllm "Qwen/Qwen2.5-7B-Instruct" ...
```
