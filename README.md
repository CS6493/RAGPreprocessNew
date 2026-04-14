# RAGPreprocessNew

RAG 预处理、检索、生成、评估一体化工程。

当前代码基于 mode 统一入口，核心目标是支持两条并行流程：

- 我们自己的检索与生成评估流程（retrieve_output -> generation_output）
- 第三方检索结果流程（例如 data/test_knowledge.json）

两条流程相互独立，但生成与评估指标逻辑保持一致。

## 1. 目录与产物

- rag_output/: 预处理与索引产物（chunks/meta/embeddings/BM25/FAISS）
- retrieve_output/: 批量检索输出（每题检索结果 + 检索指标）
- generation_output/: 生成与评估输出（每题答案 + EM/F1/FActScore + 汇总）
- data/queries.json: 我们流程的批量检索输入
- data/test_knowledge.json: 第三方检索结果输入

默认路径配置见 config.py。

## 2. 环境准备

```bash
conda create -y -n cs6493rag python=3.10
conda activate cs6493rag
pip install -r requirements.txt
```

## 3. 统一入口模式

主入口：main.py

支持模式：

- pipeline: 只做预处理建库
- retrieve: 只做批量检索并输出到 retrieve_output
- retrieve_eval: 只做检索评估（Recall 风格）
- e2e_eval: 检索 + 生成端到端评估
- generate_retrieve: 从 retrieve_output 读取检索结果做生成评估
- evaluate_generated: 只评估已经生成的预测文件
- generate_offline: 对第三方离线检索结果做生成/评估
- all: 串行执行 retrieve + generate_retrieve

查看全部参数：

```bash
python main.py --help
```

## 4. HotpotQA 常用命令

### 4.1 首次构建索引（若已有 rag_output 可跳过）

```bash
python main.py \
	--mode pipeline \
	--dataset HotpotQA \
	--max_samples 1000 \
	--chunk_size 512 \
	--chunk_overlap 50 \
	--batch_size 8
```

### 4.2 批量检索（读取 data/queries.json，输出到 retrieve_output）

```bash
python main.py \
	--mode retrieve \
	--dataset HotpotQA \
	--query_file ./data/queries.json \
	--retrieve_output_dir ./retrieve_output \
	--retrieval_method hybrid \
	--top_k 3
```

输出文件示例：

- retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json
- retrieve_output/retrieval_summary_YYYYMMDD_HHMMSS.json

每题会包含：

- retrieved_results
- hit_at_k
- top1_em
- top1_f1

### 4.3 基于检索结果做生成与评估（我们的流程）

```bash
python main.py \
	--mode generate_retrieve \
	--dataset HotpotQA \
	--retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
	--generation_output_dir ./generation_output \
	--generation_output_prefix hotpot_our \
	--use_api \
	--api_provider Qwen \
	--do_factscore
```

如果你想直接使用本地部署的 Qwen 模型，可以把上面的 API 参数替换成本地模型选择，例如：

```bash
python main.py \
	--mode generate_retrieve \
	--dataset HotpotQA \
	--retrieval_input_file ./retrieve_output/retrieval_detailed_YYYYMMDD_HHMMSS.json \
	--generation_output_dir ./generation_output \
	--generation_output_prefix hotpot_local \
	--local_gen_model Qwen/Qwen2.5-7B-Instruct \
	--do_factscore
```

可选的本地模型只有两个：`Qwen/Qwen2.5-7B` 和 `Qwen/Qwen2.5-7B-Instruct`。

说明：

- 若不传 --retrieval_input_file，会自动选择 retrieve_output 下最新 retrieval_detailed 文件。
- 输出中包含 question、answer、generated_answer、knowledge_used、retrieved_results、EM/F1/FActScore。

### 4.4 只评估已有生成结果

```bash
python main.py \
	--mode evaluate_generated \
	--generated_predictions_file ./generation_output/hotpot_our_predictions_YYYYMMDD_HHMMSS.json \
	--generation_output_dir ./generation_output
```

### 4.5 第三方检索结果流程（不影响我们的检索流程）

```bash
python main.py \
	--mode generate_offline \
	--offline_file ./data/test_knowledge.json \
	--offline_mode both \
	--generation_output_dir ./generation_output \
	--offline_output_prefix thirdparty_hotpot \
	--use_api \
	--api_provider Qwen \
	--do_factscore
```

offline_mode 可选：

- generate: 只生成
- evaluate: 只评估
- both: 生成 + 评估

### 4.6 一条命令串行执行我们的检索+生成

```bash
python main.py \
	--mode all \
	--dataset HotpotQA \
	--query_file ./data/queries.json \
	--retrieve_output_dir ./retrieve_output \
	--generation_output_dir ./generation_output \
	--generation_output_prefix hotpot_all \
	--use_api \
	--api_provider Qwen \
	--do_factscore
```

## 5. 检索模块单独执行

可独立运行 retriever.py（不通过 main.py）：

```bash
python retriever.py \
	--dataset HotpotQA \
	--query_file ./data/queries.json \
	--retrieve_output_dir ./retrieve_output \
	--retrieval_method hybrid \
	--top_k 3
```

## 6. UI 启动

```bash
streamlit run app.py
```

建议先确保对应数据集索引已存在于 rag_output。

## 7. 常见问题

### 7.1 没有输出到目标文件

请确认：

- mode 是否正确（例如 retrieve 才会写 retrieve_output）
- 输入文件后缀是否正确（.json 不是 .jon）
- 是否同时传了互斥目的参数导致未进入预期分支

### 7.2 使用 API 生成时报错

请检查：

- --use_api / --api_provider 参数
- config.py 中 API 配置
- 网络与服务可用性

### 7.3 跳过 pipeline 后检索器初始化失败

说明索引文件不存在或 chunk 参数不一致。请先运行 pipeline，或确保 chunk_size/chunk_overlap 与已建索引一致。

