# RAGPreprocessNew

一个面向 RAG（Retrieval-Augmented Generation）的数据预处理与检索基线工程，支持以下能力：

- 多数据集统一加载与清洗（Natural Questions / PubMedQA / FinanceBench / HotpotQA）
- 三阶段离线构建流程：文本分块 -> 向量化 -> 索引构建（BM25 + FAISS）
- 在线检索测试：稀疏检索与稠密检索结果融合
- 数据集统计分析与本地抽取数据导出

## 1. 项目结构

```text
RAGPreprocessNew/
├── main.py                         # 主入口：执行 pipeline 与检索测试
├── config.py                       # 全局配置、默认超参数、数据集配置、产物路径生成
├── data_loader.py                  # 数据加载 + 字段适配 + 文本清洗
├── pipeline.py                     # 三阶段构建：chunk / embedding / index
├── retriever.py                    # 混合检索器（BM25 + FAISS）
├── utils.py                        # clean_text / embedding / query rewrite
├── corpus_statistics.py            # 数据下载、query-context 提取、统计分析
├── test.py                         # 对齐检查脚本（chunks/embeddings/meta）
├── dataset_statistics_summary.csv  # 统计摘要样例
├── raw_data/                       # 原始数据缓存（json）
├── rag_datasets_extracted/         # 抽取后的 query-context 数据
├── rag_output/                     # 预处理与索引产物目录
└── indexes/                        # 其他索引目录（可选）
```

## 2. 环境与依赖

建议 Python 3.10+。

### 2.1 创建虚拟环境（可选）

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2.2 安装依赖

```bash
pip install -U \
	torch \
	transformers \
	datasets \
	tqdm \
	numpy \
	pandas \
	faiss-cpu \
	rank-bm25 \
	langchain-text-splitters
```

如果你使用 Apple Silicon 并希望加速，可按本机环境替换为适配版本（例如 faiss 或 torch 的平台包）。

## 3. 核心流程说明

主流程位于 `main.py`：

1. 读取命令行参数与数据集配置
2. 加载并清洗数据（`data_loader.py`）
3. 阶段 1：按 token 长度分块并保存 chunks/meta（`pipeline.py`）
4. 阶段 2：使用 Contriever 生成 dense embeddings
5. 阶段 3：构建 BM25 和 FAISS 索引
6. 使用 `RAGRetriever` 进行检索测试

默认模型与参数：

- Embedding 模型：facebook/contriever-msmarco
- Query 重写模型：google/flan-t5-small
- chunk_size：512
- chunk_overlap：50
- batch_size：8
- output_dir：./rag_output

## 4. 快速开始

### 4.1 完整构建 + 检索测试

```bash
python main.py \
	--dataset Natural_Questions \
	--max_samples 1000 \
	--chunk_size 512 \
	--chunk_overlap 50 \
	--batch_size 8 \
	--query "when did richmond last play in a preliminary final"
```

### 4.2 仅检索（跳过构建）

前提是对应数据集的索引产物已经存在于 `rag_output/<dataset>/`。

```bash
python main.py \
	--dataset Natural_Questions \
	--chunk_size 512 \
	--chunk_overlap 50 \
	--skip_pipeline \
	--query "when did richmond last play in a preliminary final"
```

### 4.3 可选参数

- `--dataset`：必填，支持 `Natural_Questions` / `PubMedQA` / `FinanceBench` / `HotpotQA`
- `--max_samples`：默认 1000，设为 0 表示全量处理
- `--model_name`：embedding 模型名
- `--qgen_model`：query 重写模型名
- `--output_dir`：输出目录
- `--chunk_size`：分块 token 长度
- `--chunk_overlap`：块重叠 token 数
- `--batch_size`：向量化批大小
- `--skip_pipeline`：跳过构建，直接做检索
- `--query`：测试查询语句

## 5. 输出产物说明

每个数据集在 `rag_output/<dataset>/` 下生成：

- `<prefix>_chunks.pkl`：分块文本
- `<prefix>_meta.json`：块级元数据
- `<prefix>_embeddings.npy`：dense 向量
- `<prefix>_bm25.pkl`：BM25 索引
- `<prefix>_faiss.index`：FAISS 索引

其中 `<prefix>` 规则为：

```text
{dataset}_{split}_cs{chunk_size}_co{chunk_overlap}
```

例如：

```text
Natural_Questions_train_cs512_co50
```

## 6. 检索逻辑（当前实现）

`retriever.py` 的 `search` 逻辑：

1. 先做 query rewrite
2. BM25 取前 `top_k*3`
3. FAISS 取前 `top_k*3`
4. 按顺序融合并去重，返回前 `top_k`

返回字段为：排名、数据集、块 ID。

注意：`RAGRetriever` 默认 `mock=True`，因此 query rewrite 会输出 `[MOCK] ...` 前缀。如果需要真实重写，可将该默认值改为 `False` 或在初始化处显式传入。

## 7. 数据统计脚本

运行统计流程：

```bash
python corpus_statistics.py
```

该脚本会：

1. 下载并保存各数据集原始样本到 `raw_data/`
2. 提取 `query + context` 到 `rag_datasets_extracted/`
3. 计算统计指标并汇总到 `dataset_statistics_summary.csv`

当前仓库中已包含一个统计摘要样例文件。

## 8. 结果校验脚本

运行：

```bash
python test.py
```

用于检查以下问题：

- chunks / embeddings / meta 数量是否一一对应
- embeddings 维度是否正常
- 抽样查看文本、元数据和向量是否存在 NaN/Inf

注意：`test.py` 里 `PREFIX` 当前是硬编码示例路径，与你实际目录结构不一致时，需要先修改为正确前缀（例如 `./rag_output/HotpotQA/HotpotQA_train_cs512_co50`）。

## 9. 常见问题

### 9.1 `--skip_pipeline` 后报索引文件不存在

请先执行一次不带 `--skip_pipeline` 的完整构建，确保 `rag_output/<dataset>/` 下包含 bm25/faiss/meta/chunks/embeddings。

### 9.2 下载数据集或模型较慢

- 确保网络可访问 Hugging Face
- 可提前设置镜像或缓存目录
- 项目默认使用 `~/.cache/huggingface`

### 9.3 内存或显存不足

- 减小 `--max_samples`
- 减小 `--batch_size`
- 先在单一数据集上验证流程

## 10. 后续可改进方向

- 将依赖固定到 `requirements.txt`
- 为 `RAGRetriever` 增加可配置融合打分
- 增加检索质量指标（Recall@k、MRR）
- 增加单元测试与端到端回归测试



处理 FinanceBench 数据集并修改分块参数：

Bash
python main.py --dataset FinanceBench --chunk_size 1024 --chunk_overlap 100 --max_samples 500
处理 HotpotQA 数据集并修改批处理大小：

Bash
python main.py --dataset HotpotQA --batch_size 16
跳过数据构建阶段，直接对 Natural_Questions 进行检索测试：

Bash
python main.py --dataset Natural_Questions --chunk_size 512 --skip_pipeline --query "What is the capital of France?"

测试一：只测试 BM25 稀疏检索的性能

Bash
python main.py --dataset PubMedQA --skip_pipeline --retrieval_method sparse --run_eval --eval_samples 200
测试二：只测试 Contriever 稠密检索的性能

Bash
python main.py --dataset PubMedQA --skip_pipeline --retrieval_method dense --run_eval --eval_samples 200


场景 1：端到端完整运行（预处理 + 建库 + 默认单条测试）
如果你刚下载了一个新的数据集（比如 PubMedQA），想要从零开始清洗数据、分块、生成 Embedding、构建 FAISS 和 BM25 索引，并最后跑一个简单的测试：

Bash
python main.py --dataset PubMedQA --chunk_size 512 --chunk_overlap 50
提示：如果不指定 --max_samples，默认可能会截取前 1000 条（取决于你的 default 设置），如果想跑全量数据可以加上 --max_samples 0。

场景 2：跳过预处理，直接进行单条交互式检索测试
如果你已经建好库了，只是修改了检索逻辑（比如从 dense 改成了 hybrid），想要快速验证一下检索结果，不需要重新建库：

Bash
python main.py --dataset HotpotQA --skip_pipeline --retrieval_method hybrid
说明：--skip_pipeline 会跳过耗时的阶段 1、2、3，直接加载之前保存的索引并进入测试模式。

场景 3：跳过预处理，运行批量文件检索（你刚刚尝试的场景）
如果你已经准备好了一个 JSON 文件包含大量问题，想要批量检索并保存结果：

Bash
python main.py --dataset HotpotQA --skip_pipeline --query_file data/queries.json --retrieval_method hybrid --batch_size_queries 50
场景 4：针对自有数据集的内部评估模式 (Recall@K)
如果你的数据集中本身带有 Ground Truth 答案/ID，想要评估当前检索方法的召回率（假设你代码里写了 --run_eval 分支）：

Bash
python main.py --dataset FinanceBench --skip_pipeline --run_eval --eval_samples 100 --retrieval_method dense
说明：这会随机抽取 100 个问题，跑一遍 Dense 检索，并统计准确命中的召回率。

场景 5：使用不同的切分策略建库以对比效果
想要测试更小的 Chunk Size 对 RAG 效果的影响：

Bash
python main.py --dataset Natural_Questions --chunk_size 256 --chunk_overlap 20 --max_samples 5000
说明：这会在 output_dir 下生成带有新参数标识（如 cs256_co20）的新索引文件，不会覆盖之前的 cs512 版本。