import argparse
import json
import os

from config import (
    DEFAULT_MODEL_NAME, DEFAULT_QGEN_MODEL, DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, DEFAULT_BATCH_SIZE, DEFAULT_OUTPUT_DIR,
    DEFAULT_GEN_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, # 新增引入
    DATASETS_CONFIG, get_file_paths
)
from data_loader import load_and_preprocess_dataset
from pipeline import stage1_chunking, stage2_embedding, stage3_indexing
from retriever import RAGRetriever
from generator import RAGGenerator  # 新增引入
from evaluation import evaluate_retrieval, evaluate_end_to_end # 新增引入


def parse_args():
    parser = argparse.ArgumentParser(description="RAG 端到端系统：预处理、索引、检索与生成")
    
    # --- 核心参数组 ---
    core_group = parser.add_argument_group("核心配置 (Core)")
    core_group.add_argument("--dataset", type=str, required=True, choices=list(DATASETS_CONFIG.keys()), help="选择要处理的数据集")
    core_group.add_argument("--max_samples", type=int, default=1000, help="截取数据集前N条以测试 (设为0表示处理全部)")
    core_group.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    core_group.add_argument("--skip_pipeline", action="store_true", help="跳过构建直接测试检索/生成")
    
    # --- 检索参数组 ---
    retrieval_group = parser.add_argument_group("检索配置 (Retrieval)")
    retrieval_group.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    retrieval_group.add_argument("--qgen_model", type=str, default=DEFAULT_QGEN_MODEL)
    retrieval_group.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    retrieval_group.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    retrieval_group.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    retrieval_group.add_argument("--retrieval_method", type=str, default="hybrid", choices=["dense", "sparse", "hybrid"])
    retrieval_group.add_argument("--top_k", type=int, default=3, help="检索返回的文档数量")
    
    # --- 生成参数组 (新增) ---
    gen_group = parser.add_argument_group("生成配置 (Generation)")
    gen_group.add_argument("--do_generate", action="store_true", help="是否启用答案生成模块")
    gen_group.add_argument("--gen_model", type=str, default=DEFAULT_GEN_MODEL, help="生成模型名称或路径")
    gen_group.add_argument("--use_api", action="store_true", help="是否使用 API 调用生成模型")
    gen_group.add_argument("--api_key", type=str, default=None, help="API Key (如果 use_api=True)")
    gen_group.add_argument("--api_base_url", type=str, default=None, help="API Base URL (如果 use_api=True)")
    gen_group.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    gen_group.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)

    # --- 评估与测试参数组 ---
    eval_group = parser.add_argument_group("评估与测试 (Eval & Testing)")
    eval_group.add_argument("--run_eval", action="store_true", help="是否执行自动化评估")
    eval_group.add_argument("--eval_samples", type=int, default=100, help="参与评估的样本数量")
    eval_group.add_argument("--query_file", type=str, default=None, help="批量查询的 JSON 文件路径")
    eval_group.add_argument("--batch_size_queries", type=int, default=10, help="批量测试时处理的 query 数量")
    eval_group.add_argument("--output_file", type=str, default="results/output.json", help="输出结果的保存路径")
    
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_split = DATASETS_CONFIG[args.dataset]["split"]
    paths = get_file_paths(args.output_dir, args.dataset, dataset_split, args.chunk_size, args.chunk_overlap)

    # ==================== 1. 执行/跳过 Pipeline ====================
    if not args.skip_pipeline:
        print("="*60)
        print(f"🚀 开始处理数据集: {args.dataset}")
        print("="*60)
        docs, meta = load_and_preprocess_dataset(args.dataset, max_samples=args.max_samples)
        stage1_chunking(docs, meta, args.dataset, args.model_name, args.chunk_size, args.chunk_overlap, paths)
        stage2_embedding(args.model_name, args.batch_size, args.chunk_size, paths)
        stage3_indexing(paths)
        print(f"\n🎉 🎯 {args.dataset} 的处理任务全部完成！")

    # ==================== 2. 初始化检索与生成组件 ====================
    print("\n" + "="*60)
    print(f"⚙️  初始化组件 (Dataset: {args.dataset})")
    print("="*60)
    
    try:
        retriever = RAGRetriever(
            paths=paths, embed_model_name=args.model_name,
            qgen_model_name=args.qgen_model, chunk_size=args.chunk_size, mock=True
        )
    except FileNotFoundError as e:
        print(f"❌ 检索器初始化失败: 找不到对应的索引文件，请先完整运行 pipeline。错误详情: {e}")
        return

    generator = None
    if args.do_generate:
        generator = RAGGenerator(
            model_name=args.gen_model,
            use_api=args.use_api,
            api_key=args.api_key,
            api_base_url=args.api_base_url,
            use_4bit=not args.use_api
        )

    # ==================== 3. 执行评估 / 测试模式 ====================
    
    # 模式 A: 自动化评估
    if args.run_eval:
        if args.do_generate:
            evaluate_end_to_end(
                retriever=retriever, generator=generator, dataset_name=args.dataset,
                top_k=args.top_k, method=args.retrieval_method, sample_size=args.eval_samples
            )
        else:
            evaluate_retrieval(
                retriever=retriever, dataset_name=args.dataset,
                top_k=args.top_k, method=args.retrieval_method, sample_size=args.eval_samples
            )

    # 模式 B: 批量读取 JSON 查询测试
    elif args.query_file:
        print(f"\n📂 开始批量测试 [{args.query_file}]")
        if not os.path.exists(args.query_file):
            print(f"❌ 找不到查询文件: {args.query_file}")
            return
            
        with open(args.query_file, "r", encoding="utf-8") as f:
            queries_data = json.load(f)[:args.batch_size_queries]
            
        all_results = []
        retriever.mock = True 
        from tqdm import tqdm
        
        for item in tqdm(queries_data, desc="Batch Processing"):
            original_query = item.get("query", "")
            if not original_query: continue
                
            results, final_query = retriever.search(original_query, top_k=args.top_k, method=args.retrieval_method)
            record = {
                "original_query": original_query,
                "rewritten_query": final_query,
                "retrieved_results": results
            }
            
            # 挂载 Generation 模块
            if args.do_generate:
                # 假设 retriever.py 暂未抛出原文本，这里可以使用 meta 里的信息，或者简单传入占位
                contexts = [r["meta_info"].get("text", f"Result {r['块ID']}") for r in results]
                answer = generator.generate(original_query, contexts, args.max_tokens, args.temperature)
                record["generated_answer"] = answer
                
            all_results.append(record)
            
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 批量任务完成！结果已保存至: {args.output_file}")

    # 模式 C: 单条交互测试
    else:
        test_queries = {
            "Natural_Questions": "Who wrote the song 'Hey Jude'?",
            "PubMedQA": "Do mitochondria play a role in programmed cell death?",
            "FinanceBench": "What was the capital expenditure for 3M in 2018?",
            "HotpotQA": "Were Scott Derrickson and Ed Wood of the same nationality?"
        }
        query = test_queries.get(args.dataset, "What is the main topic?")
        
        print(f"\n🤖 问题: {query}")
        results, final_query = retriever.search(query, top_k=args.top_k, method=args.retrieval_method)
        print(f"🔍 检索词: {final_query}")
        print("📄 召回文档:")
        for r in results:
            print(f"  -> [{r['排名']}] ID: {r['块ID']} | 来源: {r['meta_info'].get('hotpot_id', r['meta_info'].get('pubid', 'N/A'))}")
            
        if args.do_generate:
            contexts = [r["meta_info"].get("text", f"Context from block {r['块ID']}") for r in results]
            print("\n⏳ 正在生成答案...")
            answer = generator.generate(query, contexts, args.max_tokens, args.temperature)
            print("\n💡 生成的最终答案:")
            print("-" * 50)
            print(answer)
            print("-" * 50)

if __name__ == "__main__":
    main()