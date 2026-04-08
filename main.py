import argparse
from config import (
    DEFAULT_MODEL_NAME, DEFAULT_QGEN_MODEL, DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, DEFAULT_BATCH_SIZE, DEFAULT_OUTPUT_DIR,
    DATASETS_CONFIG, get_file_paths
)
from data_loader import load_and_preprocess_dataset
from pipeline import stage1_chunking, stage2_embedding, stage3_indexing
from retriever import RAGRetriever
from evaluation import evaluate_retrieval


def parse_args():
    parser = argparse.ArgumentParser(description="RAG 多阶段数据预处理与索引构建模块")
    
    # 核心参数
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS_CONFIG.keys()),
                        help="选择要处理的数据集")
    parser.add_argument("--max_samples", type=int, default=1000, 
                        help="截取数据集前N条以测试 (设为0表示处理全部)")
    
    # 模型与路径参数
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--qgen_model", type=str, default=DEFAULT_QGEN_MODEL)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    
    # 超参数
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    
    # 功能开关
    parser.add_argument("--skip_pipeline", action="store_true", help="跳过构建直接测试检索")
    parser.add_argument("--query", type=str, default="What is RAG technology?", help="测试检索的查询词")
    
    # 检索方法和评估参数
    parser.add_argument("--retrieval_method", type=str, default="hybrid", 
                        choices=["dense", "sparse", "hybrid"], 
                        help="选择检索策略: dense(稠密), sparse(稀疏), hybrid(混合)")
    parser.add_argument("--run_eval", action="store_true", 
                        help="是否执行自动化评估 (如果不加此参数，则默认只做单条交互式测试)")
    parser.add_argument("--eval_samples", type=int, default=100, 
                        help="参与评估的样本数量 (默认 100)")
    
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 获取数据集的具体切分 (train/test等) 用于命名
    dataset_split = DATASETS_CONFIG[args.dataset]["split"]
    
    # 获取动态文件路径字典
    paths = get_file_paths(
        base_dir=args.output_dir,
        dataset_name=args.dataset,
        split=dataset_split,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    if not args.skip_pipeline:
        print("="*60)
        print(f"🚀 开始处理数据集: {args.dataset}")
        print("="*60)
        
        # 0. 加载数据
        docs, meta = load_and_preprocess_dataset(args.dataset, max_samples=args.max_samples)
        
        # 1. 阶段一：分块
        stage1_chunking(docs, meta, args.dataset, args.model_name, args.chunk_size, args.chunk_overlap, paths)
        
        # 2. 阶段二：向量化
        stage2_embedding(args.model_name, args.batch_size, args.chunk_size, paths)
        
        # 3. 阶段三：建库
        stage3_indexing(paths)
        
        print(f"\n🎉 🎯 {args.dataset} 的处理任务全部完成！")

    # 4. 测试检索
    print("\n" + "="*60)
    print(f"🧪 检索功能测试 [{args.dataset}]")
    print("="*60)
    try:
        retriever = RAGRetriever(
            paths=paths,
            embed_model_name=args.model_name,
            qgen_model_name=args.qgen_model,
            chunk_size=args.chunk_size,
            mock=True
        )

        # 分支执行逻辑：评估模式 vs 单条测试模式
        if args.run_eval:
            evaluate_retrieval(
                retriever=retriever,
                dataset_name=args.dataset,
                top_k=3,
                method=args.retrieval_method,
                sample_size=args.eval_samples
            )
        else:
            # 单条交互测试模式
            print("\n" + "="*60)
            print(f"🧪 检索功能单条测试 [{args.dataset} | Method: {args.retrieval_method}]")
            print("="*60)
            
            test_queries = {
                "Natural_Questions": "Who wrote the song 'Hey Jude'?",
                "PubMedQA": "Do mitochondria play a role in programmed cell death?",
                "FinanceBench": "What was the capital expenditure for 3M in 2018?",
                "HotpotQA": "Were Scott Derrickson and Ed Wood of the same nationality?"
            }
            
            query = test_queries.get(args.dataset, "What is the main topic?")
            results = retriever.search(query, top_k=3, method=args.retrieval_method)
            
            for r in results:
                print(f"[{r['排名']}] Chunk ID: {r['块ID']} | Dataset: {r['数据集']}")
                print(f" -> 来源文档 ID: {r['meta_info'].get('hotpot_id', r['meta_info'].get('pubid', 'N/A'))}")
    except FileNotFoundError as e:
        print(f"检索失败: 找不到对应的索引文件，请先完整运行 pipeline。错误详情: {e}")

if __name__ == "__main__":
    main()