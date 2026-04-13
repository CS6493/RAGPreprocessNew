import argparse
import glob
import os

from config import (
    API_CONFIG,
    DATASETS_CONFIG,
    DEFAULT_API_PROVIDER,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GEN_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OFFLINE_KNOWLEDGE_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QGEN_MODEL,
    DEFAULT_RETRIEVE_OUTPUT_DIR,
    DEFAULT_RETRIEVE_QUERY_FILE,
    DEFAULT_TEMPERATURE,
    DEFAULT_GENERATION_OUTPUT_DIR,
    get_file_paths,
)
from data_loader import load_and_preprocess_dataset
from evaluation import (
    evaluate_end_to_end,
    evaluate_generated_predictions_file,
    evaluate_retrieval,
    run_generation_from_retrieval_output,
    run_offline_generation_and_evaluation,
)
from generator import RAGGenerator
from pipeline import stage1_chunking, stage2_embedding, stage3_indexing
from retriever import RAGRetriever, run_batch_retrieval


def parse_args():
    parser = argparse.ArgumentParser(description="RAG 统一入口：预处理、检索、生成、评估")

    core = parser.add_argument_group("核心配置")
    core.add_argument(
        "--mode",
        type=str,
        default="retrieve",
        choices=[
            "pipeline",
            "retrieve",
            "retrieve_eval",
            "e2e_eval",
            "generate_retrieve",
            "evaluate_generated",
            "generate_offline",
            "all",
        ],
        help="运行模式",
    )
    core.add_argument("--dataset", type=str, default="HotpotQA", choices=list(DATASETS_CONFIG.keys()))
    core.add_argument("--skip_pipeline", action="store_true", help="跳过 pipeline 阶段")
    core.add_argument("--max_samples", type=int, default=1000, help="pipeline 处理样本数 (0=全量)")

    retrieval = parser.add_argument_group("检索配置")
    retrieval.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="rag_output 根目录")
    retrieval.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    retrieval.add_argument("--qgen_model", type=str, default=DEFAULT_QGEN_MODEL)
    retrieval.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    retrieval.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    retrieval.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    retrieval.add_argument("--retrieval_method", type=str, default="hybrid", choices=["dense", "sparse", "hybrid"])
    retrieval.add_argument("--top_k", type=int, default=3)
    retrieval.add_argument("--eval_samples", type=int, default=100, help="检索评估/端到端评估样本数")

    retrieve_io = parser.add_argument_group("检索输入输出")
    retrieve_io.add_argument("--query_file", type=str, default=DEFAULT_RETRIEVE_QUERY_FILE)
    retrieve_io.add_argument("--retrieve_output_dir", type=str, default=DEFAULT_RETRIEVE_OUTPUT_DIR)
    retrieve_io.add_argument("--retrieve_sample_size", type=int, default=0, help="批量检索问题数 (0=全部)")
    retrieve_io.add_argument("--shuffle_queries", action="store_true")
    retrieve_io.add_argument("--retrieval_input_file", type=str, default=None, help="用于生成阶段的检索结果文件")

    generation = parser.add_argument_group("生成配置")
    generation.add_argument("--gen_model", type=str, default=DEFAULT_GEN_MODEL)
    generation.add_argument("--use_api", action="store_true")
    generation.add_argument("--use_local", action="store_true", help="强制使用本地模型")
    generation.add_argument("--api_provider", type=str, default=DEFAULT_API_PROVIDER, choices=list(API_CONFIG.keys()))
    generation.add_argument("--api_key", type=str, default=None)
    generation.add_argument("--api_base_url", type=str, default=None)
    generation.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    generation.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    generation.add_argument("--use_4bit", action="store_true")
    generation.add_argument("--do_factscore", action="store_true")
    generation.add_argument("--generation_output_dir", type=str, default=DEFAULT_GENERATION_OUTPUT_DIR)
    generation.add_argument("--generation_output_prefix", type=str, default="our_pipeline")

    generated_eval = parser.add_argument_group("已生成结果评估")
    generated_eval.add_argument("--generated_predictions_file", type=str, default=None)

    offline = parser.add_argument_group("第三方检索结果 (test_knowledge)")
    offline.add_argument("--offline_file", type=str, default=DEFAULT_OFFLINE_KNOWLEDGE_FILE)
    offline.add_argument("--offline_mode", type=str, default="both", choices=["generate", "evaluate", "both"])
    offline.add_argument("--offline_sample_size", type=int, default=50)
    offline.add_argument("--offline_output_prefix", type=str, default="offline_test_knowledge")

    return parser.parse_args()


def run_pipeline(args, paths):
    if args.skip_pipeline:
        print("⏭️  已跳过 pipeline")
        return

    print("=" * 60)
    print(f"🚀 开始处理数据集: {args.dataset}")
    print("=" * 60)
    docs, meta = load_and_preprocess_dataset(args.dataset, max_samples=args.max_samples)
    stage1_chunking(docs, meta, args.dataset, args.model_name, args.chunk_size, args.chunk_overlap, paths)
    stage2_embedding(args.model_name, args.batch_size, args.chunk_size, paths)
    stage3_indexing(paths)
    print(f"\n🎉 {args.dataset} pipeline 完成")


def build_retriever(args, paths):
    return RAGRetriever(
        paths=paths,
        embed_model_name=args.model_name,
        qgen_model_name=args.qgen_model,
        chunk_size=args.chunk_size,
        mock=True,
    )


def build_generator(args):
    resolved_use_api = (not args.use_local) and (args.use_api or args.api_key is not None)

    if resolved_use_api:
        provider_cfg = API_CONFIG[args.api_provider]
        model_name = provider_cfg.get("model", args.gen_model)
        api_key = args.api_key or provider_cfg.get("api_key")
        api_base_url = args.api_base_url or provider_cfg.get("base_url")
        return RAGGenerator(
            model_name=model_name,
            use_api=True,
            api_key=api_key,
            api_base_url=api_base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    return RAGGenerator(
        model_name=args.gen_model,
        use_api=False,
        use_4bit=args.use_4bit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def resolve_latest_retrieval_file(retrieve_output_dir):
    pattern = os.path.join(retrieve_output_dir, "retrieval_detailed_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main():
    args = parse_args()

    dataset_split = DATASETS_CONFIG[args.dataset]["split"]
    paths = get_file_paths(args.output_dir, args.dataset, dataset_split, args.chunk_size, args.chunk_overlap)

    retrieval_summary = None

    if args.mode in {"pipeline", "all"}:
        run_pipeline(args, paths)

    if args.mode in {"retrieve", "all"}:
        retriever = build_retriever(args, paths)
        retrieval_summary = run_batch_retrieval(
            retriever=retriever,
            query_file=args.query_file,
            output_dir=args.retrieve_output_dir,
            top_k=args.top_k,
            method=args.retrieval_method,
            sample_size=args.retrieve_sample_size,
            shuffle=args.shuffle_queries,
        )

    if args.mode == "retrieve_eval":
        retriever = build_retriever(args, paths)
        evaluate_retrieval(
            retriever=retriever,
            dataset_name=args.dataset,
            top_k=args.top_k,
            method=args.retrieval_method,
            sample_size=args.eval_samples,
        )

    if args.mode == "e2e_eval":
        retriever = build_retriever(args, paths)
        generator = build_generator(args)
        evaluate_end_to_end(
            retriever=retriever,
            generator=generator,
            dataset_name=args.dataset,
            top_k=args.top_k,
            method=args.retrieval_method,
            sample_size=args.eval_samples,
        )

    if args.mode in {"generate_retrieve", "all"}:
        generator = build_generator(args)
        retrieval_file = args.retrieval_input_file

        if not retrieval_file and retrieval_summary:
            retrieval_file = retrieval_summary.get("detailed_file")

        if not retrieval_file:
            retrieval_file = resolve_latest_retrieval_file(args.retrieve_output_dir)

        if not retrieval_file:
            raise FileNotFoundError(
                "未找到可用的检索输出文件，请先运行 mode=retrieve 或通过 --retrieval_input_file 指定文件"
            )

        run_generation_from_retrieval_output(
            generator=generator,
            retrieval_file=retrieval_file,
            sample_size=args.retrieve_sample_size,
            output_dir=args.generation_output_dir,
            output_prefix=args.generation_output_prefix,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_factscore=args.do_factscore,
            shuffle=args.shuffle_queries,
        )

    if args.mode == "evaluate_generated":
        if not args.generated_predictions_file:
            raise ValueError("evaluate_generated 模式需要通过 --generated_predictions_file 指定文件")
        evaluate_generated_predictions_file(
            predictions_file=args.generated_predictions_file,
            output_dir=args.generation_output_dir,
            output_prefix="generated_eval",
        )

    if args.mode == "generate_offline":
        generator = build_generator(args)
        run_offline_generation_and_evaluation(
            generator=generator,
            offline_file=args.offline_file,
            sample_size=args.offline_sample_size,
            output_dir=args.generation_output_dir,
            output_prefix=args.offline_output_prefix,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            run_generation=args.offline_mode in {"generate", "both"},
            run_evaluation=args.offline_mode in {"evaluate", "both"},
            do_factscore=args.do_factscore,
            shuffle=args.shuffle_queries,
        )


if __name__ == "__main__":
    main()
