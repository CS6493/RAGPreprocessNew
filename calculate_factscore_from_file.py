import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from config import LOCAL_GEN_MODEL_CHOICES, VLLM_LOCAL_CONFIG
from evaluation import calculate_fact_score_via_llm
from generator import RAGGenerator


def _load_items(input_file: str) -> Dict[str, Any]:
    with open(input_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return payload
    if isinstance(payload, list):
        return {"overall_metrics": {}, "items": payload}

    raise ValueError("输入文件必须是 JSON 数组，或包含 'items' 字段的对象。")


def _build_generator(args: argparse.Namespace) -> RAGGenerator:
    if args.use_vllm:
        cfg = VLLM_LOCAL_CONFIG[args.use_vllm]
        return RAGGenerator(
            model_name=args.use_vllm,
            use_api=True,
            api_key=cfg["api_key"],
            api_base_url=cfg["base_url"],
            max_tokens=args.max_tokens,
            temperature=0.0,
        )

    if args.use_api:
        if not args.model_name or not args.api_base_url:
            raise ValueError("use_api 模式下需要同时提供 --model_name 和 --api_base_url。")
        return RAGGenerator(
            model_name=args.model_name,
            use_api=True,
            api_key=args.api_key,
            api_base_url=args.api_base_url,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )

    if not args.model_name:
        raise ValueError("本地权重模式下需要提供 --model_name。")
    return RAGGenerator(
        model_name=args.model_name,
        use_api=False,
        use_4bit=args.use_4bit,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )


def _should_skip_item(generated_answer: str, skip_insufficient: bool) -> bool:
    if not generated_answer.strip():
        return True
    if not skip_insufficient:
        return False
    return generated_answer.strip().lower().startswith("insufficient evidence")


def compute_factscore_for_file(
    input_file: str,
    output_file: str,
    generator: RAGGenerator,
    overwrite_existing: bool,
    skip_insufficient: bool,
    max_items: int,
) -> Dict[str, Any]:
    payload = _load_items(input_file)
    items: List[Dict[str, Any]] = payload.get("items", [])

    processed = 0
    scored = 0
    fact_total = 0

    target_items = items[:max_items] if max_items > 0 else items

    for item in target_items:
        processed += 1
        generated_answer = str(item.get("generated_answer", "")).strip()
        knowledge_used = str(item.get("knowledge_used", "")).strip()

        if not overwrite_existing and item.get("fact_score") is not None:
            try:
                fact_total += int(item.get("fact_score"))
                scored += 1
            except Exception:
                pass
            continue

        if _should_skip_item(generated_answer, skip_insufficient):
            item["fact_score"] = None
            continue

        contexts = [knowledge_used] if knowledge_used else []
        if not contexts:
            item["fact_score"] = None
            continue

        fact = int(calculate_fact_score_via_llm(generated_answer, contexts, generator))
        item["fact_score"] = fact
        fact_total += fact
        scored += 1

    overall = payload.get("overall_metrics", {})
    overall["factscore_scored_count"] = scored
    overall["factscore_processed_count"] = processed
    overall["FActScore"] = (fact_total / scored) * 100 if scored > 0 else 0.0
    overall["factscore_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["overall_metrics"] = overall

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "output_file": output_file,
        "processed": processed,
        "scored": scored,
        "FActScore": overall["FActScore"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从已有 generation 结果文件单独计算 FActScore")
    parser.add_argument("--input_file", type=str, required=True, help="输入结果文件（detailed 或 list）")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径，不传则自动生成")

    parser.add_argument(
        "--use_vllm",
        type=str,
        default=None,
        choices=LOCAL_GEN_MODEL_CHOICES,
        help="使用本地 vLLM 快捷模式",
    )
    parser.add_argument("--use_api", action="store_true", help="使用自定义 API（OpenAI-compatible）")
    parser.add_argument("--model_name", type=str, default=None, help="评估用模型名")
    parser.add_argument("--api_base_url", type=str, default=None, help="API base url")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--use_4bit", action="store_true", help="本地权重模式下启用 4bit")

    parser.add_argument("--max_tokens", type=int, default=16, help="judge 输出 token 上限")
    parser.add_argument("--overwrite_existing", action="store_true", help="覆盖已有 fact_score")
    parser.add_argument(
        "--skip_insufficient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否跳过 'Insufficient evidence' 的样本（默认开启）",
    )
    parser.add_argument("--max_items", type=int, default=0, help="仅处理前 N 条，0 表示全部")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

    output_file = args.output_file
    if not output_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(args.input_file)
        output_file = f"{base}_with_factscore_{ts}{ext or '.json'}"

    generator = _build_generator(args)
    res = compute_factscore_for_file(
        input_file=args.input_file,
        output_file=output_file,
        generator=generator,
        overwrite_existing=args.overwrite_existing,
        skip_insufficient=args.skip_insufficient,
        max_items=args.max_items,
    )

    print("\n" + "=" * 60)
    print("FActScore 计算完成")
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {res['output_file']}")
    print(f"处理样本数: {res['processed']}")
    print(f"计分样本数: {res['scored']}")
    print(f"FActScore: {res['FActScore']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
