import argparse
import json
import os
import random
import re
import string
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from config import DEFAULT_GENERATION_OUTPUT_DIR

def evaluate_retrieval(retriever, dataset_name, top_k=3, method="hybrid", sample_size=100):
    """
    评估检索系统: 包括 Recall@K, NDCG@K, MRR 等指标
    """
    print(f"\n📊 开始执行评估 (Dataset: {dataset_name} | Method: {method} | Top-K: {top_k})")
    
    # 1. 提取所有不重复的问题作为测试集
    unique_queries = {}
    for item in retriever.meta:
        # 不同数据集的 query 和 answer 字段可能不同，这里做统一适配
        q = item.get("question") or item.get("original_question") or item.get("query")
        a = item.get("answer", item.get("final_decision", item.get("long_answer", item)))        # 提取能够标识该文档唯一性的 ID
        doc_id = item.get("source_id", item.get("financebench_id", item.get("hotpot_id", item.get("pubid", a))))        

        if q and doc_id and q not in unique_queries:
            unique_queries[q] = {"doc_id": doc_id, "answer": a}

    all_queries = list(unique_queries.keys())
    if not all_queries:
        print("❌ 无法从 meta 数据中提取出有效的 Question-Answer 对，请检查 data_loader.py 中 meta 的构建。")
        return

    # 2. 随机采样
    if sample_size > 0 and sample_size < len(all_queries):
        test_queries = random.sample(all_queries, sample_size)
    else:
        test_queries = all_queries

    hit_count = 0
    ndcg_total = 0.0
    mrr_total = 0.0
    
    # 3. 执行批量检索并统计指标
    for q in tqdm(test_queries, desc="Evaluating"):
        ground_truth_id = unique_queries[q]["doc_id"]
        
        # 禁用 query 重写的 mock 输出打印，以免打乱终端显示
        retriever.mock = True 
        
        results, _ = retriever.search(query=q, top_k=top_k, method=method)
        
        # 计算 Recall@K
        is_hit = False
        for res in results:
            retrieved_meta = res["meta_info"]
            retrieved_id = retrieved_meta.get("source_id", retrieved_meta.get("financebench_id", retrieved_meta.get("hotpot_id", retrieved_meta.get("pubid", retrieved_meta.get("answer")))))            
            if retrieved_id == ground_truth_id:
                is_hit = True
                break
                
        if is_hit:
            hit_count += 1
        
        # 计算 NDCG@K
        ndcg_k = ndcg_at_k(ground_truth_id, results)
        if ndcg_k is not None:
            ndcg_total += ndcg_k
        
        # 计算 MRR
        mrr_k = mrr(ground_truth_id, results)
        if mrr_k is not None:
            mrr_total += mrr_k

    recall = (hit_count / len(test_queries)) * 100
    ndcg_avg = (ndcg_total / len(test_queries)) * 100 if len(test_queries) > 0 else 0.0
    mrr_avg = (mrr_total / len(test_queries)) * 100 if len(test_queries) > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"📈 评估结果报告")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔹 检索策略: {method.upper()}")
    print(f"🔹 测试样本数: {len(test_queries)}")
    print(f"🔹 Recall@{top_k}: {recall:.2f}% ({hit_count}/{len(test_queries)})")
    print(f"🔹 NDCG@{top_k}: {ndcg_avg:.2f}%")
    print(f"🔹 MRR: {mrr_avg:.2f}%")
    print("="*50 + "\n")
    
    return {
        "recall": recall,
        "ndcg": ndcg_avg,
        "mrr": mrr_avg,
    }


# ===================== 生成指标评估 (Generation Eval) =====================

def normalize_answer(s):
    """标准化答案，去除标点、冠词，统一小写，用于计算 EM 和 F1"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def recall_at_k(gold_source_id, retrieved_results):
    """
    计算 Recall@K: 检查金标答案对应的文档是否在前K个检索结果中
    Args:
        gold_source_id: 金标文档的source_id
        retrieved_results: 检索结果列表
    Returns:
        1 if recall hit else 0
    """
    if not gold_source_id:
        return None
    
    for result in retrieved_results:
        retrieved_source_id = result.get("meta_info", {}).get("source_id", "")
        if str(retrieved_source_id).strip() == str(gold_source_id).strip():
            return 1
    return 0

def max_normalized_f1(gold_answer, retrieved_results):
    """
    计算 Max-Normalized F1: 处理chunk大小不一的情况
    取前K个检索结果中与金标答案F1分数最高的那个
    Args:
        gold_answer: 金标答案文本
        retrieved_results: 检索结果列表
    Returns:
        max F1 score among all retrieved results
    """
    if not gold_answer or not retrieved_results:
        return None
    
    max_f1 = 0.0
    for result in retrieved_results:
        answer_text = result.get("meta_info", {}).get("answer", "")
        if answer_text:
            current_f1 = f1_score(answer_text, gold_answer)
            max_f1 = max(max_f1, current_f1)
    
    return max_f1

def ndcg_at_k(gold_source_id, retrieved_results):
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    假设相关性为二元: 1 if 文档匹配gold_source_id, 0 otherwise
    Args:
        gold_source_id: 金标文档ID
        retrieved_results: 检索结果列表 (按排名排序)
    Returns:
        NDCG@K score (0.0 - 1.0)
    """
    if not gold_source_id:
        return None
    
    # 构建相关性标签
    relevance = []
    for result in retrieved_results:
        retrieved_source_id = result.get("meta_info", {}).get("source_id", "")
        is_relevant = 1 if str(retrieved_source_id).strip() == str(gold_source_id).strip() else 0
        relevance.append(is_relevant)
    
    # 计算 DCG
    dcg = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            dcg += rel / (2 ** (i + 1))  # log2(i+2) = log2(rank+1)
    
    # 计算 IDCG (理想DCG，第一个result相关)
    idcg = 1.0 / 1.0  # 第一个result如果相关，IDCG = 1 / (2^1) = 0.5, 但理想情况是1.0
    if len(relevance) > 0:
        idcg = sum(1.0 / (2 ** (i + 1)) for i in range(1) if i < len(relevance))
    
    # NDCG = DCG / IDCG
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    
    return min(ndcg, 1.0)

def mrr(gold_source_id, retrieved_results):
    """
    计算 MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名
    Args:
        gold_source_id: 金标文档ID
        retrieved_results: 检索结果列表
    Returns:
        1/rank if found, 0 if not found
    """
    if not gold_source_id:
        return None
    
    for rank, result in enumerate(retrieved_results, start=1):
        retrieved_source_id = result.get("meta_info", {}).get("source_id", "")
        if str(retrieved_source_id).strip() == str(gold_source_id).strip():
            return 1.0 / rank
    
    return 0.0

def calculate_fact_score_via_llm(prediction, contexts, generator):
    """
    轻量级 FActScore 实现 (LLM-as-a-Judge)。
    利用大模型判断生成的 prediction 中的事实是否完全被 contexts 支持。
    """
    context_str = "\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    eval_prompt = (
        "You are an impartial evaluator. Evaluate whether the information in the 'Statement' "
        "is completely supported by the 'Context'.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Statement: {prediction}\n\n"
        "Output exactly '1' if the statement is fully supported by the context, '0' if it contradicts or contains unverified hallucinations. "
        "No other text."
    )
    # 构造一条系统评估消息，借用原有的生成链路
    messages = [
        {"role": "system", "content": "You are a factual verification assistant."},
        {"role": "user", "content": eval_prompt}
    ]
    
    if generator.use_api:
        judge_res = generator._generate_api(messages, max_tokens=10, temperature=0.0)
    else:
        judge_res = generator._generate_local(messages, max_new_tokens=10, temperature=0.0)
        
    return 1 if "1" in str(judge_res) else 0


def evaluate_end_to_end(retriever, generator, dataset_name, top_k=3, method="hybrid", sample_size=100):
    """
    端到端生成评估：包含 Retrieval 和 Generation
    """
    print(f"\n🧠 开始执行端到端生成评估 (Dataset: {dataset_name} | Method: {method})")
    
    # 1. 构建测试集 (选取具备有效问题和真实答案的样本)
    test_data = []
    for item in retriever.meta:
        q = item.get("question") or item.get("original_question") or item.get("query")
        a = item.get("answer") or item.get("final_decision") or item.get("long_answer")
        if q and a:
            test_data.append({"query": q, "answer": a})
            
    # 去重
    unique_test_data = {item['query']: item for item in test_data}.values()
    all_queries = list(unique_test_data)
    
    if sample_size > 0 and sample_size < len(all_queries):
        test_queries = random.sample(all_queries, sample_size)
    else:
        test_queries = all_queries

    em_total, f1_total, fact_score_total = 0, 0, 0
    retriever.mock = True 
    
    # 2. 批量推理
    for item in tqdm(test_queries, desc="End-to-End Evaluating"):
        q = item["query"]
        ground_truth = item["answer"]
        
        # 检索上下文
        retrieved_results, _ = retriever.search(query=q, top_k=top_k, method=method)
        
        # 提取上下文文本列表。假设 meta_info 中有对应的文本。
        # 此处需要根据您的数据结构适配，假设 retriever.meta 中存了 'text' 或是从 dataset 取
        # 因为前文的 meta 没有保存原始 chunk text，实际场景中建议 retriever 返回时带上原文 text
        contexts = [res["meta_info"].get("text", "Context not found in meta") for res in retrieved_results]
        
        # 生成答案
        prediction = generator.generate(question=q, contexts=contexts)
        
        # 计算指标
        em_total += exact_match_score(prediction, ground_truth)
        f1_total += f1_score(prediction, ground_truth)
        
        # FActScore 判断
        if prediction.strip() and prediction.lower() != "insufficient evidence.":
            fact_score_total += calculate_fact_score_via_llm(prediction, contexts, generator)

    # 3. 统计结果
    num_samples = len(test_queries)
    em_avg = (em_total / num_samples) * 100
    f1_avg = (f1_total / num_samples) * 100
    fact_avg = (fact_score_total / num_samples) * 100
    
    print("\n" + "="*60)
    print(f"🏆 端到端生成评估报告")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔹 检索策略: {method.upper()} | Top-K: {top_k}")
    print(f"🔹 测试样本数: {num_samples}")
    print(f"🔹 Exact Match (EM): {em_avg:.2f}%")
    print(f"🔹 Token F1 Score:   {f1_avg:.2f}%")
    print(f"🔹 FActScore (Faithfulness): {fact_avg:.2f}%")
    print("="*60 + "\n")
    
    return {"EM": em_avg, "F1": f1_avg, "FActScore": fact_avg}


# ===================== 离线检索结果驱动的生成与评估 =====================

def _extract_ground_truth(item: Dict[str, Any]) -> str:
    """兼容不同字段命名，提取标准答案。"""
    return str(
        item.get("answer")
        or item.get("final_decision")
        or item.get("long_answer")
        or item.get("gold_answer")
        or ""
    ).strip()


def _extract_text_contexts_from_retrieval_results(
    retrieved_results: List[Dict[str, Any]],
    max_contexts: int = 5,
    max_chars_per_context: int = 1200,
) -> List[str]:
    contexts: List[str] = []
    for row in retrieved_results[:max_contexts]:
        meta_info = row.get("meta_info", {})
        txt = str(meta_info.get("text", "")).strip()
        if not txt:
            continue
        contexts.append(txt[:max_chars_per_context])
    return contexts


def load_retrieval_output_data(
    retrieval_file: str,
    sample_size: int = 0,
    shuffle: bool = False,
) -> List[Dict[str, Any]]:
    with open(retrieval_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        data = payload.get("items") or payload.get("records") or []
    elif isinstance(payload, list):
        data = payload
    else:
        raise ValueError("retrieval_file 需要是 JSON 数组或包含 items/records 的对象")

    valid = [x for x in data if str(x.get("question", "")).strip() and isinstance(x.get("retrieved_results", []), list)]

    if shuffle:
        random.shuffle(valid)
    if sample_size > 0:
        valid = valid[:sample_size]
    return valid


def run_generation_from_retrieval_output(
    generator,
    retrieval_file: str,
    sample_size: int = 0,
    output_dir: str = DEFAULT_GENERATION_OUTPUT_DIR,
    output_prefix: str = "retrieve_generation",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_contexts: int = 5,
    max_chars_per_context: int = 1200,
    do_factscore: bool = True,
    shuffle: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    data = load_retrieval_output_data(retrieval_file=retrieval_file, sample_size=sample_size, shuffle=shuffle)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = os.path.join(output_dir, f"{output_prefix}_predictions_{timestamp}.json")
    detailed_file = os.path.join(output_dir, f"{output_prefix}_detailed_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"{output_prefix}_summary_{timestamp}.json")

    records: List[Dict[str, Any]] = []
    em_total = 0
    f1_total = 0.0
    fact_total = 0
    eval_count = 0

    print(f"\n📦 已加载检索结果: {len(data)} 条 | 文件: {retrieval_file}")
    print("🤖 开始基于检索结果生成答案...")

    for item in tqdm(data, desc="Generate From Retrieval"):
        question = str(item.get("question", "")).strip()
        gold_answer = str(item.get("gold_answer") or item.get("answer") or "").strip()
        retrieved_results = item.get("retrieved_results", [])

        contexts = _extract_text_contexts_from_retrieval_results(
            retrieved_results=retrieved_results,
            max_contexts=max_contexts,
            max_chars_per_context=max_chars_per_context,
        )
        knowledge_used = "\n\n".join(contexts)

        pred = generator.generate(
            question=question,
            contexts=contexts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        em = None
        f1 = None
        if gold_answer:
            em = int(exact_match_score(pred, gold_answer))
            f1 = float(f1_score(pred, gold_answer))
            em_total += em
            f1_total += f1
            eval_count += 1

        fact = None
        if do_factscore and pred and pred.lower() != "insufficient evidence.":
            fact = int(calculate_fact_score_via_llm(pred, contexts if contexts else [knowledge_used], generator))
            fact_total += fact

        records.append(
            {
                "question": question,
                "answer": gold_answer,
                "generated_answer": pred,
                "em": em,
                "f1": f1,
                "fact_score": fact,
                "knowledge_used": knowledge_used,
                "retrieved_results": retrieved_results,
                "retrieval_meta": {
                    "rewritten_query": item.get("rewritten_query"),
                    "retrieval_method": item.get("retrieval_method"),
                    "top_k": item.get("top_k"),
                    "hit_at_k": item.get("hit_at_k"),
                    "recall_at_k": item.get("recall_at_k"),
                    "ndcg_at_k": item.get("ndcg_at_k"),
                    "mrr": item.get("mrr"),
                    "top1_answer": item.get("top1_answer"),
                },
            }
        )

    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary: Dict[str, Any] = {
        "retrieval_file": retrieval_file,
        "sample_count": len(records),
        "evaluated_count": eval_count,
        "predictions_file": pred_file,
        "detailed_file": detailed_file,
    }
    if eval_count > 0:
        summary["EM"] = (em_total / eval_count) * 100
        summary["F1"] = (f1_total / eval_count) * 100
    if do_factscore and records:
        summary["FActScore"] = (fact_total / len(records)) * 100

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump({"overall_metrics": summary, "items": records}, f, ensure_ascii=False, indent=2)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("🏆 基于检索结果的生成评估报告")
    print(f"🔹 样本数: {len(records)}")
    if "EM" in summary:
        print(f"🔹 EM: {summary['EM']:.2f}%")
        print(f"🔹 F1: {summary['F1']:.2f}%")
    if "FActScore" in summary:
        print(f"🔹 FActScore: {summary['FActScore']:.2f}%")
    print(f"🔹 结果文件: {pred_file}")
    print(f"🔹 摘要文件: {summary_file}")
    print("=" * 60 + "\n")

    return summary


def evaluate_generated_predictions_file(
    predictions_file: str,
    output_dir: str = DEFAULT_GENERATION_OUTPUT_DIR,
    output_prefix: str = "generated_eval",
) -> Dict[str, Any]:
    with open(predictions_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        records = payload.get("items") or payload.get("records") or []
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("predictions_file 需要是 JSON 数组或包含 items/records 的对象")

    eval_res = _evaluate_prediction_records(records, prediction_field="generated_answer")
    overall = eval_res["overall"]
    items = eval_res["items"]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_file = os.path.join(output_dir, f"{output_prefix}_detailed_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"{output_prefix}_summary_{timestamp}.json")

    summary = {
        "predictions_file": predictions_file,
        "EM": overall.get("EM", 0.0),
        "F1": overall.get("F1", 0.0),
        "Count": overall.get("Count", 0),
        "detailed_file": detailed_file,
    }

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump({"overall_metrics": summary, "items": items}, f, ensure_ascii=False, indent=2)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📊 已完成生成结果评估")
    print(f"🔹 Count: {summary['Count']}")
    print(f"🔹 EM: {summary['EM']:.2f}%")
    print(f"🔹 F1: {summary['F1']:.2f}%")
    print(f"🔹 摘要文件: {summary_file}")
    print("=" * 60 + "\n")

    return summary


def load_offline_qa_data(offline_file: str, sample_size: int = 0, shuffle: bool = False) -> List[Dict[str, Any]]:
    """加载离线检索结果文件（如 data/test_knowledge.json）。"""
    with open(offline_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_data = [
        x for x in data
        if x.get("question") and str(x.get("knowledge", "")).strip()
    ]

    if shuffle:
        random.shuffle(valid_data)

    if sample_size > 0:
        valid_data = valid_data[:sample_size]

    return valid_data


def _evaluate_prediction_records(
    records: List[Dict[str, Any]],
    prediction_field: str = "generated_answer",
) -> Dict[str, Any]:
    """对已包含预测答案的记录计算每题与整体 EM/F1。"""
    if not records:
        return {
            "overall": {"EM": 0.0, "F1": 0.0, "Count": 0},
            "items": [],
        }

    em_total = 0
    f1_total = 0.0
    valid_count = 0
    evaluated_items: List[Dict[str, Any]] = []

    for item in records:
        pred = str(item.get(prediction_field, "")).strip()
        gt = _extract_ground_truth(item)
        item_eval = dict(item)

        if not gt:
            item_eval.update(
                {
                    "em": None,
                    "f1": None,
                    "normalized_prediction": normalize_answer(pred),
                    "normalized_answer": "",
                    "has_ground_truth": False,
                }
            )
            evaluated_items.append(item_eval)
            continue

        valid_count += 1
        em = int(exact_match_score(pred, gt))
        f1 = f1_score(pred, gt)

        em_total += em
        f1_total += f1
        item_eval.update(
            {
                "em": em,
                "f1": f1,
                "normalized_prediction": normalize_answer(pred),
                "normalized_answer": normalize_answer(gt),
                "has_ground_truth": True,
            }
        )
        evaluated_items.append(item_eval)

    if valid_count == 0:
        return {
            "overall": {"EM": 0.0, "F1": 0.0, "Count": 0},
            "items": evaluated_items,
        }

    return {
        "overall": {
            "EM": (em_total / valid_count) * 100,
            "F1": (f1_total / valid_count) * 100,
            "Count": valid_count,
        },
        "items": evaluated_items,
    }


def run_offline_generation_and_evaluation(
    generator,
    offline_file: str,
    sample_size: int = 50,
    output_dir: str = DEFAULT_GENERATION_OUTPUT_DIR,
    output_prefix: str = "offline_test_knowledge",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_sources: int = 8,
    max_chars_per_source: int = 1200,
    run_generation: bool = True,
    run_evaluation: bool = True,
    do_factscore: bool = False,
    shuffle: bool = False,
) -> Dict[str, Any]:
    """
    基于离线 knowledge 数据执行：
    1) 生成（可单独执行）
    2) 评估（可单独执行，或接着生成结果评估）
    """
    os.makedirs(output_dir, exist_ok=True)
    data = load_offline_qa_data(offline_file=offline_file, sample_size=sample_size, shuffle=shuffle)

    print(f"\n📦 离线样本加载完成: {len(data)} 条 | 文件: {offline_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = os.path.join(output_dir, f"{output_prefix}_predictions_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"{output_prefix}_summary_{timestamp}.json")
    detailed_file = os.path.join(output_dir, f"{output_prefix}_detailed_{timestamp}.json")

    records: List[Dict[str, Any]] = []
    fact_total = 0
    generated_in_evaluate_mode = False

    if run_generation:
        print("\n🤖 开始离线知识驱动生成...")
        for item in tqdm(data, desc="Offline Generate"):
            question = str(item.get("question", "")).strip()
            knowledge = str(item.get("knowledge", "")).strip()
            ground_truth = _extract_ground_truth(item)

            prediction = generator.generate_from_knowledge(
                question=question,
                knowledge=knowledge,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            out = {
                "id": item.get("id"),
                "question": question,
                "answer": ground_truth,
                "generated_answer": prediction,
                "type": item.get("type"),
                "level": item.get("level"),
                "knowledge_used": knowledge,
            }

            if run_evaluation and do_factscore and prediction and prediction.lower() != "insufficient evidence.":
                fact = calculate_fact_score_via_llm(prediction, [knowledge], generator)
                out["fact_score"] = fact
                fact_total += fact

            records.append(out)

        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ 生成结果已保存: {pred_file}")
    else:
        # evaluate-only 模式：若输入文件没有 generated_answer，则自动补全生成
        missing_pred = 0
        for item in data:
            if not str(item.get("generated_answer", "")).strip():
                missing_pred += 1

        if missing_pred > 0 and generator is not None:
            print(f"\n⚠️ 检测到 {missing_pred} 条样本缺少 generated_answer，开始自动生成后再评估...")
            generated_in_evaluate_mode = True
            for item in tqdm(data, desc="Auto Generate For Evaluate"):
                question = str(item.get("question", "")).strip()
                knowledge = str(item.get("knowledge", "")).strip()
                ground_truth = _extract_ground_truth(item)

                prediction = str(item.get("generated_answer", "")).strip()
                if not prediction:
                    prediction = generator.generate_from_knowledge(
                        question=question,
                        knowledge=knowledge,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                out = dict(item)
                out.update(
                    {
                        "id": item.get("id"),
                        "question": question,
                        "answer": ground_truth,
                        "generated_answer": prediction,
                        "type": item.get("type"),
                        "level": item.get("level"),
                        "knowledge_used": knowledge,
                    }
                )
                records.append(out)

            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"✅ 自动补全生成结果已保存: {pred_file}")
        else:
            # 不运行生成时，默认把离线文件当作已含预测文件来评估
            for item in data:
                records.append(item)

    summary: Dict[str, Any] = {
        "offline_file": offline_file,
        "sample_count": len(records),
        "predictions_file": pred_file if (run_generation or generated_in_evaluate_mode) else offline_file,
        "detailed_file": detailed_file,
    }

    if run_evaluation:
        eval_res = _evaluate_prediction_records(records, prediction_field="generated_answer")
        metrics = eval_res["overall"]
        records_with_metrics = eval_res["items"]
        summary.update(metrics)

        if do_factscore and records:
            summary["FActScore"] = (fact_total / len(records)) * 100

        # 如已做生成，则把每题指标回写到预测文件
        if run_generation:
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(records_with_metrics, f, ensure_ascii=False, indent=2)

        detailed_payload = {
            "overall_metrics": {
                "EM": summary.get("EM", 0.0),
                "F1": summary.get("F1", 0.0),
                "Count": summary.get("Count", 0),
                "FActScore": summary.get("FActScore"),
            },
            "items": records_with_metrics,
        }
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(detailed_payload, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print("🏆 离线生成评估报告")
        print(f"🔹 样本数: {summary.get('Count', 0)}")
        print(f"🔹 Exact Match (EM): {summary.get('EM', 0.0):.2f}%")
        print(f"🔹 Token F1 Score:   {summary.get('F1', 0.0):.2f}%")
        if "FActScore" in summary:
            print(f"🔹 FActScore (Faithfulness): {summary['FActScore']:.2f}%")
        print("=" * 60 + "\n")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ 评估摘要已保存: {summary_file}")

    return summary


def _build_cli_args():
    parser = argparse.ArgumentParser(description="离线检索结果驱动的 RAG 生成与评估")
    parser.add_argument("--mode", type=str, default="both", choices=["generate", "evaluate", "both"], help="运行模式")
    parser.add_argument("--offline_file", type=str, default="data/test_knowledge.json", help="离线检索结果文件")
    parser.add_argument("--sample_size", type=int, default=50, help="样本数（0 表示全部）")
    parser.add_argument("--shuffle", action="store_true", help="是否先打乱样本")

    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    parser.add_argument("--output_prefix", type=str, default="offline_test_knowledge", help="输出文件前缀")

    # 生成参数
    parser.add_argument("--gen_model", type=str, default=None, help="生成模型（API 模式默认使用 config 中 provider 对应模型）")
    parser.add_argument("--use_api", action="store_true", help="是否通过 API 调用模型（默认会自动启用 API）")
    parser.add_argument("--use_local", action="store_true", help="强制使用本地模型，不走 API")
    parser.add_argument("--api_provider", type=str, default=None, help="API 提供商名称，优先读取 config.API_CONFIG")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--api_base_url", type=str, default=None, help="API Base URL")
    parser.add_argument("--max_tokens", type=int, default=128, help="最大生成 token")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--use_4bit", action="store_true", help="本地加载时启用 4bit 量化")

    # knowledge 解析参数
    parser.add_argument("--max_sources", type=int, default=8, help="从 knowledge 中提取的最大来源条数")
    parser.add_argument("--max_chars_per_source", type=int, default=1200, help="每条来源最大字符数")

    # 评估参数
    parser.add_argument("--do_factscore", action="store_true", help="是否额外计算 LLM Judge 的 FactScore")

    return parser.parse_args()


if __name__ == "__main__":
    args = _build_cli_args()

    from generator import RAGGenerator
    from config import API_CONFIG, DEFAULT_API_PROVIDER, DEFAULT_GEN_MODEL

    need_generator = args.mode in {"generate", "both", "evaluate"}
    generator = None
    if need_generator:
        # 默认优先走 config 里的 API；仅在显式 --use_local 时使用本地模型
        resolved_use_api = (not args.use_local) or args.use_api

        resolved_model_name = args.gen_model
        resolved_api_key = args.api_key
        resolved_api_base_url = args.api_base_url

        if resolved_use_api:
            provider = args.api_provider or DEFAULT_API_PROVIDER
            if provider not in API_CONFIG:
                raise ValueError(f"未在 config.API_CONFIG 中找到提供商: {provider}")
            cfg = API_CONFIG[provider]
            resolved_model_name = resolved_model_name or cfg.get("model") or DEFAULT_GEN_MODEL
            resolved_api_key = resolved_api_key or cfg.get("api_key")
            resolved_api_base_url = resolved_api_base_url or cfg.get("base_url")
            print(f"[*] 使用 config API 提供商: {provider} | 模型: {resolved_model_name}")
        else:
            resolved_model_name = resolved_model_name or DEFAULT_GEN_MODEL

        generator = RAGGenerator(
            model_name=resolved_model_name,
            use_api=resolved_use_api,
            api_key=resolved_api_key,
            api_base_url=resolved_api_base_url,
            use_4bit=args.use_4bit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    if args.mode in {"generate", "both"}:
        run_offline_generation_and_evaluation(
            generator=generator,
            offline_file=args.offline_file,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_sources=args.max_sources,
            max_chars_per_source=args.max_chars_per_source,
            run_generation=True,
            run_evaluation=(args.mode == "both"),
            do_factscore=args.do_factscore,
            shuffle=args.shuffle,
        )
    else:
        # evaluate-only: 若无 generated_answer，则会自动先生成再评估
        summary = run_offline_generation_and_evaluation(
            generator=generator,
            offline_file=args.offline_file,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            run_generation=False,
            run_evaluation=True,
            do_factscore=False,
            shuffle=args.shuffle,
        )
        print("\n📌 evaluate-only 结束。")
        print(json.dumps(summary, ensure_ascii=False, indent=2))