import json
import pickle
import os
import random
from datetime import datetime

import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

from utils import get_embeddings, rewrite_query
from evaluation import exact_match_score, f1_score
from config import (
    DATASETS_CONFIG,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QGEN_MODEL,
    DEFAULT_RETRIEVE_OUTPUT_DIR,
    DEFAULT_RETRIEVE_QUERY_FILE,
    DEVICE,
    get_file_paths,
)

class RAGRetriever:
    def __init__(self, paths, embed_model_name, qgen_model_name, chunk_size, mock=True):
        self.mock = mock
        self.chunk_size = chunk_size
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.model = AutoModel.from_pretrained(embed_model_name).to(DEVICE)
        self.q_tokenizer = AutoTokenizer.from_pretrained(qgen_model_name)
        self.q_model = T5ForConditionalGeneration.from_pretrained(qgen_model_name).to(DEVICE)
        
        # 加载索引
        with open(paths["bm25"], "rb") as f:
            self.bm25 = pickle.load(f)
        self.dense = faiss.read_index(paths["faiss"])
        with open(paths["meta"], "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def _sparse_search(self, query, top_k):
        """执行 BM25 稀疏检索，返回 [(chunk_id, score), ...]"""
        bm_scores = self.bm25.get_scores(query.split())
        # 获取 top_k 索引并提取对应得分
        top_indices = np.argsort(bm_scores)[::-1][:top_k]
        return [(int(idx), float(bm_scores[idx])) for idx in top_indices]

    def _dense_search(self, query, top_k):
        """执行 Faiss 稠密检索，返回 [(chunk_id, distance), ...]"""
        q_emb = get_embeddings([query], self.tokenizer, self.model, DEVICE, self.chunk_size)
        distances, indices = self.dense.search(q_emb, top_k)
        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    def _compute_rrf(self, sparse_results, dense_results, k=60):
        """
        RRF 倒数排序融合算法
        RRF_score = 1 / (k + rank)
        """
        rrf_scores = {}
        
        # 处理稀疏结果
        for rank, (chunk_id, _) in enumerate(sparse_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            
        # 处理稠密结果
        for rank, (chunk_id, _) in enumerate(dense_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            
        # 根据 RRF 得分降序排序
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_rrf

    def search(self, query, top_k=3, method="hybrid"):
        """主检索接口，返回结构化且带有分数的候选列表"""
        rewritten_q = rewrite_query(query, self.q_tokenizer, self.q_model, DEVICE, self.mock)
        if not self.mock:
            print(f"\n🔍 重写后查询：{rewritten_q}")
        else:
            rewritten_q = query # Mock 模式下使用原句

        # 获取更深一层的候选池 (top_k * 3) 以保证融合后质量
        pool_size = top_k * 3 
        sparse_res = self._sparse_search(rewritten_q, pool_size) if method in ["sparse", "hybrid"] else []
        dense_res = self._dense_search(rewritten_q, pool_size) if method in ["dense", "hybrid"] else []

        final_candidates = []

        if method == "sparse":
            for rank, (idx, score) in enumerate(sparse_res[:top_k]):
                final_candidates.append({"chunk_id": idx, "rank": rank+1, "method": method, "bm25_score": score})
        
        elif method == "dense":
            for rank, (idx, dist) in enumerate(dense_res[:top_k]):
                final_candidates.append({"chunk_id": idx, "rank": rank+1, "method": method, "l2_distance": dist})
        
        elif method == "hybrid":
            rrf_sorted = self._compute_rrf(sparse_res, dense_res)
            # 方便溯源，将原始得分映射为字典
            sparse_dict = dict(sparse_res)
            dense_dict = dict(dense_res)
            
            for rank, (idx, rrf_score) in enumerate(rrf_sorted[:top_k]):
                final_candidates.append({
                    "chunk_id": idx,
                    "rank": rank+1,
                    "method": method,
                    "rrf_score": rrf_score,
                    "bm25_score": sparse_dict.get(idx, None),
                    "l2_distance": dense_dict.get(idx, None)
                })
            
        for candidate in final_candidates:
            idx = int(candidate["chunk_id"]) 
            candidate["meta_info"] = self.meta[idx]
            candidate["排名"] = candidate["rank"]
            candidate["块ID"] = candidate["chunk_id"]
            candidate["数据集"] = self.meta[idx].get("dataset", "unknown")

        return final_candidates, rewritten_q


def _normalize_question(text):
    return " ".join(str(text or "").strip().lower().split())


def _extract_answer(meta_info):
    return (
        meta_info.get("answer")
        or meta_info.get("final_decision")
        or meta_info.get("long_answer")
        or ""
    )


def _build_gold_by_question(meta_list):
    table = {}
    for item in meta_list:
        q = item.get("question") or item.get("query") or item.get("original_question")
        if not q:
            continue
        norm_q = _normalize_question(q)
        if norm_q in table:
            continue
        table[norm_q] = {
            "question": q,
            "answer": _extract_answer(item),
            "source_id": item.get("source_id", item.get("hotpot_id", item.get("pubid", item.get("financebench_id")))),
        }
    return table


def run_batch_retrieval(
    retriever,
    query_file,
    output_dir=DEFAULT_RETRIEVE_OUTPUT_DIR,
    top_k=3,
    method="hybrid",
    sample_size=0,
    shuffle=False,
):
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"找不到查询文件: {query_file}")

    with open(query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        raise ValueError("query_file 需要是 JSON 数组，元素至少包含 query 或 question 字段")

    normalized_gold = _build_gold_by_question(retriever.meta)

    if shuffle:
        random.shuffle(queries)
    if sample_size and sample_size > 0:
        queries = queries[:sample_size]

    records = []
    evaluated_count = 0
    hit_total = 0
    em_total = 0
    f1_total = 0.0

    retriever.mock = True
    for item in queries:
        original_query = str(item.get("query") or item.get("question") or "").strip()
        if not original_query:
            continue

        results, rewritten_q = retriever.search(original_query, top_k=top_k, method=method)

        norm_q = _normalize_question(original_query)
        gold = normalized_gold.get(norm_q, {})
        gold_answer = str(item.get("answer") or gold.get("answer") or "").strip()
        gold_source_id = str(item.get("source_id") or gold.get("source_id") or "").strip()

        top1_answer = ""
        if results:
            top1_answer = str(_extract_answer(results[0].get("meta_info", {}))).strip()

        hit_at_k = None
        if gold_source_id:
            hit_at_k = int(any(str(r.get("meta_info", {}).get("source_id", "")).strip() == gold_source_id for r in results))
            hit_total += hit_at_k

        top1_em = None
        top1_f1 = None
        if gold_answer:
            top1_em = int(exact_match_score(top1_answer, gold_answer))
            top1_f1 = float(f1_score(top1_answer, gold_answer))
            em_total += top1_em
            f1_total += top1_f1
            evaluated_count += 1

        record = {
            "question": original_query,
            "rewritten_query": rewritten_q,
            "gold_answer": gold_answer,
            "gold_source_id": gold_source_id,
            "top1_answer": top1_answer,
            "top1_em": top1_em,
            "top1_f1": top1_f1,
            "hit_at_k": hit_at_k,
            "top_k": top_k,
            "retrieval_method": method,
            "retrieved_results": results,
        }
        records.append(record)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_file = os.path.join(output_dir, f"retrieval_detailed_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"retrieval_summary_{timestamp}.json")

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary = {
        "query_file": query_file,
        "records_count": len(records),
        "evaluated_count": evaluated_count,
        "top_k": top_k,
        "retrieval_method": method,
        "detailed_file": detailed_file,
    }
    if evaluated_count > 0:
        summary["Top1_EM"] = (em_total / evaluated_count) * 100
        summary["Top1_F1"] = (f1_total / evaluated_count) * 100

    hit_eval_count = sum(1 for x in records if x.get("hit_at_k") is not None)
    if hit_eval_count > 0:
        summary["Hit@K"] = (hit_total / hit_eval_count) * 100
        summary["hit_evaluated_count"] = hit_eval_count

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📦 检索批量任务完成")
    print(f"🔹 总问题数: {len(records)}")
    if "Top1_EM" in summary:
        print(f"🔹 Top1 EM: {summary['Top1_EM']:.2f}%")
        print(f"🔹 Top1 F1: {summary['Top1_F1']:.2f}%")
    if "Hit@K" in summary:
        print(f"🔹 Hit@{top_k}: {summary['Hit@K']:.2f}%")
    print(f"🔹 明细文件: {detailed_file}")
    print(f"🔹 摘要文件: {summary_file}")
    print("=" * 60 + "\n")

    return summary


def _build_cli_args():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 检索模块单独执行")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS_CONFIG.keys()))
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--qgen_model", type=str, default=DEFAULT_QGEN_MODEL)

    parser.add_argument("--query_file", type=str, default=DEFAULT_RETRIEVE_QUERY_FILE)
    parser.add_argument("--retrieve_output_dir", type=str, default=DEFAULT_RETRIEVE_OUTPUT_DIR)
    parser.add_argument("--retrieval_method", type=str, default="hybrid", choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _build_cli_args()
    split = DATASETS_CONFIG[args.dataset]["split"]
    paths = get_file_paths(args.output_dir, args.dataset, split, args.chunk_size, args.chunk_overlap)

    retriever = RAGRetriever(
        paths=paths,
        embed_model_name=args.model_name,
        qgen_model_name=args.qgen_model,
        chunk_size=args.chunk_size,
        mock=True,
    )

    run_batch_retrieval(
        retriever=retriever,
        query_file=args.query_file,
        output_dir=args.retrieve_output_dir,
        top_k=args.top_k,
        method=args.retrieval_method,
        sample_size=args.sample_size,
        shuffle=args.shuffle,
    )