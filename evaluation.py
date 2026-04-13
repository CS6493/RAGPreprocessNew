import random
from tqdm import tqdm

from generation_utils import exact_match_score, token_f1_score as f1_score


def evaluate_retrieval(retriever, dataset_name, top_k=3, method="hybrid", sample_size=100):
    print(f"\n📊 开始执行评估 (Dataset: {dataset_name} | Method: {method} | Top-K: {top_k})")

    unique_queries = {}
    for item in retriever.meta:
        q = item.get("question") or item.get("original_question") or item.get("query")
        a = item.get("answer", item.get("final_decision", item.get("long_answer", item)))
        doc_id = item.get("source_id", item.get("financebench_id", item.get("hotpot_id", item.get("pubid", a))))

        if q and doc_id and q not in unique_queries:
            unique_queries[q] = {"doc_id": doc_id, "answer": a}

    all_queries = list(unique_queries.keys())
    if not all_queries:
        print("❌ 无法从 meta 数据中提取出有效的 Question-Answer 对，请检查 data_loader.py 中 meta 的构建。")
        return

    test_queries = random.sample(all_queries, sample_size) if 0 < sample_size < len(all_queries) else all_queries
    hit_count = 0

    for q in tqdm(test_queries, desc="Evaluating"):
        ground_truth_id = unique_queries[q]["doc_id"]
        retriever.mock = True
        results, _ = retriever.search(query=q, top_k=top_k, method=method)

        is_hit = False
        for res in results:
            retrieved_meta = res["meta_info"]
            retrieved_id = retrieved_meta.get("source_id", retrieved_meta.get("financebench_id", retrieved_meta.get("hotpot_id", retrieved_meta.get("pubid", retrieved_meta.get("answer")))))
            if retrieved_id == ground_truth_id:
                is_hit = True
                break
        if is_hit:
            hit_count += 1

    recall = (hit_count / len(test_queries)) * 100
    print("\n" + "=" * 50)
    print("📈 评估结果报告")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔹 检索策略: {method.upper()}")
    print(f"🔹 测试样本数: {len(test_queries)}")
    print(f"🔹 Recall@{top_k}: {recall:.2f}% ({hit_count}/{len(test_queries)})")
    print("=" * 50 + "\n")
    return recall


def calculate_fact_score_via_llm(prediction, contexts, generator):
    context_str = "\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    eval_prompt = (
        "You are an impartial evaluator. Evaluate whether the information in the 'Statement' "
        "is completely supported by the 'Context'.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Statement: {prediction}\n\n"
        "Output exactly '1' if the statement is fully supported by the context, '0' if it contradicts or contains unverified hallucinations. "
        "No other text."
    )
    judge = generator.generate(
        question="Is the statement fully supported by the provided context?",
        contexts=[eval_prompt],
        return_debug=True,
    )
    return 1 if "1" in str(judge["prediction"]) else 0


def evaluate_end_to_end(retriever, generator, dataset_name, top_k=3, method="hybrid", sample_size=100):
    print(f"\n🧠 开始执行端到端生成评估 (Dataset: {dataset_name} | Method: {method})")

    test_data = []
    for item in retriever.meta:
        q = item.get("question") or item.get("original_question") or item.get("query")
        a = item.get("answer") or item.get("final_decision") or item.get("long_answer")
        if q and a:
            test_data.append({"query": q, "answer": a})

    unique_test_data = {item["query"]: item for item in test_data}.values()
    all_queries = list(unique_test_data)
    test_queries = random.sample(all_queries, sample_size) if 0 < sample_size < len(all_queries) else all_queries

    em_total, f1_total, fact_total = 0, 0, 0
    retriever.mock = True

    for item in tqdm(test_queries, desc="End-to-End Evaluating"):
        q = item["query"]
        ground_truth = item["answer"]

        retrieved_results, _ = retriever.search(query=q, top_k=top_k, method=method)
        contexts = [res["meta_info"].get("text", "Context not found in meta") for res in retrieved_results]

        out = generator.generate(question=q, contexts=contexts, return_debug=True)
        prediction = out["prediction"]

        em_total += exact_match_score(prediction, ground_truth)
        f1_total += f1_score(prediction, ground_truth)

        if prediction.strip() and prediction.lower() != "insufficient evidence.":
            fact_total += calculate_fact_score_via_llm(prediction, contexts, generator)

    num_samples = len(test_queries)
    em_avg = (em_total / num_samples) * 100 if num_samples else 0
    f1_avg = (f1_total / num_samples) * 100 if num_samples else 0
    fact_avg = (fact_total / num_samples) * 100 if num_samples else 0

    print("\n" + "=" * 60)
    print("🏆 端到端生成评估报告")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔹 检索策略: {method.upper()} | Top-K: {top_k}")
    print(f"🔹 测试样本数: {num_samples}")
    print(f"🔹 Exact Match (EM): {em_avg:.2f}%")
    print(f"🔹 Token F1 Score:   {f1_avg:.2f}%")
    print(f"🔹 FActScore (Faithfulness): {fact_avg:.2f}%")
    print("=" * 60 + "\n")

    return {"EM": em_avg, "F1": f1_avg, "FActScore": fact_avg}
