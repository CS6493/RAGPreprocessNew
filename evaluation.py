import random
import re
import string
from collections import Counter
from tqdm import tqdm

def evaluate_retrieval(retriever, dataset_name, top_k=3, method="hybrid", sample_size=100):
    """
    评估检索系统的 Recall@K
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
    
    # 3. 执行批量检索并统计召回率
    for q in tqdm(test_queries, desc="Evaluating"):
        ground_truth_id = unique_queries[q]["doc_id"]
        
        # 禁用 query 重写的 mock 输出打印，以免打乱终端显示
        retriever.mock = True 
        
        results, _ = retriever.search(query=q, top_k=top_k, method=method)
        
        # 判断 Ground Truth 是否在召回的 Top-K 块中
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
    print("\n" + "="*50)
    print(f"📈 评估结果报告")
    print(f"🔹 数据集: {dataset_name}")
    print(f"🔹 检索策略: {method.upper()}")
    print(f"🔹 测试样本数: {len(test_queries)}")
    print(f"🔹 Recall@{top_k}: {recall:.2f}% ({hit_count}/{len(test_queries)})")
    print("="*50 + "\n")
    
    return recall


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
        judge_res = generator._generate_via_api(messages, max_tokens=10, temperature=0.0)
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