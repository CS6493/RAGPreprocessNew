import random
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
        a = item.get("answer") or item.get("final_decision") or item.get("long_answer")
        # 提取能够标识该文档唯一性的 ID
        doc_id = item.get("hotpot_id") or item.get("pubid") or item.get("financebench_id") or a
        
        if q and doc_id and q not in unique_queries:
            unique_queries[q] = doc_id

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
        ground_truth_id = unique_queries[q]
        
        # 禁用 query 重写的 mock 输出打印，以免打乱终端显示
        retriever.mock = True 
        
        results = retriever.search(query=q, top_k=top_k, method=method)
        
        # 判断 Ground Truth 是否在召回的 Top-K 块中
        is_hit = False
        for res in results:
            retrieved_meta = res["meta_info"]
            retrieved_id = retrieved_meta.get("hotpot_id") or retrieved_meta.get("pubid") or retrieved_meta.get("financebench_id") or retrieved_meta.get("answer")
            
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