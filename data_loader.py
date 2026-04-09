from datasets import load_dataset
from tqdm import tqdm
from utils import clean_text
from config import DATASETS_CONFIG

def load_and_preprocess_dataset(dataset_name, max_samples=1000):
    if dataset_name not in DATASETS_CONFIG:
        raise ValueError(f"未找到数据集 {dataset_name} 的配置。")
    
    cfg = DATASETS_CONFIG[dataset_name]
    load_kwargs = {"path": cfg["path"], "split": cfg["split"]}
    if "name" in cfg:
        load_kwargs["name"] = cfg["name"]
        
    print(f"📥 正在加载数据集: {dataset_name} ({load_kwargs})")
    ds = load_dataset(**load_kwargs)
    
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
        
    docs = []
    meta = []
    desc = f"清洗 {dataset_name}"
    
    if dataset_name == "HotpotQA":
        for item in tqdm(ds, desc=desc):
            # 将 title 和 sentences 对应拼接起来，并在句子之间正确加入空格
            ctx_parts = []
            titles = item["context"].get("title", [])
            sentences_list = item["context"].get("sentences", [])
            
            for t, sents in zip(titles, sentences_list):
                paragraph = " ".join(sents) # 用空格把一个段落的多个句子拼起来
                ctx_parts.append(f"Document Title: {t}\n{paragraph}")
            
            ctx = "\n\n".join(ctx_parts) # 不同文档之间用换行隔开
            docs.append(clean_text(ctx))
            
            meta.append({
                "dataset": dataset_name,
                "hotpot_id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "type": item.get("type", ""),
                "level": item.get("level", ""),
                "source_id": item.get("id", ""),
            })
            
    elif dataset_name == "PubMedQA":
        for item in tqdm(ds, desc=desc):
            ctx_data = item.get("context", {})
            if isinstance(ctx_data, dict) and "contexts" in ctx_data:
                contexts = ctx_data.get("contexts", [])
                labels = ctx_data.get("labels", [])
                
                # 如果 labels 和 contexts 数量对齐，则带上标签进行格式化拼接
                if len(contexts) == len(labels) and len(contexts) > 0:
                    ctx_parts = [f"{label}: {text}" for label, text in zip(labels, contexts)]
                    ctx = "\n\n".join(ctx_parts)
                else:
                    # 兜底：直接用换行拼接
                    ctx = "\n\n".join(contexts)
            else:
                ctx = str(ctx_data)
                
            docs.append(clean_text(ctx))
            
            meta.append({
                "dataset": dataset_name,
                "pubid": str(item.get("pubid", "unknown")),
                "question": item.get("question", ""),
                "answer": item.get("final_decision", ""),
                "long_answer": item.get("long_answer", ""),     # 详细的医学解释
                "final_decision": item.get("final_decision", ""), # 简答结论 (yes/no/maybe)
                "meshes": ctx_data.get("meshes", []),            # 医学主题词列表，极好的检索过滤条件
                "source_id": item.get("pubid", ""),
            })
            
    elif dataset_name == "FinanceBench":
        for item in tqdm(ds, desc=desc):
            evidence_list = item.get("evidence", [])
            evidence_texts = []
            # 遍历解析 evidence 列表中的字典结构
            if isinstance(evidence_list, list) and len(evidence_list) > 0:
                for ev in evidence_list:
                    # 优先取精准的 evidence_text，也可根据需要改为 evidence_text_full_page
                    text = ev.get("evidence_text", "").strip()
                    if text:
                        evidence_texts.append(text)
            # 将多段 evidence 组合起来
            evidence_str = "\n\n".join(evidence_texts)
            
            # 如果极端情况下没有 evidence，兜底使用问答拼接
            if not evidence_str.strip():
                evidence_str = f"Question: {item.get('question','')}\nAnswer: {item.get('answer','')}"
                
            docs.append(clean_text(evidence_str))
            
            meta.append({
                "dataset": dataset_name,
                "financebench_id": item.get("financebench_id", "unknown"),
                "company": item.get("company", "unknown"),
                "doc_name": item.get("doc_name", "unknown"),
                "doc_type": item.get("doc_type", "unknown"),     # 如 10k, 10q
                "doc_period": str(item.get("doc_period", "")),   # 年份，如 2018
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "source_id": item.get("financebench_id", "unknown"),
                "doc_link": item.get("doc_link", "unknown")
            })
            
    elif dataset_name == "Natural_Questions":
        for item in tqdm(ds, desc=desc):
            txt = f"{item.get('query', '')} {item.get('answer', '')}"
            docs.append(clean_text(txt))
            meta.append({
                "dataset": dataset_name,
                "query": item.get("query", "unknown"),
                "question": item.get("query", ""),   
                "answer": item.get("answer", ""),
                "source_id": "nq_" + str(hash(item.get("query",""))) # NQ通常无ID，生成一个
            })
            
    return docs, meta