# retriever.py 完整修改
import json
import pickle
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from utils import get_embeddings, rewrite_query
from config import DEVICE

class RAGRetriever:
    def __init__(self, paths, embed_model_name, qgen_model_name, chunk_size, mock=True):
        self.mock = mock
        self.chunk_size = chunk_size
        
        # Load Models
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.model = AutoModel.from_pretrained(embed_model_name).to(DEVICE)
        self.q_tokenizer = AutoTokenizer.from_pretrained(qgen_model_name)
        self.q_model = T5ForConditionalGeneration.from_pretrained(qgen_model_name).to(DEVICE)
        
        # Load Indexes
        with open(paths["bm25"], "rb") as f:
            self.bm25 = pickle.load(f)
        self.dense = faiss.read_index(paths["faiss"])
        with open(paths["meta"], "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def search(self, query, top_k=3, method="hybrid"):
        """
        支持三种模式: 'sparse' (BM25), 'dense' (Contriever), 'hybrid' (混合)
        """
        # 查询重写
        query = rewrite_query(query, self.q_tokenizer, self.q_model, DEVICE, self.mock)
        
        res, seen = [], set()
        candidates = []

        # 1. 稀疏检索 (Sparse - BM25)
        if method in ["sparse", "hybrid"]:
            bm_scores = self.bm25.get_scores(query.split())
            bm_top = np.argsort(bm_scores)[::-1][:top_k if method == "sparse" else top_k*2]
            if method == "sparse":
                candidates = list(bm_top)

        # 2. 稠密检索 (Dense - Contriever)
        if method in ["dense", "hybrid"]:
            q_emb = get_embeddings([query], self.tokenizer, self.model, DEVICE, self.chunk_size)
            _, dense_top = self.dense.search(q_emb, top_k if method == "dense" else top_k*2)
            if method == "dense":
                candidates = list(dense_top[0])

        # 3. 混合检索 (Hybrid - 交替合并)
        if method == "hybrid":
            # 简单的交替合并策略 (Round-Robin)
            for b, d in zip(list(bm_top), list(dense_top[0])):
                candidates.extend([b, d])

        # 获取元数据，去重并截断
        for idx in candidates:
            if len(res) >= top_k:
                break
            if idx in seen:
                continue
            seen.add(idx)
            res.append({
                "排名": len(res) + 1, 
                "数据集": self.meta[idx]["dataset"], 
                "块ID": int(idx),
                "meta_info": self.meta[idx] # 保留完整 meta 用于后续评估
            })
            
        return res