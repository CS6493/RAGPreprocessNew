# retriever.py 优化建议

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

        # 统一装载 Meta 信息
        for cand in final_candidates:
            cand["meta_info"] = self.meta[cand["chunk_id"]]

        return final_candidates, rewritten_q