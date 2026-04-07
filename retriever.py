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

    def search(self, query, top_k=3):
        query = rewrite_query(query, self.q_tokenizer, self.q_model, DEVICE, self.mock)
        print(f"\n🔍 最终查询：{query}")
        
        bm_scores = self.bm25.get_scores(query.split())
        bm_top = np.argsort(bm_scores)[::-1][:top_k*3]
        
        q_emb = get_embeddings([query], self.tokenizer, self.model, DEVICE, self.chunk_size)
        _, dense_top = self.dense.search(q_emb, top_k*3)

        res, seen = [], set()
        for idx in list(bm_top) + list(dense_top[0]):
            if len(res) >= top_k or idx in seen: 
                continue
            seen.add(idx)
            res.append({
                "排名": len(res) + 1, 
                "数据集": self.meta[idx]["dataset"], 
                "块ID": int(idx)
            })
        return res