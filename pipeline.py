import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from utils import get_embeddings
from config import DEVICE

def stage1_chunking(docs, meta, dataset_name, model_name, chunk_size, chunk_overlap, paths):
    print(f"\n🚀 [阶段1] 分块处理: {dataset_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(tokenizer.encode(x, truncation=False)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks, all_meta = [], []
    for i, doc in enumerate(tqdm(docs, desc="Token分块")):
        chunks = splitter.split_text(doc)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_meta = meta[i].copy()
            chunk_meta["text"] = chunk  # 这里的 "text" 供 Generator 使用
            all_meta.append(chunk_meta)

    with open(paths["chunks"], "wb") as f:
        pickle.dump(all_chunks, f)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 阶段1完成 | 块数: {len(all_chunks)} | 保存至: {paths['chunks']}")


def stage2_embedding(model_name, batch_size, chunk_size, paths):
    print("\n🚀 [阶段2] 生成 Embedding")
    with open(paths["chunks"], "rb") as f:
        chunks = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="向量化"):
        batch = chunks[i : i + batch_size]
        emb = get_embeddings(batch, tokenizer, model, DEVICE, chunk_size)
        embeddings.append(emb)
    
    embeddings = np.vstack(embeddings).astype("float32")
    np.save(paths["embeddings"], embeddings)
    print(f"✅ 阶段2完成 | 维度: {embeddings.shape} | 保存至: {paths['embeddings']}")


def stage3_indexing(paths):
    print("\n🚀 [阶段3] 构建 BM25 与 FAISS 索引")
    with open(paths["chunks"], "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(paths["embeddings"])

    # BM25
    bm25 = BM25Okapi([c.split() for c in chunks])
    with open(paths["bm25"], "wb") as f:
        pickle.dump(bm25, f)

    # FAISS
    dense_index = faiss.IndexFlatIP(embeddings.shape[1])
    dense_index.add(embeddings)
    faiss.write_index(dense_index, paths["faiss"])

    print("✅ 阶段3完成 | 索引构建完毕。")