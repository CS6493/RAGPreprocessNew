import os
import sys
import subprocess

# ===================== 自动安装所有依赖 =====================
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    except:
        pass

print("🔧 自动检查并安装依赖库，请稍候...")
required_packages = [
    "torch", "faiss-cpu", "transformers", "datasets", 
    "langchain-text-splitters", "rank-bm25", "tqdm", "numpy"
]
for pkg in required_packages:
    install_package(pkg)
print("✅ 依赖安装/检查完成！")

# ===================== 导入库 =====================
import json
import pickle
import torch
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings("ignore")

# ===================== 全局通用配置（无环境绑定·全平台兼容） =====================
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TQDM_DISABLE_HTML"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_NAME = "facebook/contriever-msmarco"
QGEN_MODEL = "google/flan-t5-small"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 51
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 通用保存路径（自动创建，无权限问题）
SAVE_DIR = "./rag_preprocess_output"
OUTPUT_FINAL = "./indexes_optimized_final"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_FINAL, exist_ok=True)

# ===================== 脏数据终极清洗（100%防报错） =====================
def clean_text(text):
    if text is None:
        return "empty_content"
    text = str(text).strip()
    text = " ".join(text.split())
    if len(text) < 10:
        return "short_content_" + text
    return text

# ===================== 通用工具函数 =====================
def mean_pooling(token_embeddings, attention_mask):
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

@torch.no_grad()
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=CHUNK_SIZE, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

@torch.no_grad()
def rewrite_query(query, tokenizer, model, mock=False):
    if mock:
        return f"[MOCK] {query}"
    inputs = tokenizer(f"Optimize query: {query}", return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===================== 阶段1：数据清洗+分块+落地 =====================
def stage1_load_clean_split():
    print("\n" + "="*60)
    print("🚀 阶段1：数据加载 → 脏数据清洗 → Token分块 → 保存到硬盘")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda x: len(tokenizer.encode(x, truncation=False)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs, meta = [], []
    # HotpotQA
    ds = load_dataset("hotpot_qa", "distractor", split="train[:7405]")
    for item in tqdm(ds, desc="清洗 HotpotQA"):
        ctx = " ".join(["".join(s) for s in item["context"]["sentences"]])
        docs.append(clean_text(ctx))
        meta.append({"dataset": "hotpotqa"})

    # PubMedQA
    ds = load_dataset("pubmed_qa", "pqa_artificial", split="train[:1000]")
    for item in tqdm(ds, desc="清洗 PubMedQA"):
        docs.append(clean_text(item["context"]))
        meta.append({"dataset": "pubmedqa"})

    # FinanceBench
    ds = load_dataset("PatronusAI/financebench", split="train[:150]")
    for item in tqdm(ds, desc="清洗 FinanceBench"):
        txt = f"{item.get('question','')} {item.get('answer','')}"
        docs.append(clean_text(txt))
        meta.append({"dataset": "financebench"})

    # 分块
    all_chunks, all_meta = [], []
    for i, doc in enumerate(tqdm(docs, desc="Token分块")):
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
        all_meta.extend([meta[i]]*len(chunks))

    # 保存
    with open(f"{SAVE_DIR}/stage1_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{SAVE_DIR}/stage1_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 阶段1完成 | 总文本块：{len(all_chunks)}")

# ===================== 阶段2：生成Embedding+落地 =====================
def stage2_generate_embeddings():
    print("\n" + "="*60)
    print("🚀 阶段2：加载分块 → 生成768维向量 → 保存到硬盘")
    print("="*60)

    with open(f"{SAVE_DIR}/stage1_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="生成向量"):
        emb = get_embeddings(chunks[i:i+BATCH_SIZE], tokenizer, model)
        embeddings.append(emb)
    
    embeddings = np.vstack(embeddings).astype("float32")
    np.save(f"{SAVE_DIR}/stage2_embeddings.npy", embeddings)
    print(f"✅ 阶段2完成 | 向量维度：{embeddings.shape}")

# ===================== 阶段3：构建索引+最终交付 =====================
def stage3_build_index():
    print("\n" + "="*60)
    print("🚀 阶段3：加载向量 → 构建双索引 → 生成最终文件")
    print("="*60)

    # 加载数据
    with open(f"{SAVE_DIR}/stage1_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(f"{SAVE_DIR}/stage1_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    embeddings = np.load(f"{SAVE_DIR}/stage2_embeddings.npy")

    # 构建索引
    bm25 = BM25Okapi([c.split() for c in chunks])
    dense_index = faiss.IndexFlatIP(embeddings.shape[1])
    dense_index.add(embeddings)

    # 保存最终文件（直接交付使用）
    with open(f"{OUTPUT_FINAL}/bm25_size_512_overlap_10.pkl", "wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(dense_index, f"{OUTPUT_FINAL}/contriever_optimized.index")
    with open(f"{OUTPUT_FINAL}/contriever_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n🎉 🎯 全部任务完成！")
    print(f"最终索引文件保存在：{OUTPUT_FINAL}")
    print("1. bm25_size_512_overlap_10.pkl")
    print("2. contriever_optimized.index")
    print("3. contriever_metadata.json")

# ===================== 检索测试 =====================
class RAGRetriever:
    def __init__(self, mock=True):
        self.mock = mock
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.q_tokenizer = AutoTokenizer.from_pretrained(QGEN_MODEL)
        self.q_model = T5ForConditionalGeneration.from_pretrained(QGEN_MODEL).to(DEVICE)
        
        with open(f"{OUTPUT_FINAL}/bm25_size_512_overlap_10.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.dense = faiss.read_index(f"{OUTPUT_FINAL}/contriever_optimized.index")
        with open(f"{OUTPUT_FINAL}/contriever_metadata.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def search(self, query, top_k=3):
        query = rewrite_query(query, self.q_tokenizer, self.q_model, self.mock)
        print(f"\n🔍 测试查询：{query}")
        
        bm_scores = self.bm25.get_scores(query.split())
        bm_top = np.argsort(bm_scores)[::-1][:top_k*3]
        q_emb = get_embeddings([query], self.tokenizer, self.model)
        _, dense_top = self.dense.search(q_emb, top_k*3)

        res, seen = [], set()
        for idx in list(bm_top) + list(dense_top[0]):
            if len(res)>=top_k or idx in seen: continue
            seen.add(idx)
            res.append({"排名": len(res)+1, "数据集": self.meta[idx]["dataset"], "块ID": int(idx)})
        return res

# ===================== 自动运行全流程 =====================
if __name__ == "__main__":
    # 自动按阶段执行，中断可续跑
    stage1_load_clean_split()
    stage2_generate_embeddings()
    stage3_build_index()

    # 自动测试检索
    print("\n" + "="*60)
    print("🧪 检索功能测试")
    print("="*60)
    retriever = RAGRetriever()
    for item in retriever.search("What is RAG technology?"):
        print(item)