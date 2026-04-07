import pickle
import numpy as np
import json
import os

# ===================== 配置您的文件路径 =====================
# 根据您的截图，文件名如下。请确保脚本与这些文件在同一目录下，或补全绝对路径。
PREFIX = "./rag_output/HotpotQA_train_cs512_co50"
CHUNKS_FILE = f"{PREFIX}_chunks.pkl"
EMBEDDINGS_FILE = f"{PREFIX}_embeddings.npy"
META_FILE = f"{PREFIX}_meta.json"

def verify_rag_pipeline_output():
    print("="*60)
    print(f"🔍 开始验证 RAG 预处理文件: {PREFIX}")
    print("="*60)

    # ---------------------------------------------------------
    # 1. 尝试加载所有数据
    # ---------------------------------------------------------
    try:
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ 成功加载 Chunks: {CHUNKS_FILE}")

        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"✅ 成功加载 Embeddings: {EMBEDDINGS_FILE}")

        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"✅ 成功加载 Metadata: {META_FILE}")
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        return
    except Exception as e:
        print(f"❌ 文件加载出错: {e}")
        return

    # ---------------------------------------------------------
    # 2. 数量与维度对齐检查（最重要的一步）
    # ---------------------------------------------------------
    print("\n📊 [阶段 1] 数据对齐与维度检查")
    len_chunks = len(chunks)
    len_emb = embeddings.shape[0]
    emb_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
    len_meta = len(meta)

    print(f"   - Chunks 列表长度: {len_chunks}")
    print(f"   - Embeddings 矩阵形状: {embeddings.shape} (行数: {len_emb}, 维度: {emb_dim})")
    print(f"   - Metadata 列表长度: {len_meta}")

    if len_chunks == len_emb == len_meta:
        print("   👉 结论: 完美！三者的数据量完全一对一匹配。")
    else:
        print("   ❌ 警告: 数据量不匹配！请检查分块和向量化代码是否存在丢数据的 Bug。")

    # ---------------------------------------------------------
    # 3. 质量抽样检查（查看第一条和最后一条数据）
    # ---------------------------------------------------------
    print("\n🔬 [阶段 2] 数据内容抽样检查")
    if len_chunks > 0:
        indices_to_check = [0, len_chunks - 1] # 检查首尾两条
        labels = ["第一条", "最后一条"]

        for idx, label in zip(indices_to_check, labels):
            print(f"\n   --- {label}数据 (Index: {idx}) ---")
            
            # 检查 Chunk
            chunk_text = chunks[idx]
            display_text = repr(chunk_text[:100]) + ("..." if len(chunk_text) > 100 else "")
            print(f"   [文本]: {display_text} (总长度: {len(chunk_text)} 字符)")
            
            # 检查 Meta
            print(f"   [元数据]: {json.dumps(meta[idx], ensure_ascii=False)}")
            
            # 检查 Embedding
            emb_vec = embeddings[idx]
            print(f"   [向量预览]: {emb_vec[:4]} ... (类型: {emb_vec.dtype})")
            
            # 异常值检测
            if np.isnan(emb_vec).any() or np.isinf(emb_vec).any():
                print("   ⚠️ 警告: 该条向量中检测到 NaN 或 Inf 异常值！")
    else:
        print("   ⚠️ 数据集为空，无法进行抽样。")

    print("\n✨ 检查完毕！")

if __name__ == "__main__":
    verify_rag_pipeline_output()