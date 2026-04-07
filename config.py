import os
import torch

# ===================== 全局环境配置 =====================
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TQDM_DISABLE_HTML"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ===================== 默认模型参数 =====================
DEFAULT_MODEL_NAME = "facebook/contriever-msmarco"
DEFAULT_QGEN_MODEL = "google/flan-t5-small"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 默认路径 =====================
DEFAULT_OUTPUT_DIR = "./rag_output"

# ===================== 数据集配置 =====================
DATASETS_CONFIG = {
    "Natural_Questions": {
        "path": "sentence-transformers/natural-questions", 
        "split": "train", 
        "streaming": False
    },
    "PubMedQA": {
        "path": "qiaojin/PubMedQA", 
        "name": "pqa_labeled", 
        "split": "train", 
        "streaming": False
    },
    "FinanceBench": {
        "path": "PatronusAI/financebench", 
        "split": "train", 
        "streaming": False
    },
    "HotpotQA": {
        "path": "hotpot_qa", 
        "name": "distractor", 
        "split": "train", 
        "streaming": False
    }
}

def get_file_paths(base_dir, dataset_name, split, chunk_size, chunk_overlap):
    """根据参数生成合理且具有辨识度的文件名"""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, f"{dataset_name}"), exist_ok=True)
    prefix = f"{dataset_name}_{split}_cs{chunk_size}_co{chunk_overlap}"
    
    return {
        "chunks": os.path.join(base_dir, f"{dataset_name}", f"{prefix}_chunks.pkl"),
        "meta": os.path.join(base_dir, f"{dataset_name}", f"{prefix}_meta.json"),
        "embeddings": os.path.join(base_dir, f"{dataset_name}", f"{prefix}_embeddings.npy"),
        "bm25": os.path.join(base_dir, f"{dataset_name}", f"{prefix}_bm25.pkl"),
        "faiss": os.path.join(base_dir, f"{dataset_name}", f"{prefix}_faiss.index")
    }