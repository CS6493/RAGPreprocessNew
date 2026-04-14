import os
import torch

# ===================== 全局环境配置 =====================
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
# os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

os.environ["TQDM_DISABLE_HTML"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ===================== 默认模型参数 =====================
DEFAULT_MODEL_NAME = "facebook/contriever-msmarco"
DEFAULT_QGEN_MODEL = "google/flan-t5-small"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 默认路径 =====================
DEFAULT_OUTPUT_DIR = "./rag_output"
DEFAULT_RETRIEVE_QUERY_FILE = "./data/queries.json"
DEFAULT_RETRIEVE_OUTPUT_DIR = "./retrieve_output"
DEFAULT_GENERATION_OUTPUT_DIR = "./generation_output"
DEFAULT_OFFLINE_KNOWLEDGE_FILE = "./data/test_knowledge.json"

# ===================== 数据集配置 =====================
DATASETS_CONFIG = {
    "HotpotQA": {
        "path": "hotpot_qa", 
        "name": "distractor", 
        "split": "train", 
        "streaming": False
    },
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
    }
}

# ===================== 生成模型 (Generation) 默认参数 =====================
DEFAULT_GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_GEN_MODEL_CHOICES = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
]
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7

# 新增 API 相关配置
API_CONFIG = {
    "DeepSeek": {
        "api_key": "sk-cee7878a656148c38d091624a06887a3", # 替换为你的 Key
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "Qwen": {
        "api_key": "sk-fd297ac637d844feb60524a70dd8e368", # 替换为你的 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus" # 或 qwen-max
    },
    "Qwen2.5-7b-instruct": {
        "api_key": "sk-fd297ac637d844feb60524a70dd8e368", # 替换为你的 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen2.5-7b-instruct"
    }   
}

# 默认生成模型选择
DEFAULT_API_PROVIDER = "DeepSeek"

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