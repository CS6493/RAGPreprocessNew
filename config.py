import os
import torch

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TQDM_DISABLE_HTML"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEFAULT_MODEL_NAME = "facebook/contriever-msmarco"
DEFAULT_QGEN_MODEL = "google/flan-t5-small"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_OUTPUT_DIR = "./rag_output"

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

DEFAULT_GEN_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_GEN_MODE = "base"
DEFAULT_MAX_TOKENS = 96
DEFAULT_TEMPERATURE = 0.0
DEFAULT_DO_SAMPLE = False
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_USE_4BIT = True
DEFAULT_TOP_K_CONTEXTS = 3
DEFAULT_MAX_TOTAL_CONTEXT_CHARS = 3200
DEFAULT_MAX_SENTENCES_PER_CONTEXT = 2
DEFAULT_USE_CONTEXT_COMPRESSION = True
DEFAULT_USE_FALLBACK_PROMPT = True

LOCAL_MODEL_PRESETS = {
    "qwen_base_7b": "Qwen/Qwen2.5-7B",
    "qwen_instruct_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_base_3b": "Qwen/Qwen2.5-3B",
    "qwen_instruct_3b": "Qwen/Qwen2.5-3B-Instruct",
}

API_CONFIG = {
    "DeepSeek": {
        "api_key": "sk-",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "Qwen": {
        "api_key": "sk-",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus"
    }
}
DEFAULT_API_PROVIDER = "DeepSeek"

def get_file_paths(base_dir, dataset_name, split, chunk_size, chunk_overlap):
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
