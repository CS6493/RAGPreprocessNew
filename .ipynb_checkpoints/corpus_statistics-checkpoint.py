import os
import argparse
from datasets import load_dataset

# 继承您原有的配置项
DATASETS_CONFIG = {
    "Natural_Questions": {
        "path": "sentence-transformers/natural-questions", 
        "split": "train"
    },
    "PubMedQA": {
        "path": "qiaojin/PubMedQA", 
        "name": "pqa_labeled", 
        "split": "train"
    },
    "FinanceBench": {
        "path": "PatronusAI/financebench", 
        "split": "train"
    },
    "HotpotQA": {
        "path": "hotpot_qa", 
        "name": "distractor", 
        "split": "train"
    }
}

def download_and_save(dataset_name, sample_size=None, output_base_dir="./dataset"):
    if dataset_name not in DATASETS_CONFIG:
        raise ValueError(f"❌ 未知的数据集: {dataset_name}")
    
    cfg = DATASETS_CONFIG[dataset_name]
    load_kwargs = {"path": cfg["path"], "split": cfg["split"]}
    if "name" in cfg:
        load_kwargs["name"] = cfg["name"]
        
    print(f"\n📥 正在从 HuggingFace 下载 [{dataset_name}] ...")
    print(f"   参数: {load_kwargs}")
    
    # 1. 加载数据集
    ds = load_dataset(**load_kwargs)
    
    # 2. 截取采样 (如果提供了 sample_size 且有效)
    if sample_size is not None and sample_size > 0:
        actual_size = min(sample_size, len(ds))
        ds = ds.select(range(actual_size))
        print(f"✂️ 已按要求截取前 {actual_size} 条数据。")
    else:
        print(f"📦 已获取完整数据集，共 {len(ds)} 条数据。")

    # 3. 创建保存目录
    save_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 保存为 HuggingFace 磁盘格式 (方便后续代码快速 load_from_disk)
    ds.save_to_disk(save_dir)
    print(f"💾 HuggingFace 原生格式已保存至: {save_dir}/")
    
    # 5. [额外福利] 保存一份 JSONL 格式，方便人类直接查看
    json_path = os.path.join(save_dir, f"{dataset_name}.jsonl")
    ds.to_json(json_path, force_ascii=False)
    print(f"📝 纯文本 JSONL 格式已保存至: {json_path}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="独立下载 RAG 项目评测数据集")
    
    # 支持的数据集列表，加了一个 'all' 方便一键下载所有
    choices = list(DATASETS_CONFIG.keys()) + ["all"]
    
    parser.add_argument("--dataset", type=str, required=True, choices=choices, 
                        help="选择要下载的数据集，或者输入 'all' 下载全部")
    
    # sample_size 默认不传，代表 None
    parser.add_argument("--sample_size", type=int, default=None, 
                        help="截取前 N 条数据，不传此参数则下载全部")
    
    parser.add_argument("--output_dir", type=str, default="./dataset", 
                        help="保存的根目录，默认为当前目录下的 ./dataset")

    args = parser.parse_args()

    # 如果选择 all，则循环下载所有数据集
    if args.dataset == "all":
        print(f"🚀 开始批量下载所有数据集 (Sample Size: {args.sample_size if args.sample_size else '全部'})...")
        for ds_name in DATASETS_CONFIG.keys():
            download_and_save(ds_name, args.sample_size, args.output_dir)
    else:
        download_and_save(args.dataset, args.sample_size, args.output_dir)

if __name__ == "__main__":
    main()