import os
import json
import copy
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

class RAGDatasetProfiler:
    def __init__(self, sample_size=1000, output_dir="./rag_data", raw_dir="./raw_data"):
        """
        统一支持四种数据集的获取与统计
        :param sample_size: 默认抽取样本量
        :param output_dir: 本地保存提取后(query+context)数据集的路径
        :param raw_dir: 本地保存原始数据集(raw data)的路径
        """
        self.sample_size = sample_size
        self.output_dir = output_dir
        self.raw_dir = raw_dir
        
        self.dataset_configs = {
            # 修改 1: 更改 NQ 数据集来源。支持 streaming 开关（设为 True 为流式处理，False 为全量下载）
            "Natural_Questions": {"path": "sentence-transformers/natural-questions", "split": "train", "streaming": False},
            "PubMedQA": {"path": "qiaojin/PubMedQA", "name": "pqa_labeled", "split": "train", "streaming": False},
            "FinanceBench": {"path": "PatronusAI/financebench", "split": "train", "streaming": False},
            "HotpotQA": {"path": "hotpot_qa", "name": "distractor", "split": "train", "streaming": False}
        }

    def download_raw_data(self, dataset_name):
        """
        第一步：从云端下载原始数据并保存到本地 raw_dir
        """
        config = self.dataset_configs.get(dataset_name)
        if not config:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        print(f"\n[1/3] 开始下载 {dataset_name} 的原始数据... (Streaming: {config['streaming']})")
        
        dataset = load_dataset(
            config["path"], 
            name=config.get("name"), 
            split=config["split"], 
            streaming=config["streaming"]
        )

        raw_data = []
        iterator = iter(dataset) if config["streaming"] else dataset
        
        for i, item in enumerate(tqdm(iterator, total=self.sample_size, desc=f"Fetching {dataset_name}")):
            if i >= self.sample_size:
                break
            
            # 使用深拷贝避免在流式处理时出现意外修改
            item_data = copy.deepcopy(item)
            
            # 注释/移除旧版 NQ 的 Token 截断逻辑，因为 sentence-transformers 已经是处理好的文本
            
            raw_data.append(item_data)

        # 确保原始数据文件夹存在
        os.makedirs(self.raw_dir, exist_ok=True)
        raw_file_path = os.path.join(self.raw_dir, f"{dataset_name}_raw.json")
        
        # 将原始数据保存为 JSON
        with open(raw_file_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 原始数据已保存至: {raw_file_path}")
        return raw_file_path

    def process_and_extract(self, dataset_name, raw_file_path, output_format="csv"):
        """
        第二步：读取本地的原始数据，提取 query 和 context，并保存到 output_dir
        """
        print(f"[2/3] 从本地原始数据提取 Query 和 Context...")
        
        # 读取刚刚保存的原始数据
        with open(raw_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        extracted_data = []
        
        for item in tqdm(raw_data, desc=f"Processing {dataset_name}"):
            # 修改 2: 适配 sentence-transformers/natural-questions 的字段
            if dataset_name == "Natural_Questions":
                query = item.get('query', '')
                answer = item.get('answer', '')
                context = " ".join(answer) if isinstance(answer, list) else str(answer)
                
            elif dataset_name == "PubMedQA":
                query = item['question']
                context = " ".join(item['context']['contexts'])
                
            elif dataset_name == "FinanceBench":
                query = item['question']
                evidence_list = item.get('evidence', [])
                context = " ".join([str(ev.get('evidence_text', '')) for ev in evidence_list])
                
            # 修改 3: 修复 HotpotQA Context 提取错误
            elif dataset_name == "HotpotQA":
                query = item.get('question', '')
                # 获取 context 下的 sentences 列表 (结构为 List of Lists)
                sentences_list = item.get('context', {}).get('sentences', [])
                # 先将内层列表合并成段落，再将所有段落合并成完整 context
                context = " ".join([" ".join(sents) for sents in sentences_list])

            extracted_data.append({"query": query, "context": context})

        df = pd.DataFrame(extracted_data)

        # 确保提取后的文件夹存在
        os.makedirs(self.output_dir, exist_ok=True)
        out_file_path = os.path.join(self.output_dir, f"{dataset_name}.{output_format}")
        
        # 保存提取后的数据 (使用 utf-8-sig 防止在 Windows Excel 中打开时中文乱码)
        if output_format == "csv":
            df.to_csv(out_file_path, index=False, encoding='utf-8-sig')
        elif output_format == "json":
            df.to_json(out_file_path, orient="records", force_ascii=False, indent=4)
            
        print(f"✅ 提取后的数据已保存至: {out_file_path}")
        return df

    def compute_statistics(self, df):
        """第三步：计算 RAG 强相关的统计信息"""
        print(f"[3/3] 计算统计特征...")
        df['query_length'] = df['query'].apply(lambda x: len(str(x).split()))
        df['context_length'] = df['context'].apply(lambda x: len(str(x).split()))
        
        def overlap_ratio(row):
            q_words = set(str(row['query']).lower().split())
            c_words = set(str(row['context']).lower().split())
            if not q_words: return 0
            return len(q_words.intersection(c_words)) / len(q_words)
            
        df['vocab_overlap_ratio'] = df.apply(overlap_ratio, axis=1)

        stats_summary = {
            "Query平均长度(词)": round(df['query_length'].mean(), 2),
            "Query P90长度": round(np.percentile(df['query_length'], 90), 2),
            "Context平均长度(词)": round(df['context_length'].mean(), 2),
            "Context P90长度": round(np.percentile(df['context_length'], 90), 2),
            "平均词汇重合度": f"{round(df['vocab_overlap_ratio'].mean() * 100, 2)}%"
        }
        return df, stats_summary


# --- 运行示例 ---
if __name__ == "__main__":
    # 实例化并指定两个保存目录
    profiler = RAGDatasetProfiler(
        sample_size=1000, 
        output_dir="./rag_datasets_extracted", 
        raw_dir="./raw_data"
    )
    
    all_datasets_df = {}
    all_statistics = []  # 用于收集所有数据集的统计信息
    
    print("\n=== 开始处理 RAG 数据集 ===")
    for db_name in profiler.dataset_configs.keys():
        print(f"\n{'='*40}\n处理数据集: {db_name}\n{'='*40}")
        
        # 1. 下载原始数据
        raw_path = profiler.download_raw_data(db_name)
        
        # 2. 从本地原始数据提取 query + context
        df = profiler.process_and_extract(db_name, raw_path, output_format="csv")
        
        # 3. 计算统计信息
        df, stats = profiler.compute_statistics(df)
        all_datasets_df[db_name] = df
        
        # 打印当前数据集的统计结果并将数据集名称加入字典，方便后续汇总
        print(f"统计信息:")
        stats_for_csv = {"数据集名称": db_name}
        for k, v in stats.items():
            print(f"  - {k}: {v}")
            stats_for_csv[k] = v
            
        all_statistics.append(stats_for_csv)
        
    # 4. 将所有统计信息汇总并保存为单独的 CSV 文件
    print("\n=== 保存最终统计摘要 ===")
    stats_df = pd.DataFrame(all_statistics)
    stats_csv_path = "dataset_statistics_summary.csv"
    stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 所有数据集的统计摘要已成功保存至: {stats_csv_path}")