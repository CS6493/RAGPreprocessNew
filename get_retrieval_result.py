import os
import re
import json

def main():
    # ================= 配置区域 =================
    # 请修改这里的路径为你的实际文件夹路径
    input_dir = "./retrieve_output"  # 存放原始JSON文件的文件夹
    output_file = "./retrieve_output/retrieve_aggregated_metrics.json"  # 结果保存路径
    
    # 定义需要提取的指标（与JSON文件中的key完全一致）
    target_metrics = [
        "Top1_EM",
        "Top1_F1",
        "Max_Normalized_F1",
        "Hit@K",
        "Recall@K",
        "NDCG@K",
        "MRR"
    ]
    # ===========================================

    # 临时存储结构: { 数据集名称: { 检索方法: { 指标字典 } } }
    temp_results = {}
    processed_files = 0
    failed_files = []

    # 正则表达式解析文件名
    # 匹配格式: retrieval_summary_{数据集}_cs512co50_{检索方法}.json
    filename_regex = re.compile(
        r"^retrieval_summary_(.+?)_cs512co50_(.+?)\.json$",
        re.IGNORECASE
    )

    # 检查输入文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"❌ 错误：输入文件夹 '{input_dir}' 不存在！")
        return

    print(f"📂 开始扫描文件夹: {input_dir}")
    print("-" * 60)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        # 只处理JSON文件
        if not os.path.isfile(file_path) or not filename.endswith(".json"):
            continue

        # 匹配文件名格式
        match = filename_regex.match(filename)
        if not match:
            print(f"⚠️  跳过不匹配的文件: {filename}")
            continue

        dataset_name = match.group(1)
        method_name = match.group(2)

        try:
            # 读取并解析JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取指定指标
            metrics = {}
            for metric in target_metrics:
                if metric in data:
                    metrics[metric] = data[metric]
                else:
                    metrics[metric] = None
                    print(f"⚠️  文件 {filename} 中缺少指标: {metric}")

            # 存入临时结果
            if dataset_name not in temp_results:
                temp_results[dataset_name] = {}
            
            temp_results[dataset_name][method_name] = metrics
            processed_files += 1
            print(f"✅ 成功处理: {filename}")
            print(f"   数据集: {dataset_name}, 检索方法: {method_name}")

        except json.JSONDecodeError:
            failed_files.append(filename)
            print(f"❌ JSON解析失败: {filename}")
        except Exception as e:
            failed_files.append(filename)
            print(f"❌ 处理文件 {filename} 时出错: {str(e)}")

    print("-" * 60)
    print(f"\n📊 处理完成统计:")
    print(f"   成功处理: {processed_files} 个文件")
    print(f"   处理失败: {len(failed_files)} 个文件")
    
    if failed_files:
        print(f"   失败文件列表: {', '.join(failed_files)}")

    # 转换为用户要求的输出格式
    # 格式: [ { 数据集名称: { 检索方法: { 指标... } } } ]
    final_results = []
    for dataset, methods in temp_results.items():
        final_results.append({
            dataset: methods
        })

    # 保存结果到JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        
        print(f"\n💾 结果已成功保存到: {output_file}")
    except Exception as e:
        print(f"\n❌ 保存结果文件时出错: {str(e)}")

if __name__ == "__main__":
    main()