import json
import random

# 1. 加载文件
file_path = './rag_output/HotpotQA/HotpotQA_train_cs512_co50_meta.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 使用字典进行去重
# 我们以 question 作为键，这样即使同一个问题出现在 10 个 chunk 里，也只会被记录一次
unique_qa_dict = {}

for item in data:
    q = item.get("question")
    a = item.get("answer")
    if q and a:
        # 如果问题不在字典中，则添加
        if q not in unique_qa_dict:
            unique_qa_dict[q] = a

# 3. 将去重后的结果转回列表格式
all_unique_pairs = [{"question": q, "answer": a} for q, a in unique_qa_dict.items()]

print(f"总 chunk 数: {len(data)}")
print(f"去重后的唯一问题数: {len(all_unique_pairs)}")

# 4. 随机抽取 100 个
sample_size = min(len(all_unique_pairs), 100)
sampled_qa = random.sample(all_unique_pairs, sample_size)

# 5. 按照要求的格式输出
for qa in sampled_qa:
    print(json.dumps(qa, ensure_ascii=False))