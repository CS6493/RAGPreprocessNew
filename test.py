import json

# 定义文件名
input_file = './rag_output/FinanceBench/FinanceBench_train_cs512_co50_meta.json'
output_file = './data/financebench_queries_50.json'

def extract_unique_qa(limit=50):
    try:
        # 1. 读取原始 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extracted_data = []
        seen_questions = set()  # 用于记录已经提取过的问题内容
        
        # 2. 遍历并去重提取
        for item in data:
            # 如果已经提取够了，直接跳出循环
            if len(extracted_data) >= limit:
                break
                
            question = item.get("question")
            answer = item.get("answer")
            
            # 只有当 question 不在集合中时，才进行提取
            if question not in seen_questions:
                extracted_data.append({
                    "question": question,
                    "answer": answer
                })
                seen_questions.add(question) # 将问题加入“已见”集合
        
        # 3. 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
            
        print(f"提取完成！共提取了 {len(extracted_data)} 个唯一问题，已保存至 {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except json.JSONDecodeError:
        print("错误：JSON 格式不正确")

if __name__ == "__main__":
    extract_unique_qa(50)