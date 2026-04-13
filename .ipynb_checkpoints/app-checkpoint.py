import streamlit as st
import os
import json
import time
import pandas as pd

# 导入您现有的项目模块 (完全解耦，不修改原代码)
from config import DATASETS_CONFIG, DEFAULT_OUTPUT_DIR, get_file_paths, API_CONFIG
from retriever import RAGRetriever
from generator import RAGGenerator
from evaluation import exact_match_score, f1_score, calculate_fact_score_via_llm

# ==========================================
# 1. 页面基本配置
# ==========================================
st.set_page_config(page_title="RAG 端到端系统演示", page_icon="🤖", layout="wide")
st.title("🤖 端到端 RAG 系统可视化平台")

# ==========================================
# 2. 全局状态缓存与初始化 (避免每次点击重新加载模型)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_rag_system(dataset_name, method, top_k, gen_model, use_api):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.join(current_dir, "rag_output")
    paths = get_file_paths(abs_output_dir, dataset_name, "train", 512, 50)
    
    retriever = RAGRetriever(
        paths=paths, 
        embed_model_name="facebook/contriever-msmarco", 
        qgen_model_name="google/flan-t5-small", 
        chunk_size=512
    )
    
    # 👇 修改部分 👇
    api_key = None
    api_base_url = None
    actual_model_name = gen_model # 默认使用 UI 传过来的名字（针对本地模型）
    
    if use_api:
        if "deepseek" in gen_model.lower():
            api_key = API_CONFIG["DeepSeek"]["api_key"]
            api_base_url = API_CONFIG["DeepSeek"]["base_url"]
            # 🌟 强制将请求模型名替换为 config.py 中配置的真实名称 (deepseek-chat)
            actual_model_name = API_CONFIG["DeepSeek"]["model"] 
        else:
            api_key = API_CONFIG["Qwen"]["api_key"]
            api_base_url = API_CONFIG["Qwen"]["base_url"]
            # 🌟 强制将请求模型名替换为 config.py 中配置的真实名称 (qwen-plus)
            actual_model_name = API_CONFIG["Qwen"]["model"]

    generator = RAGGenerator(
        model_name=actual_model_name, # 🌟 这里传入修正后的实际模型名
        use_api=use_api,
        api_key=api_key,
        api_base_url=api_base_url
    )
    
    return retriever, generator

# ==========================================
# 3. 侧边栏：参数配置区域 (Req 4: 参数选择)
# ==========================================
with st.sidebar:
    st.header("⚙️ 全局参数设置")
    
    selected_dataset = st.selectbox("📂 选择数据集", list(DATASETS_CONFIG.keys()), index=0)
    selected_method = st.selectbox("🔍 检索策略", ["hybrid", "dense", "sparse"], index=0)
    top_k = st.slider("📄 召回文档数 (Top-K)", min_value=1, max_value=10, value=3)
    
    st.divider()
    st.subheader("🧠 生成模型设置")
    use_api = st.toggle("🌐 使用 API 模式", value=True)
    
    if use_api:
        gen_model = st.selectbox("模型选择", ["Qwen", "DeepSeek"])
    else:
        gen_model = st.text_input("本地模型路径", "Qwen/Qwen2.5-7B-Instruct")

# 根据侧边栏配置加载核心组件
with st.spinner("正在加载系统组件 (如果是首次加载本地模型可能需要较长时间)..."):
    try:
        retriever, generator = load_rag_system(selected_dataset, selected_method, top_k, gen_model, use_api)
        st.sidebar.success("✅ 系统加载完成！")
    except Exception as e:
        st.sidebar.error(f"加载失败，请确保您已经先运行了 offline pipeline 生成了索引！\n错误: {e}")
        st.stop()

# ==========================================
# 4. 主界面：双 Tab 切换 (Req 1: 顶部菜单 Tab)
# ==========================================
tab1, tab2 = st.tabs(["💬 基础问答 (QA)", "📊 批量评估 (Batch Eval)"])

# ------------------------------------------
# Tab 1: 基础问答区域 (Req 2)
# ------------------------------------------
with tab1:
    st.markdown("### 提出您的问题")
    
    # 使用 Streamlit 原生的聊天输入框
    user_query = st.chat_input("请输入您想查询的问题...")
    
    if user_query:
        # 显示用户问题
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            # 1. 检索阶段
            with st.spinner("🔍 正在检索相关文档..."):
                start_time = time.time()
                # 调用现有方法
                retrieved_results, final_query = retriever.search(user_query, top_k=top_k, method=selected_method)
                contexts = [res["meta_info"].get("text", "") for res in retrieved_results]
                retrieval_time = time.time() - start_time
            
            # 2. 生成阶段
            with st.spinner("🧠 正在生成答案..."):
                start_time = time.time()
                # 调用现有方法
                answer = generator.generate(question=user_query, contexts=contexts)
                generation_time = time.time() - start_time
                
            # 展示最终答案
            st.write(answer)
            st.caption(f"⏱️ 检索耗时: {retrieval_time:.2f}s | 生成耗时: {generation_time:.2f}s")
            
            # Req 2: 展开栏展示检索详情与改写结果
            with st.expander("⚙️ 查看内部处理过程 (检索详情 & 重写结果)", expanded=False):
                st.markdown(f"**原问题重写为:** `{final_query}`")
                st.markdown("#### 📄 召回的文档块 (Top-K)")
                for i, res in enumerate(retrieved_results):
                    st.info(f"**[{res['排名']}] 来源: {res['meta_info'].get('source_id', 'Unknown')}** | 得分: {res.get('rrf_score', res.get('bm25_score', 'N/A'))}")
                    st.write(res["meta_info"].get("text", "No text context available."))
                    with st.popover("查看完整 Meta 信息"):
                        st.json(res["meta_info"])

# ------------------------------------------
# Tab 2: 批量评估区域 (Req 3)
# ------------------------------------------
with tab2:
    st.markdown("### 批量评测任务")
    
    st.markdown("请在下方输入您的测试集（每行一个 JSON 格式，包含 `question` 和 `answer` 字段）：")
    default_batch_data = """{"question": "When did richmond last play in a preliminary final?", "answer": "2017"}
{"question": "What is the capital of France?", "answer": "Paris"}"""
    
    batch_input = st.text_area("批量查询输入", value=default_batch_data, height=150)
    
    if st.button("🚀 开始批量评测", type="primary"):
        # 解析输入
        try:
            queries = [json.loads(line) for line in batch_input.strip().split("\n") if line.strip()]
        except Exception as e:
            st.error(f"JSON 解析失败，请检查输入格式: {e}")
            st.stop()
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_data = []
        em_total, f1_total, fact_total = 0, 0, 0
        
        # 运行批量评测循环
        for idx, item in enumerate(queries):
            q = item["question"]
            truth = item.get("answer", "")
            status_text.text(f"正在处理 ({idx+1}/{len(queries)}): {q}")
            
            # 检索与生成
            ret_res, _ = retriever.search(q, top_k=top_k, method=selected_method)
            ctxs = [r["meta_info"].get("text", "") for r in ret_res]
            pred = generator.generate(question=q, contexts=ctxs)
            
            # 计算独立指标
            em = exact_match_score(pred, truth)
            f1 = f1_score(pred, truth)
            fact = 0
            if pred.strip() and pred.lower() != "insufficient evidence.":
                fact = calculate_fact_score_via_llm(pred, ctxs, generator)
            
            em_total += em
            f1_total += f1
            fact_total += fact
            
            results_data.append({
                "Question": q,
                "Ground Truth": truth,
                "Prediction": pred,
                "EM": em,
                "F1": round(f1, 2),
                "FActScore": round(fact, 2),
                "Contexts": ctxs,
                "Retrieved Meta": [r["meta_info"] for r in ret_res]
            })
            
            progress_bar.progress((idx + 1) / len(queries))
            
        status_text.text("✅ 批量评测完成！")
        
        # 1. 展示全局评估指标
        st.subheader("🏆 全局评估结果 (Evaluation Metrics)")
        col1, col2, col3 = st.columns(3)
        num_samples = len(queries)
        col1.metric("🎯 Exact Match (EM)", f"{(em_total/num_samples)*100:.1f}%")
        col2.metric("📏 F1 Score", f"{(f1_total/num_samples)*100:.1f}%")
        col3.metric("🔍 FActScore", f"{(fact_total/num_samples)*100:.1f}%")
        
        st.divider()
        
        # 2. 逐条结果展示 (支持查看对应的 Chunk 和 Meta)
        st.subheader("📋 详细查询与生成结果")
        for i, res in enumerate(results_data):
            with st.expander(f"Q{i+1}: {res['Question']} | F1: {res['F1']} | FActScore: {res['FActScore']}"):
                st.markdown(f"**真实答案 (Ground Truth):** {res['Ground Truth']}")
                st.markdown(f"**模型预测 (Prediction):** {res['Prediction']}")
                
                st.markdown("#### 🔍 本次生成的依据 (检索结果)")
                for j, ctx in enumerate(res["Contexts"]):
                    st.info(f"**Context {j+1}:** {ctx}")
                    with st.popover(f"查看 Context {j+1} 元数据"):
                        st.json(res["Retrieved Meta"][j])