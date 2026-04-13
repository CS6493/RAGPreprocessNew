import json
import os
import time

import pandas as pd
import streamlit as st

from config import (
    API_CONFIG,
    DATASETS_CONFIG,
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_SENTENCES_PER_CONTEXT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K_CONTEXTS,
    DEFAULT_USE_4BIT,
    LOCAL_MODEL_PRESETS,
    get_file_paths,
)
from evaluation import calculate_fact_score_via_llm, exact_match_score, f1_score
from generator import RAGGenerator
from retriever import RAGRetriever

st.set_page_config(page_title="RAG 端到端系统演示", page_icon="🤖", layout="wide")
st.title("🤖 端到端 RAG 系统可视化平台")


@st.cache_resource(show_spinner=False)
def load_rag_system(
    dataset_name,
    retrieval_top_k,
    gen_model,
    gen_mode,
    use_api,
    max_new_tokens,
    temperature,
    do_sample,
    repetition_penalty,
    no_repeat_ngram_size,
    use_4bit,
    context_top_k,
    max_total_context_chars,
    max_sentences_per_context,
    use_context_compression,
    use_fallback_prompt,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.join(current_dir, "rag_output")
    paths = get_file_paths(abs_output_dir, dataset_name, "train", 512, 50)

    retriever = RAGRetriever(
        paths=paths,
        embed_model_name="facebook/contriever-msmarco",
        qgen_model_name="google/flan-t5-small",
        chunk_size=512,
    )

    api_key = None
    api_base_url = None
    actual_model_name = gen_model

    if use_api:
        if "deepseek" in gen_model.lower():
            api_key = API_CONFIG["DeepSeek"]["api_key"]
            api_base_url = API_CONFIG["DeepSeek"]["base_url"]
            actual_model_name = API_CONFIG["DeepSeek"]["model"]
        else:
            api_key = API_CONFIG["Qwen"]["api_key"]
            api_base_url = API_CONFIG["Qwen"]["base_url"]
            actual_model_name = API_CONFIG["Qwen"]["model"]

    generator = RAGGenerator(
        model_name=actual_model_name,
        generation_mode=gen_mode,
        use_api=use_api,
        api_key=api_key,
        api_base_url=api_base_url,
        use_4bit=use_4bit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_k_contexts=context_top_k,
        max_total_context_chars=max_total_context_chars,
        max_sentences_per_context=max_sentences_per_context,
        use_context_compression=use_context_compression,
        use_fallback_prompt=use_fallback_prompt,
    )

    return retriever, generator


with st.sidebar:
    st.header("⚙️ 全局参数设置")
    selected_dataset = st.selectbox("📂 选择数据集", list(DATASETS_CONFIG.keys()), index=0)
    selected_method = st.selectbox("🔍 检索策略", ["hybrid", "dense", "sparse"], index=0)
    retrieval_top_k = st.slider("📄 检索 Top-K", min_value=1, max_value=10, value=3)

    st.divider()
    st.subheader("🧠 生成模型设置")
    gen_mode = st.selectbox("生成模式", ["base", "instruct"], index=0)
    use_api = st.toggle("🌐 使用 API 模式", value=False)

    if use_api:
        gen_model = st.selectbox("模型选择", ["Qwen", "DeepSeek"])
    else:
        preset_name = "qwen_base_7b" if gen_mode == "base" else "qwen_instruct_7b"
        preset = LOCAL_MODEL_PRESETS[preset_name]
        gen_model = st.text_input("本地模型路径", value=preset)

    max_new_tokens = st.slider("Max new tokens", min_value=16, max_value=256, value=DEFAULT_MAX_TOKENS, step=8)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(DEFAULT_TEMPERATURE), step=0.05)
    do_sample = st.toggle("Do sample", value=DEFAULT_DO_SAMPLE)
    repetition_penalty = st.slider("Repetition penalty", min_value=1.0, max_value=1.3, value=float(DEFAULT_REPETITION_PENALTY), step=0.01)
    no_repeat_ngram_size = st.slider("No repeat ngram size", min_value=0, max_value=8, value=DEFAULT_NO_REPEAT_NGRAM_SIZE, step=1)
    use_4bit = st.toggle("Use 4-bit quantization", value=DEFAULT_USE_4BIT)
    use_context_compression = st.toggle("Use context compression", value=True)
    use_fallback_prompt = st.toggle("Use fallback prompt", value=True)
    context_top_k = st.slider("Generation context Top-K", min_value=1, max_value=5, value=DEFAULT_TOP_K_CONTEXTS)
    max_total_context_chars = st.slider("Max total context chars", min_value=800, max_value=5000, value=DEFAULT_MAX_TOTAL_CONTEXT_CHARS, step=100)
    max_sentences_per_context = st.slider("Max sentences per context", min_value=1, max_value=5, value=DEFAULT_MAX_SENTENCES_PER_CONTEXT)

with st.spinner("正在加载系统组件 (首次加载本地模型会较慢)..."):
    try:
        retriever, generator = load_rag_system(
            selected_dataset,
            retrieval_top_k,
            gen_model,
            gen_mode,
            use_api,
            max_new_tokens,
            temperature,
            do_sample,
            repetition_penalty,
            no_repeat_ngram_size,
            use_4bit,
            context_top_k,
            max_total_context_chars,
            max_sentences_per_context,
            use_context_compression,
            use_fallback_prompt,
        )
        st.sidebar.success("✅ 系统加载完成！")
    except Exception as e:
        st.sidebar.error(f"加载失败，请确保已先运行 pipeline 构建索引。\n错误: {e}")
        st.stop()

tab1, tab2 = st.tabs(["💬 基础问答 (QA)", "📊 批量评估 (Batch Eval)"])

with tab1:
    st.markdown("### 提出您的问题")
    user_query = st.chat_input("请输入您想查询的问题...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 正在检索相关文档..."):
                start_time = time.time()
                retrieved_results, final_query = retriever.search(user_query, top_k=retrieval_top_k, method=selected_method)
                contexts = [res["meta_info"].get("text", "") for res in retrieved_results]
                retrieval_time = time.time() - start_time

            with st.spinner("🧠 正在生成答案..."):
                start_time = time.time()
                debug = generator.generate(question=user_query, contexts=contexts, return_debug=True)
                generation_time = time.time() - start_time

            st.write(debug["prediction"])
            st.caption(
                f"⏱️ 检索耗时: {retrieval_time:.2f}s | 生成耗时: {generation_time:.2f}s | "
                f"Prompt: {debug['prompt_variant']} | Heuristic: {debug['heuristic_score']:.2f}"
            )

            with st.expander("⚙️ 查看内部处理过程 (检索详情 & 生成详情)", expanded=False):
                st.markdown(f"**重写后查询:** `{final_query}`")
                st.markdown(f"**Question type:** `{debug['question_type']}`")
                st.markdown("#### 🧩 生成前压缩后的 Context")
                for i, ctx in enumerate(debug["prepared_contexts"], start=1):
                    st.code(f"[Prepared {i}] {ctx}", language="text")

                st.markdown("#### 📄 原始召回文档块")
                for i, res in enumerate(retrieved_results):
                    score = res.get("rrf_score", res.get("bm25_score", res.get("l2_distance", "N/A")))
                    st.info(f"**[{res['排名']}] 来源: {res['meta_info'].get('source_id', 'Unknown')}** | 分数: {score}")
                    st.write(res["meta_info"].get("text", "No text context available."))
                    with st.popover("查看完整 Meta 信息"):
                        st.json(res["meta_info"])

                st.markdown("#### 🧠 原始模型输出")
                st.code(debug["raw_output"], language="text")

with tab2:
    st.markdown("### 批量评测任务")
    st.markdown("请在下方输入测试集（每行一个 JSON，包含 `question` 和 `answer` 字段）:")

    default_batch_data = """{"question": "When did richmond last play in a preliminary final?", "answer": "2017"}
{"question": "What is the capital of France?", "answer": "Paris"}"""
    batch_input = st.text_area("批量查询输入", value=default_batch_data, height=150)

    if st.button("🚀 开始批量评测", type="primary"):
        try:
            queries = [json.loads(line) for line in batch_input.strip().split("\n") if line.strip()]
        except Exception as e:
            st.error(f"JSON 解析失败，请检查输入格式: {e}")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()

        rows = []
        em_total, f1_total, fact_total = 0, 0, 0

        for idx, item in enumerate(queries):
            q = item["question"]
            truth = item.get("answer", "")
            status_text.text(f"正在处理 ({idx+1}/{len(queries)}): {q}")

            ret_res, _ = retriever.search(q, top_k=retrieval_top_k, method=selected_method)
            ctxs = [r["meta_info"].get("text", "") for r in ret_res]
            debug = generator.generate(question=q, contexts=ctxs, return_debug=True)
            pred = debug["prediction"]

            em = exact_match_score(pred, truth)
            f1 = f1_score(pred, truth)
            fact = 0
            if pred.strip() and pred.lower() != "insufficient evidence.":
                fact = calculate_fact_score_via_llm(pred, ctxs, generator)

            em_total += em
            f1_total += f1
            fact_total += fact

            rows.append({
                "Question": q,
                "Ground Truth": truth,
                "Prediction": pred,
                "EM": em,
                "F1": round(f1, 3),
                "FActScore": round(fact, 3),
                "Prompt Variant": debug["prompt_variant"],
                "Heuristic Score": round(debug["heuristic_score"], 3),
                "Prepared Contexts": debug["prepared_contexts"],
                "Retrieved Meta": [r["meta_info"] for r in ret_res],
                "Raw Output": debug["raw_output"],
            })
            progress_bar.progress((idx + 1) / len(queries))

        status_text.text("✅ 批量评测完成！")
        st.subheader("🏆 全局评估结果")
        num_samples = len(queries) if queries else 1
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Exact Match (EM)", f"{(em_total/num_samples)*100:.1f}%")
        col2.metric("📏 F1 Score", f"{(f1_total/num_samples)*100:.1f}%")
        col3.metric("🔍 FActScore", f"{(fact_total/num_samples)*100:.1f}%")

        st.divider()
        st.subheader("📋 详细查询与生成结果")

        result_df = pd.DataFrame([{k: v for k, v in row.items() if k not in {"Prepared Contexts", "Retrieved Meta", "Raw Output"}} for row in rows])
        st.dataframe(result_df, use_container_width=True)

        for i, res in enumerate(rows):
            with st.expander(f"Q{i+1}: {res['Question']} | F1: {res['F1']} | Prompt: {res['Prompt Variant']}"):
                st.markdown(f"**真实答案:** {res['Ground Truth']}")
                st.markdown(f"**模型预测:** {res['Prediction']}")
                st.markdown(f"**Heuristic score:** {res['Heuristic Score']}")
                st.markdown("#### 🧩 Prepared contexts")
                for j, ctx in enumerate(res["Prepared Contexts"], start=1):
                    st.code(f"[Prepared {j}] {ctx}", language="text")
                st.markdown("#### 🧠 原始模型输出")
                st.code(res["Raw Output"], language="text")
                st.markdown("#### 🔍 检索结果")
                for j, meta in enumerate(res["Retrieved Meta"], start=1):
                    st.json(meta)
