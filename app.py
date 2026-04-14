import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from config import API_CONFIG, DATASETS_CONFIG, get_file_paths
from evaluation import (
    calculate_fact_score_via_llm,
    exact_match_score,
    f1_score,
    max_normalized_f1,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from generator import RAGGenerator
from retriever import RAGRetriever


st.set_page_config(page_title="RAG End-to-End System", layout="wide")
st.title("End-to-End RAG System Dashboard")


API_RUNTIME_OPTIONS = {
    "Qwen-plus": {"provider": "Qwen", "model": "qwen-plus"},
    "Qwen2.5-7b-instruct": {"provider": "Qwen2.5-7b-instruct", "model": "qwen2.5-7b-instruct"},
    "DeepSeek-chat": {"provider": "DeepSeek", "model": "deepseek-chat"},
}


def _normalize_question(text: str) -> str:
    return str(text or "").strip().lower()


def _extract_source_id(meta_info: Dict) -> Optional[str]:
    for key in ["source_id", "financebench_id", "hotpot_id", "pubid"]:
        value = meta_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _build_gold_lookup(meta_items: List[Dict]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for item in meta_items:
        question = item.get("question") or item.get("original_question") or item.get("query")
        q_norm = _normalize_question(question)
        if not q_norm:
            continue
        sid = _extract_source_id(item)
        if sid and q_norm not in lookup:
            lookup[q_norm] = sid
    return lookup


def _dataset_query_file(dataset_name: str) -> str:
    return f"./data/{dataset_name.lower()}_queries_50.json"


def _load_default_batch_json(dataset_name: str) -> str:
    file_path = _dataset_query_file(dataset_name)
    if not os.path.exists(file_path):
        return "[]"
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_batch_payload(batch_text: str) -> List[Dict]:
    payload = json.loads(batch_text)
    if isinstance(payload, list):
        queries = payload
    elif isinstance(payload, dict):
        queries = payload.get("items") or payload.get("queries") or []
    else:
        raise ValueError("批量输入需要是 JSON 数组，或包含 items/queries 的 JSON 对象。")

    normalized = []
    for row in queries:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if question:
            normalized.append({"question": question, "answer": answer})
    return normalized


@st.cache_resource(show_spinner=False)
def load_rag_system(dataset_name, method, top_k, use_api, api_option, local_gen_model):
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
    actual_model_name = local_gen_model

    if use_api:
        option_cfg = API_RUNTIME_OPTIONS[api_option]
        provider_cfg = API_CONFIG[option_cfg["provider"]]
        api_key = provider_cfg["api_key"]
        api_base_url = provider_cfg["base_url"]
        actual_model_name = option_cfg["model"]

    generator = RAGGenerator(
        model_name=actual_model_name,
        use_api=use_api,
        api_key=api_key,
        api_base_url=api_base_url,
    )

    return retriever, generator


with st.sidebar:
    st.header("Global Parameters")

    datasets = list(DATASETS_CONFIG.keys())
    dataset_default_index = datasets.index("HotpotQA") if "HotpotQA" in datasets else 0
    selected_dataset = st.selectbox("Dataset", datasets, index=dataset_default_index)
    selected_method = st.selectbox("Retrieval Method", ["hybrid", "dense", "sparse"], index=0)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=3)

    st.divider()
    st.subheader("Generator")
    use_api = st.toggle("Use API Mode", value=True)

    if use_api:
        api_option = st.selectbox("API Model", list(API_RUNTIME_OPTIONS.keys()), index=0)
        local_gen_model = "Qwen/Qwen2.5-7B-Instruct"
    else:
        api_option = "Qwen-plus"
        local_gen_model = st.selectbox(
            "Local Model",
            ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
            index=1,
        )


with st.spinner("Loading system components..."):
    try:
        retriever, generator = load_rag_system(
            selected_dataset,
            selected_method,
            top_k,
            use_api,
            api_option,
            local_gen_model,
        )
        st.sidebar.success("System loaded")
    except Exception as e:
        st.sidebar.error(f"Load failed. Please ensure index files exist. Error: {e}")
        st.stop()


tab1, tab2 = st.tabs(["Basic QA", "Batch Evaluation"])


with tab1:
    st.markdown("### Ask a Question")
    user_query = st.chat_input("Enter your question...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving documents..."):
                start_time = time.time()
                retrieved_results, final_query = retriever.search(user_query, top_k=top_k, method=selected_method)
                contexts = [res["meta_info"].get("text", "") for res in retrieved_results]
                retrieval_time = time.time() - start_time

            with st.spinner("Generating answer..."):
                start_time = time.time()
                answer = generator.generate(question=user_query, contexts=contexts)
                generation_time = time.time() - start_time

            st.write(answer)
            st.caption(f"Retrieval time: {retrieval_time:.2f}s | Generation time: {generation_time:.2f}s")

            with st.expander("Internal Details", expanded=False):
                st.markdown(f"Rewritten query: `{final_query}`")
                st.markdown("#### Retrieved Chunks")
                for res in retrieved_results:
                    rank = res.get("排名", "N/A")
                    source_id = res["meta_info"].get("source_id", "Unknown")
                    score = res.get("rrf_score", res.get("bm25_score", "N/A"))
                    st.info(f"Rank: {rank} | Source: {source_id} | Score: {score}")
                    st.write(res["meta_info"].get("text", "No text context available."))
                    with st.popover("Meta"):
                        st.json(res["meta_info"])


with tab2:
    st.markdown("### Batch Evaluation")

    default_query_file = _dataset_query_file(selected_dataset)
    st.caption(f"Default query source: {default_query_file}")

    if "batch_input_cache" not in st.session_state:
        st.session_state["batch_input_cache"] = ""
    if st.session_state.get("batch_dataset") != selected_dataset:
        st.session_state["batch_dataset"] = selected_dataset
        st.session_state["batch_input_cache"] = _load_default_batch_json(selected_dataset)

    batch_input = st.text_area(
        "Batch input (full JSON)",
        value=st.session_state["batch_input_cache"],
        height=260,
        key="batch_input_textarea",
    )
    st.session_state["batch_input_cache"] = batch_input

    if st.button("Run Batch Evaluation", type="primary"):
        try:
            queries = _parse_batch_payload(batch_input)
        except Exception as e:
            st.error(f"JSON parse failed: {e}")
            st.stop()

        if not queries:
            st.warning("No valid queries found. Please check input JSON.")
            st.stop()

        gold_source_lookup = _build_gold_lookup(getattr(retriever, "meta", []))

        progress_bar = st.progress(0)
        status_text = st.empty()

        results_data = []
        totals = {
            "em": 0.0,
            "f1": 0.0,
            "fact": 0.0,
            "hit_at_k": 0.0,
            "top1_em": 0.0,
            "top1_f1": 0.0,
            "max_norm_f1": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "mrr": 0.0,
        }
        counts = {k: 0 for k in totals.keys()}

        for idx, item in enumerate(queries):
            q = item["question"]
            truth = item.get("answer", "")
            status_text.text(f"Processing ({idx + 1}/{len(queries)}): {q}")

            retrieved_results, rewritten_query = retriever.search(q, top_k=top_k, method=selected_method)
            ctxs = [r["meta_info"].get("text", "") for r in retrieved_results]
            pred = generator.generate(question=q, contexts=ctxs)

            em = float(exact_match_score(pred, truth)) if truth else 0.0
            f1 = float(f1_score(pred, truth)) if truth else 0.0
            fact = 0.0
            if pred.strip() and pred.lower() != "insufficient evidence.":
                fact = float(calculate_fact_score_via_llm(pred, ctxs, generator))

            top1_answer = ""
            if retrieved_results:
                top1_answer = str(retrieved_results[0].get("meta_info", {}).get("answer", "")).strip()
                if not top1_answer:
                    top1_answer = str(retrieved_results[0].get("meta_info", {}).get("text", "")).strip()

            top1_em_val = float(exact_match_score(top1_answer, truth)) if truth and top1_answer else None
            top1_f1_val = float(f1_score(top1_answer, truth)) if truth and top1_answer else None
            max_norm_f1_val = max_normalized_f1(truth, retrieved_results) if truth else None

            q_norm = _normalize_question(q)
            gold_source_id = gold_source_lookup.get(q_norm)
            recall_k_val = recall_at_k(gold_source_id, retrieved_results) if gold_source_id else None
            ndcg_k_val = ndcg_at_k(gold_source_id, retrieved_results) if gold_source_id else None
            mrr_val = mrr(gold_source_id, retrieved_results) if gold_source_id else None
            hit_at_k_val = recall_k_val if recall_k_val is not None else None

            metric_values = {
                "em": em,
                "f1": f1,
                "fact": fact,
                "hit_at_k": hit_at_k_val,
                "top1_em": top1_em_val,
                "top1_f1": top1_f1_val,
                "max_norm_f1": max_norm_f1_val,
                "recall_at_k": recall_k_val,
                "ndcg_at_k": ndcg_k_val,
                "mrr": mrr_val,
            }

            for key, value in metric_values.items():
                if value is not None:
                    totals[key] += float(value)
                    counts[key] += 1

            results_data.append(
                {
                    "Question": q,
                    "Rewritten Query": rewritten_query,
                    "Ground Truth": truth,
                    "Prediction": pred,
                    "EM": em,
                    "F1": f1,
                    "FActScore": fact,
                    "Hit@K": hit_at_k_val,
                    "Top1 EM": top1_em_val,
                    "Top1 F1": top1_f1_val,
                    "Max-Normalized F1": max_norm_f1_val,
                    "Recall@K": recall_k_val,
                    "NDCG@K": ndcg_k_val,
                    "MRR": mrr_val,
                    "Contexts": ctxs,
                    "Retrieved Meta": [r["meta_info"] for r in retrieved_results],
                }
            )

            progress_bar.progress((idx + 1) / len(queries))

        status_text.text("Batch evaluation completed")

        st.subheader("Generation Metrics")
        g1, g2, g3 = st.columns(3)
        g1.metric("Exact Match", f"{(totals['em'] / max(counts['em'], 1)) * 100:.2f}%")
        g2.metric("F1", f"{(totals['f1'] / max(counts['f1'], 1)) * 100:.2f}%")
        g3.metric("FActScore", f"{(totals['fact'] / max(counts['fact'], 1)) * 100:.2f}%")

        st.subheader("Retrieval Metrics")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Hit@K", f"{(totals['hit_at_k'] / max(counts['hit_at_k'], 1)) * 100:.2f}%" if counts["hit_at_k"] else "N/A")
        r2.metric("Top1 EM", f"{(totals['top1_em'] / max(counts['top1_em'], 1)) * 100:.2f}%" if counts["top1_em"] else "N/A")
        r3.metric("Top1 F1", f"{(totals['top1_f1'] / max(counts['top1_f1'], 1)) * 100:.2f}%" if counts["top1_f1"] else "N/A")
        r4.metric("Max-Normalized F1", f"{(totals['max_norm_f1'] / max(counts['max_norm_f1'], 1)) * 100:.2f}%" if counts["max_norm_f1"] else "N/A")

        r5, r6, r7 = st.columns(3)
        r5.metric("Recall@K", f"{(totals['recall_at_k'] / max(counts['recall_at_k'], 1)) * 100:.2f}%" if counts["recall_at_k"] else "N/A")
        r6.metric("NDCG@K", f"{(totals['ndcg_at_k'] / max(counts['ndcg_at_k'], 1)) * 100:.2f}%" if counts["ndcg_at_k"] else "N/A")
        r7.metric("MRR", f"{(totals['mrr'] / max(counts['mrr'], 1)) * 100:.2f}%" if counts["mrr"] else "N/A")

        st.divider()

        st.subheader("Detailed Results")
        result_df = pd.DataFrame(
            [
                {
                    "Question": x["Question"],
                    "EM": x["EM"],
                    "F1": round(x["F1"], 4),
                    "FActScore": x["FActScore"],
                    "Hit@K": x["Hit@K"],
                    "Recall@K": x["Recall@K"],
                    "NDCG@K": x["NDCG@K"],
                    "MRR": x["MRR"],
                }
                for x in results_data
            ]
        )
        st.dataframe(result_df, use_container_width=True)

        for i, res in enumerate(results_data):
            with st.expander(
                f"Q{i + 1}: {res['Question']} | F1={res['F1']:.3f} | Recall@K={res['Recall@K'] if res['Recall@K'] is not None else 'N/A'}"
            ):
                st.markdown(f"Ground Truth: {res['Ground Truth']}")
                st.markdown(f"Prediction: {res['Prediction']}")
                st.markdown(f"Rewritten Query: {res['Rewritten Query']}")

                st.markdown("Per-item metrics")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("EM", f"{res['EM']:.0f}")
                mcol2.metric("F1", f"{res['F1']:.3f}")
                mcol3.metric("FActScore", f"{res['FActScore']:.0f}")
                mcol4.metric("NDCG@K", f"{res['NDCG@K']:.3f}" if res["NDCG@K"] is not None else "N/A")

                st.markdown("Retrieved contexts")
                for j, ctx in enumerate(res["Contexts"]):
                    st.info(f"Context {j + 1}: {ctx}")
                    with st.popover(f"Context {j + 1} metadata"):
                        st.json(res["Retrieved Meta"][j])