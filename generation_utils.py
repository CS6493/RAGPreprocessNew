import gc
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional

STOPWORDS = {
    "a","an","the","of","in","on","at","to","for","from","by","with","and","or","but","as",
    "is","are","was","were","be","been","being","do","does","did","have","has","had",
    "what","which","who","whom","when","where","why","how","both","same","that","this",
    "these","those","it","its","their","his","her","into","about","than","then","also"
}

INSTRUCT_SYSTEM_PROMPT = """You are a careful retrieval-grounded question answering assistant.
Answer using only the retrieved evidence provided by the user.
Do not rely on outside knowledge.
If the evidence is insufficient, answer exactly: Insufficient evidence.
Keep the final answer as short as possible while still correct."""


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_text(text: Any) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_answer_for_ynm(text: Any) -> str:
    t = normalize_text(text)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    if re.search(r"\bmaybe\b", t):
        return "maybe"
    return t


def exact_match_score(pred: Any, gold: Any) -> float:
    return float(normalize_answer_for_ynm(pred) == normalize_answer_for_ynm(gold))


def token_f1_score(pred: Any, gold: Any) -> float:
    pred_tokens = normalize_answer_for_ynm(pred).split()
    gold_tokens = normalize_answer_for_ynm(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def tokenize_simple(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def question_keywords(question: str) -> List[str]:
    return [t for t in tokenize_simple(question) if t not in STOPWORDS and len(t) > 1]


def is_yesno_question(question: str) -> bool:
    q = str(question).strip().lower()
    starters = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ",
                "had ", "can ", "could ", "will ", "would ")
    return q.startswith(starters)


def detect_question_type(question: str) -> str:
    q = str(question).strip().lower()
    if is_yesno_question(q):
        return "yes_no"
    if q.startswith("who"):
        return "who"
    if q.startswith("when") or "what year" in q:
        return "when"
    if q.startswith("where"):
        return "where"
    if "how many" in q or "how much" in q:
        return "numeric"
    return "span"


def context_to_text(ctx: Any) -> str:
    if isinstance(ctx, str):
        return ctx.strip()
    if isinstance(ctx, dict):
        title = str(ctx.get("title", "")).strip()
        text = str(ctx.get("text", ctx.get("page_content", ctx.get("evidence_text", ctx.get("chunk_text", ""))))).strip()
        if title and text:
            return f"[{title}] {text}"
        return (title or text).strip()
    return str(ctx).strip()


def contexts_to_list(contexts: Any, top_k: int = 3) -> List[str]:
    out = []
    for c in safe_list(contexts):
        s = context_to_text(c)
        if s:
            out.append(s)
    return out[:top_k]


def join_contexts(contexts: List[str]) -> str:
    return "\n\n".join([f"[Document {i}] {ctx}" for i, ctx in enumerate(contexts, start=1)])


def is_abstention(text: Any) -> bool:
    t = str(text).lower()
    triggers = [
        "insufficient evidence",
        "not enough information",
        "cannot answer",
        "can't answer",
        "unknown",
        "not provided in the context",
        "not enough evidence",
        "insufficient information",
    ]
    return any(x in t for x in triggers)


def prediction_in_context(pred: Any, contexts: Any) -> float:
    pred_n = normalize_text(pred)
    ctx_n = normalize_text(" ".join(contexts_to_list(contexts, top_k=999)))
    if not pred_n:
        return 0.0
    return float(pred_n in ctx_n)


def clear_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def sentence_split(text: str) -> List[str]:
    text = str(text).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?!;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def unique_preserve_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        nx = normalize_text(x)
        if nx and nx not in seen:
            out.append(x)
            seen.add(nx)
    return out


def lexical_overlap_count(question: str, text: str) -> int:
    return len(set(question_keywords(question)) & set(tokenize_simple(text)))


def sentence_score(question: str, sentence: str) -> float:
    qtype = detect_question_type(question)
    qset = set(question_keywords(question))
    tset = set(tokenize_simple(sentence))

    score = 3.0 * len(qset & tset)
    if qtype in {"when", "numeric"} and re.search(r"\d", sentence):
        score += 2.0
    if qtype == "who" and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", sentence):
        score += 1.0
    if qtype == "where" and any(tok in tset for tok in ["city", "state", "country", "located", "born", "based"]):
        score += 1.0
    if qtype == "yes_no" and any(tok in tset for tok in ["american", "british", "same", "different", "located", "united", "states"]):
        score += 1.0
    score -= 0.003 * max(0, len(sentence) - 220)
    return score


def compress_single_context(question: str, context: str, max_sentences: int = 2) -> str:
    sentences = sentence_split(context)
    if not sentences:
        return context.strip()

    scored = [(idx, s, sentence_score(question, s)) for idx, s in enumerate(sentences)]
    scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)

    kept = [x for x in scored_sorted if x[2] > 0][:max_sentences]
    if not kept:
        kept = scored[:1]

    kept = sorted(kept, key=lambda x: x[0])
    out = " ".join([x[1] for x in kept]).strip()
    return out if out else context.strip()


def prepare_contexts_for_question(
    question: str,
    contexts: Any,
    top_k: int = 3,
    max_total_chars: int = 3200,
    max_sentences_per_context: int = 2,
    use_context_compression: bool = True,
) -> List[str]:
    cleaned = []
    for ctx in contexts_to_list(contexts, top_k=999):
        ctx = re.sub(r"\s+", " ", ctx).strip()
        if ctx:
            cleaned.append(ctx)

    cleaned = unique_preserve_order(cleaned)

    if use_context_compression:
        cleaned = [
            compress_single_context(question, ctx, max_sentences=max_sentences_per_context)
            for ctx in cleaned
        ]

    scored = []
    for ctx in cleaned:
        score = sentence_score(question, ctx) + 0.1 * lexical_overlap_count(question, ctx)
        scored.append((score, ctx))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    total_chars = 0
    for _, ctx in scored:
        if len(out) >= top_k:
            break
        if total_chars + len(ctx) > max_total_chars and out:
            continue
        out.append(ctx)
        total_chars += len(ctx)

    if not out:
        out = cleaned[:top_k]
    return out[:top_k]


def extract_final_answer(raw_text: str, question: str) -> str:
    text = str(raw_text).strip()

    m = re.search(r"final answer\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()

    text = re.split(r"\n(?:evidence|reasoning)\s*:", text, flags=re.IGNORECASE)[0].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tagged = [ln for ln in lines if re.match(r"^(answer|final answer)\s*[:\-]", ln, flags=re.I)]
        text = tagged[-1] if tagged else lines[-1]

    text = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" '\"`*")

    if is_abstention(text):
        return "Insufficient evidence."

    qtype = detect_question_type(question)
    norm_yn = normalize_answer_for_ynm(text)
    if qtype == "yes_no" and norm_yn in {"yes", "no", "maybe"}:
        return norm_yn

    if len(text.split()) > 14:
        text = re.split(r"\s+(?:because|since|which|that)\s+", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        text = re.split(r"[.;]\s*", text, maxsplit=1)[0].strip()

    return text if text else "Insufficient evidence."


def extract_answer_from_instruct_output(text: str, question: Optional[str] = None) -> str:
    text = (text or "").strip()
    if not text:
        return "Insufficient evidence."

    lowered = text.lower()
    if "final answer:" in lowered:
        idx = lowered.rfind("final answer:")
        text = text[idx + len("final answer:"):].strip().splitlines()[0].strip()

    else:
        text = text.splitlines()[0].strip()

    if question:
        return extract_final_answer(text, question)
    return text or "Insufficient evidence."


def candidate_answer_score(answer: str, question: str, contexts: Any) -> float:
    score = 0.0
    toks = tokenize_simple(answer)
    qtype = detect_question_type(question)

    score += 4.0 if not is_abstention(answer) else -1.0
    if qtype == "yes_no" and normalize_answer_for_ynm(answer) in {"yes", "no", "maybe"}:
        score += 3.0

    if 1 <= len(toks) <= 8:
        score += 1.5
    elif len(toks) > 16:
        score -= 1.0

    score += 1.5 * prediction_in_context(answer, contexts)
    return score


def build_base_prompt(question: str, contexts: List[str]) -> str:
    context_block = join_contexts(contexts)
    qtype = detect_question_type(question)
    if qtype == "yes_no":
        answer_rule = "If the evidence supports yes/no, output exactly yes or no."
    else:
        answer_rule = "Output the shortest possible answer phrase, not a full explanatory sentence."

    return f"""You are a careful question answering system.
Use only the retrieved evidence below.

Question type: {qtype}
Question: {question}

Retrieved evidence:
{context_block}

Rules:
1. Use only the evidence above.
2. If the evidence is not sufficient, output exactly: Insufficient evidence.
3. {answer_rule}
4. Do not add background knowledge.
5. Keep the final answer extremely concise.

Output format:
Evidence: <brief supporting phrase(s) or none>
Final answer: <answer>"""


def build_base_fallback_prompt(question: str, contexts: List[str]) -> str:
    context_block = join_contexts(contexts)
    qtype = detect_question_type(question)
    final_line = "Final answer: yes or no or Insufficient evidence." if qtype == "yes_no" else \
        "Final answer: the shortest answer phrase possible, or Insufficient evidence."

    return f"""Answer strictly from the evidence.

Question: {question}

Evidence:
{context_block}

Return only two lines:
Evidence: <minimal supporting span(s) or none>
{final_line}"""


def build_instruct_user_prompt(question: str, contexts: List[str]) -> str:
    context_block = join_contexts(contexts)
    qtype = detect_question_type(question)

    if qtype == "yes_no":
        answer_rule = "If the evidence supports a binary answer, output exactly yes or no."
    else:
        answer_rule = "Output the shortest possible answer phrase, not a full explanatory sentence."

    return f"""Question type: {qtype}
Question: {question}

Retrieved evidence:
{context_block}

Rules:
1. Use only the evidence above.
2. If the evidence is not sufficient, output exactly: Insufficient evidence.
3. {answer_rule}
4. Do not add background knowledge.
5. Keep the final answer extremely concise.

Return exactly two lines:
Evidence: <brief supporting phrase(s) or none>
Final answer: <answer>"""


def build_instruct_fallback_user_prompt(question: str, contexts: List[str]) -> str:
    context_block = join_contexts(contexts)
    qtype = detect_question_type(question)
    final_line = "Final answer: yes or no or Insufficient evidence." if qtype == "yes_no" else \
        "Final answer: the shortest answer phrase possible, or Insufficient evidence."

    return f"""Answer strictly from the evidence.

Question: {question}

Evidence:
{context_block}

Return only two lines:
Evidence: <minimal supporting span(s) or none>
{final_line}"""


def build_chat_messages(question: str, contexts: List[str], fallback: bool = False) -> List[Dict[str, str]]:
    user_prompt = build_instruct_fallback_user_prompt(question, contexts) if fallback else build_instruct_user_prompt(question, contexts)
    return [
        {"role": "system", "content": INSTRUCT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
