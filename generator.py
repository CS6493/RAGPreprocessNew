import torch
import re
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from openai import OpenAI  # 用于兼容大部分标准大模型 API 调用
except ImportError:
    OpenAI = None

class RAGGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        use_4bit: bool = True,
        device_map: str = "auto",
        max_tokens: int = 128,
        temperature: float = 0.1
    ):
        """
        初始化生成模块。
        :param use_api: 如果为 True，则不下载模型，而是调用 API（适合跑较大模型或节省显存）。
        :param api_key, api_base_url: 使用 API 时必填，支持兼容 OpenAI SDK 的任意服务商。
        :param use_4bit: 本地加载时是否使用原有逻辑进行 4-bit 量化。
        """
        self.model_name = model_name
        self.use_api = use_api
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_chat_template = "instruct" in self.model_name.lower()
        
        if self.use_api:
            if OpenAI is None:
                raise ImportError("use_api=True 需要安装 openai 库，请先执行: pip install openai")
            print(f"[*] 初始化 API 客户端模式 | 模型: {self.model_name}")
            # 现代的大模型服务（包括自建 vLLM）基本都支持基于 openai 库的调用
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)
            self.model = None
            self.tokenizer = None
        else:
            print(f"[*] 初始化本地权重加载模式 | 模型: {self.model_name}")
            self._load_local_model(use_4bit, device_map)

    def _load_local_model(self, use_4bit: bool, device_map: str):
        """保留原 notebook 中的 4-bit 量化加载逻辑"""
        # 判断设备的混合精度支持能力
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ) if use_4bit and torch.cuda.is_available() else None

        model_kwargs = {"low_cpu_mem_usage": True, "device_map": device_map}
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = compute_dtype
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    # ================= 原有上下文处理与 Prompt 构建逻辑 =================
    
    def detect_question_type(self, question: str) -> str:
        """保留原有的问题类型侦测机制，用于约束回答格式"""
        q = str(question).strip().lower()
        starters = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ", "can ", "could ", "will ", "would ")
        if q.startswith(starters): return "yes_no"
        if q.startswith("who"): return "who"
        if q.startswith("when") or "what year" in q: return "when"
        if q.startswith("where"): return "where"
        if "how many" in q or "how much" in q: return "numeric"
        return "span"

    def join_contexts(self, contexts: List[str]) -> str:
        return "\n\n".join([f"[Document {i}] {ctx}" for i, ctx in enumerate(contexts, start=1)])

    def extract_sources_from_knowledge(
        self,
        knowledge: str,
        max_sources: int = 8,
        max_chars_per_source: int = 1200,
    ) -> List[str]:
        """从离线 knowledge 文本中提取 Sources 段落，转为可喂给生成模型的上下文列表。"""
        if not knowledge:
            return []

        # 仅抓取 -----Sources----- 下方的 csv 代码块内容
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```",
            str(knowledge),
            flags=re.DOTALL,
        )
        if not sources_match:
            return [str(knowledge).strip()[:max_chars_per_source]]

        raw_csv = sources_match.group(1).strip()
        lines = [ln for ln in raw_csv.splitlines() if ln.strip()]
        if len(lines) <= 1:
            return [str(knowledge).strip()[:max_chars_per_source]]

        contexts: List[str] = []
        for line in lines[1:]:  # 跳过表头: id,\tcontent
            if ",\t" in line:
                _, content = line.split(",\t", 1)
            elif "," in line:
                _, content = line.split(",", 1)
            else:
                content = line

            cleaned = content.strip().strip('"')
            if not cleaned:
                continue
            contexts.append(cleaned[:max_chars_per_source])
            if len(contexts) >= max_sources:
                break

        return contexts if contexts else [str(knowledge).strip()[:max_chars_per_source]]

    def build_instruct_messages(self, question: str, contexts: List[str]) -> List[Dict[str, str]]:
        """构建 Instruct 模型标准的 Chat 消息数组"""
        context_block = self.join_contexts(contexts)
        qtype = self.detect_question_type(question)
        
        # 原逻辑：依据问题类型注入不同的格式要求
        if qtype == "yes_no":
            answer_rule = "If the evidence supports yes/no, answer with exactly yes or no."
        else:
            answer_rule = "Return the shortest direct answer phrase, not a long explanation."

        return [
            {
                "role": "system",
                "content": (
                    "You are a careful question answering assistant. "
                    "Use only the retrieved evidence. Do not use outside knowledge. "
                    "If the evidence is insufficient, reply exactly: Insufficient evidence. "
                    + answer_rule
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved evidence:\n{context_block}\n\nReturn a very concise final answer."
            }
        ]

    def build_plain_prompt(self, question: str, contexts: List[str]) -> str:
        """为基础模型构建纯文本提示词，不依赖 chat template。"""
        context_block = self.join_contexts(contexts)
        qtype = self.detect_question_type(question)

        if qtype == "yes_no":
            answer_rule = "If the evidence supports yes/no, answer with exactly yes or no."
        else:
            answer_rule = "Return the shortest direct answer phrase, not a long explanation."

        return (
            "You are a careful question answering assistant. "
            "Use only the retrieved evidence. Do not use outside knowledge. "
            "If the evidence is insufficient, reply exactly: Insufficient evidence. "
            f"{answer_rule}\n\n"
            f"Question: {question}\n\n"
            f"Retrieved evidence:\n{context_block}\n\n"
            "Final answer:"
        )

    def render_local_prompt(self, question: str, contexts: List[str]):
        """根据模型类型渲染本地推理输入。Instruct 模型优先使用 chat template。"""
        messages = self.build_instruct_messages(question=question, contexts=contexts)
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return messages, self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return messages, self.build_plain_prompt(question=question, contexts=contexts)

    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将消息数组转换为适用于基础模型的纯文本提示词。"""
        parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().capitalize()
            content = str(message.get("content", "")).strip()
            if content:
                parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    # ================= 核心生成推断逻辑 (自动分发本地与API) =================

    def generate(self, question: str, contexts: List[str], max_tokens: int = None, temperature: float = None) -> str:
        """接收问题和相关文档列表，生成回答"""
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature

        messages, prompt = self.render_local_prompt(question=question, contexts=contexts)
        
        if self.use_api:
            return self._generate_api(messages, max_tokens, temperature)
        else:
            return self._generate_local(messages, prompt, max_tokens, temperature)

    def generate_from_knowledge(
        self,
        question: str,
        knowledge: str,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """严格使用完整 knowledge 字段作为证据，不做回退或额外替换。"""
        contexts = [str(knowledge).strip()]
        return self.generate(
            question=question,
            contexts=contexts,
            max_tokens=max_tokens,
            temperature=temperature,
        )


    def _generate_api(self, messages, max_tokens, temperature) -> str:
        try:
            # 本地部署的 Qwen2.5-7B（非 Instruct）使用 /v1/completions，输入单字符串 prompt。
            normalized_name = str(self.model_name).strip().lower()
            if normalized_name in {"qwen/qwen2.5-7b", "qwen2.5-7b", "qwen-2.5-7b"}:
                prompt = self.messages_to_prompt(messages)
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return str(response.choices[0].text).strip()

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API Error: {str(e)}"

    @torch.inference_mode()
    def _generate_local(
        self,
        messages: List[Dict[str, str]],
        prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
    ) -> str:
        """保持原有的基于 HuggingFace Transformers 的推理流程"""
        if prompt is None:
            prompt = self.messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # 截断 Prompt，仅获取新生成的答案部分
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()