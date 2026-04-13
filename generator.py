import torch
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None

from openai import OpenAI

from generation_utils import (
    build_base_fallback_prompt,
    build_base_prompt,
    build_chat_messages,
    candidate_answer_score,
    clear_memory,
    detect_question_type,
    extract_answer_from_instruct_output,
    extract_final_answer,
    is_abstention,
    prepare_contexts_for_question,
)


class RAGGenerator:
    """
    Local-first generation module that directly integrates the prompt design and
    context-compression logic from the final project notebooks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        generation_mode: str = "base",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        use_4bit: bool = True,
        force_cpu: bool = False,
        device_map: str = "auto",
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 4,
        use_context_compression: bool = True,
        top_k_contexts: int = 3,
        max_total_context_chars: int = 3200,
        max_sentences_per_context: int = 2,
        use_fallback_prompt: bool = True,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.generation_mode = generation_mode
        self.use_api = use_api
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.use_context_compression = use_context_compression
        self.top_k_contexts = top_k_contexts
        self.max_total_context_chars = max_total_context_chars
        self.max_sentences_per_context = max_sentences_per_context
        self.use_fallback_prompt = use_fallback_prompt
        self.force_cpu = force_cpu
        self.device_map = device_map
        self.use_4bit = use_4bit
        self.trust_remote_code = trust_remote_code

        self.client = None
        self.model = None
        self.tokenizer = None

        if self.use_api:
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        else:
            self._load_local_model()

    def _pick_compute_dtype(self):
        if not torch.cuda.is_available() or self.force_cpu:
            return torch.float32
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16

    def _make_quantization_config(self):
        if not self.use_4bit or not torch.cuda.is_available() or self.force_cpu or BitsAndBytesConfig is None:
            return None
        try:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._pick_compute_dtype(),
            )
        except Exception:
            return None

    def _load_local_model(self):
        clear_memory()
        quant_config = self._make_quantization_config()
        compute_dtype = self._pick_compute_dtype()

        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.trust_remote_code,
        }

        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = self.device_map
        elif torch.cuda.is_available() and not self.force_cpu:
            model_kwargs["torch_dtype"] = compute_dtype
            model_kwargs["device_map"] = self.device_map
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def unload(self):
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        self.model = None
        self.tokenizer = None
        clear_memory()

    def get_inference_device(self) -> str:
        if torch.cuda.is_available() and not self.force_cpu:
            return "cuda"
        return "cpu"

    def prepare_contexts(self, question: str, contexts: List[str]) -> List[str]:
        return prepare_contexts_for_question(
            question=question,
            contexts=contexts,
            top_k=self.top_k_contexts,
            max_total_chars=self.max_total_context_chars,
            max_sentences_per_context=self.max_sentences_per_context,
            use_context_compression=self.use_context_compression,
        )

    @torch.inference_mode()
    def _generate_local_plain(self, prompt: str) -> str:
        device = self.get_inference_device()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        generated = self.model.generate(
            **inputs,
            **gen_kwargs,
        )
        output_ids = generated[0][input_len:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def _generate_local_chat(self, messages: List[Dict[str, str]]) -> str:
        device = self.get_inference_device()
        if hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\n\nASSISTANT:"
            inputs = self.tokenizer(prompt, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        generated = self.model.generate(
            **inputs,
            **gen_kwargs,
        )
        output_ids = generated[0][input_len:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _generate_api_chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _run_primary(self, question: str, prepared_contexts: List[str]) -> Dict[str, Any]:
        if self.generation_mode == "base":
            prompt = build_base_prompt(question, prepared_contexts)
            raw = self._generate_api_chat([{"role": "user", "content": prompt}]) if self.use_api else self._generate_local_plain(prompt)
            pred = extract_final_answer(raw, question)
        elif self.generation_mode == "instruct":
            messages = build_chat_messages(question, prepared_contexts, fallback=False)
            raw = self._generate_api_chat(messages) if self.use_api else self._generate_local_chat(messages)
            pred = extract_answer_from_instruct_output(raw, question)
        else:
            raise ValueError("generation_mode must be 'base' or 'instruct'")

        return {
            "raw_output": raw,
            "prediction": pred,
            "heuristic_score": candidate_answer_score(pred, question, prepared_contexts),
            "prompt_variant": "primary",
        }

    def _run_fallback(self, question: str, prepared_contexts: List[str]) -> Dict[str, Any]:
        if self.generation_mode == "base":
            prompt = build_base_fallback_prompt(question, prepared_contexts)
            raw = self._generate_api_chat([{"role": "user", "content": prompt}]) if self.use_api else self._generate_local_plain(prompt)
            pred = extract_final_answer(raw, question)
        else:
            messages = build_chat_messages(question, prepared_contexts, fallback=True)
            raw = self._generate_api_chat(messages) if self.use_api else self._generate_local_chat(messages)
            pred = extract_answer_from_instruct_output(raw, question)

        return {
            "raw_output": raw,
            "prediction": pred,
            "heuristic_score": candidate_answer_score(pred, question, prepared_contexts),
            "prompt_variant": "fallback",
        }

    def generate(self, question: str, contexts: List[str], return_debug: bool = False) -> Any:
        prepared_contexts = self.prepare_contexts(question, contexts)
        chosen = self._run_primary(question, prepared_contexts)

        if self.use_fallback_prompt and (is_abstention(chosen["prediction"]) or chosen["heuristic_score"] < 0.55):
            fallback = self._run_fallback(question, prepared_contexts)
            if fallback["heuristic_score"] >= chosen["heuristic_score"]:
                chosen = fallback

        result = {
            **chosen,
            "prepared_contexts": prepared_contexts,
            "question_type": detect_question_type(question),
            "generation_mode": self.generation_mode,
        }
        return result if return_debug else result["prediction"]
