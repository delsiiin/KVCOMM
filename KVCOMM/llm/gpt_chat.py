"""Chat backends and local HF model execution (default mode only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import copy
import os
from time import perf_counter
import threading

import async_timeout
from openai import AsyncOpenAI
import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from KVCOMM.llm.config import KVCommConfig
from KVCOMM.llm.format import Message
from KVCOMM.llm.llm import LLM
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.models.monkeypatch import replace_llama, replace_qwen2
from KVCOMM.utils.log import logger
from KVCOMM.utils.metrics import GenerationResult

MINE_API_KEYS = os.getenv("API_KEY")


def _escape_loguru_markup(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.replace("<", "\\<")


def _total_message_chars(messages: List[Dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += len(str(message.get("content", "")))
    return total


class _TTFTTracer(StoppingCriteria):
    """Stopping criteria used to capture TTFT."""

    def __init__(self, prompt_length: int):
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft: Optional[float] = None

    def reset(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if self.ttft is None and input_ids.shape[-1] > self.prompt_length:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.ttft = perf_counter() - self.start_time
        return False


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict[str, str]]) -> str:
    api_kwargs = dict(api_key=MINE_API_KEYS)
    try:
        aclient = AsyncOpenAI(**api_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create the async client: {e}")
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model, messages=msg)
        response_message = completion.choices[0].message.content
        if isinstance(response_message, str):
            return response_message
        return str(response_message)
    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")


@LLMRegistry.register("GPTChat")
class GPTChat(LLM):
    """Thin wrapper around OpenAI-style chat completions."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        normalised = LLMChat._normalise_messages(messages)
        input_char_len = _total_message_chars(normalised)
        response_text = await achat(self.model_name, normalised)
        metadata: Dict[str, Any] = {}
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        metadata["input_char_len"] = input_char_len
        metadata["output_char_len"] = len(response_text) if response_text is not None else 0
        return GenerationResult(text=response_text, mode="default", ttft=0.0, metadata=metadata)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        raise NotImplementedError("GPTChat does not implement sync generation.")


@LLMRegistry.register("LLMChat")
class LLMChat(LLM):
    """Local HF model chat backend for default execution mode."""

    _shared_model = None
    _shared_tokenizer = None
    _shared_compress_mode: Optional[bool] = None
    _shared_compress_method: Optional[str] = None
    _shared_compress_budget: Optional[int] = None
    _shared_compress_divide_length: Optional[int] = None
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_name: str,
        prefix: str = None,
        config: KVCommConfig | None = None,
        compress_mode: bool = False,
        compress_method: str = "rkv",
        compress_budget: int = 1024,
        compress_divide_length: int = 128,
    ):
        self.model_name = model_name
        self.config = (config or KVCommConfig.from_env()).validate()
        self.compress_mode = bool(compress_mode)
        self.compress_method = (compress_method or "rkv").lower().strip()
        self.compress_budget = int(compress_budget)
        self.compress_divide_length = int(compress_divide_length)
        self._kvcomm_call_seq = 0
        self.lock = asyncio.Lock()
        self._initialize_shared_resources()
        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self._chat_markers = self._extract_chat_markers()
        self.default_assistant_prompt = "A: "
        self.base_messages_template: List[Dict[str, str]] = [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        if prefix is not None:
            self._prepare_prefix_template(prefix)

    def _extract_chat_markers(self) -> Dict[str, str]:
        template = getattr(self.tokenizer, "chat_template", "") or ""
        markers = {"begin": "", "start": "", "end": "", "eot": ""}
        for token in ("<|begin_of_text|>", "<s>", getattr(self.tokenizer, "bos_token", "") or ""):
            if token and token in template:
                markers["begin"] = token
                break
        for token in ("<|start_header_id|>", "<|im_start|>"):
            if token and token in template:
                markers["start"] = token
                break
        for token in ("<|end_header_id|>", "<|im_end|>", "\n"):
            if token and token in template:
                markers["end"] = token
                break
        for token in ("<|eot_id|>", "<|im_end|>", getattr(self.tokenizer, "eos_token", "") or ""):
            if token and token in template:
                markers["eot"] = token
                break
        return markers

    def _prepare_prefix_template(self, prefix: Union[str, List[Dict[str, str]]]) -> None:
        if isinstance(prefix, list):
            self.base_messages_template = prefix
            return
        if isinstance(prefix, dict):
            self.base_messages_template = [prefix]
            return
        if isinstance(prefix, str):
            self.default_assistant_prompt = self._extract_assistant_prompt(prefix)
            return
        raise TypeError("Unsupported prefix template type.")

    def _extract_assistant_prompt(self, legacy_prefix: str) -> str:
        start = self.start_header_id
        end = self.end_header_id
        if start and end:
            marker = f"{start}assistant{end}\n"
            if marker in legacy_prefix:
                tail = legacy_prefix.split(marker, 1)[-1]
                eot = self.eot_id
                if eot:
                    tail = tail.replace(eot, "")
                return tail
        return legacy_prefix

    @property
    def begin_of_text(self) -> str:
        return self._chat_markers.get("begin", "")

    @property
    def start_header_id(self) -> str:
        return self._chat_markers.get("start", "")

    @property
    def end_header_id(self) -> str:
        return self._chat_markers.get("end", "")

    @property
    def eot_id(self) -> str:
        return self._chat_markers.get("eot", "")

    @staticmethod
    def _normalise_messages(
        messages: Union[List[Message], List[Dict[str, str]], Dict[str, Any], Tuple[Any, ...], str]
    ) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, tuple):
            if len(messages) == 2 and all(isinstance(item, str) for item in messages):
                return [{"role": "system", "content": messages[0]}, {"role": "user", "content": messages[1]}]
            return LLMChat._normalise_messages(list(messages))
        if isinstance(messages, dict):
            result: List[Dict[str, str]] = []
            system_prompt = messages.get("system") or messages.get("system_prompt")
            if system_prompt:
                result.append({"role": "system", "content": system_prompt})
            if "user" in messages:
                result.append({"role": "user", "content": str(messages["user"])})
            if "assistant" in messages:
                result.append({"role": "assistant", "content": str(messages["assistant"])})
            conversation = messages.get("messages") or messages.get("conversation")
            if conversation is not None:
                result.extend(LLMChat._normalise_messages(conversation))
            return result
        if not isinstance(messages, list):
            raise TypeError("messages must be a string, sequence, or list.")
        normalised: List[Dict[str, str]] = []
        for item in messages:
            if isinstance(item, Message):
                normalised.append({"role": item.role, "content": item.content})
            elif isinstance(item, dict):
                normalised.append({"role": item.get("role", "user"), "content": item.get("content", "")})
            elif isinstance(item, str):
                normalised.append({"role": "user", "content": item})
            else:
                normalised.extend(LLMChat._normalise_messages(item))
        return normalised

    def _legacy_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = [self.begin_of_text or getattr(self.tokenizer, "bos_token", "") or ""]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if self.start_header_id and self.end_header_id:
                prompt_parts.append(f"{self.start_header_id}{role}{self.end_header_id}\n{content}{self.eot_id}")
            else:
                prompt_parts.append(f"[{role.upper()}]\n{content}")
        if self.start_header_id and self.end_header_id:
            prompt_parts.append(f"{self.start_header_id}assistant{self.end_header_id}\n")
        else:
            prompt_parts.append("[ASSISTANT]\n")
        return "".join(prompt_parts)

    def _build_chat_inputs(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        assistant_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], str, int]:
        normalised = self._normalise_messages(messages)
        assistant_prompt = assistant_prompt or self.default_assistant_prompt
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                normalised,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            ) + assistant_prompt
            tokenized = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)
            if isinstance(tokenized, dict):
                inputs = tokenized
            else:
                inputs = {"input_ids": tokenized, "attention_mask": torch.ones_like(tokenized)}
        except (ValueError, AttributeError, NotImplementedError, TypeError):
            prompt_text = self._legacy_prompt_from_messages(normalised) + assistant_prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        return inputs, prompt_text, input_length

    def _render_base_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        rendered: List[Dict[str, str]] = []
        for block in self.base_messages_template:
            rendered.append(
                {
                    "role": block.get("role", "user"),
                    "content": block.get("content", "").format(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    ),
                }
            )
        return rendered

    def build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: Optional[str] = None,
        *,
        add_generation_prompt: bool = True,
        return_messages: bool = False,
    ) -> Dict[str, Any]:
        messages = self._render_base_messages(system_prompt, user_prompt)
        inputs, prompt_text, prompt_length = self._build_chat_inputs(
            messages,
            assistant_prompt=assistant_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        result: Dict[str, Any] = {"inputs": inputs, "prompt_text": prompt_text, "prompt_length": prompt_length}
        if return_messages:
            result["messages"] = messages
        return result

    def set_id(self, node_id: str, role: str):
        self.node_id = node_id
        self.role = role

    def _next_kvcomm_call_seq(self) -> int:
        seq = getattr(self, "_kvcomm_call_seq", 0)
        self._kvcomm_call_seq = seq + 1
        return seq

    def _build_kvcomm_state_key(
        self,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        seq = self._next_kvcomm_call_seq()
        req = request_uid or "sync"
        agent = agent_id or getattr(self, "node_id", "unknown")
        return f"{req}:{agent}:{seq}"

    def _cleanup_kvcomm_state(self, state_key: str) -> None:
        if not state_key:
            return
        seq_states = getattr(self.model, "_kvcomm_seq_states", None)
        if isinstance(seq_states, dict):
            seq_states.pop(state_key, None)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        inputs, _, prompt_length = self._build_chat_inputs(messages)
        kvcomm_state_key = self._build_kvcomm_state_key()
        try:
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                return_legacy_cache=False,
                use_cache=True,
                kvcomm_state_key=kvcomm_state_key,
            )
        finally:
            self._cleanup_kvcomm_state(kvcomm_state_key)
        generated_sequence = outputs.sequences[:, prompt_length:]
        return self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True).strip()

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_cache: Optional[bool] = False,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        ) -> GenerationResult:
        async with self.lock:
            if max_tokens is None:
                max_tokens = self.DEFAULT_MAX_TOKENS
            if temperature is None:
                temperature = self.DEFAULT_TEMPERATURE
            normalised_messages = self._normalise_messages(messages)
            input_char_len = _total_message_chars(normalised_messages)
            inputs, prompt_text, prompt_length = self._build_chat_inputs(normalised_messages)
            logger.opt(colors=True).debug(
                "<blue>[PROMPT]</blue> Agent {} Role {} Prompt:\n{}",
                getattr(self, "node_id", "unknown"),
                getattr(self, "role", "unknown"),
                _escape_loguru_markup(prompt_text),
            )
            generation_kwargs = {
                "do_sample": False,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_dict_in_generate": True,
                "return_legacy_cache": False,
                "use_cache": True,
            }
            kvcomm_state_key = self._build_kvcomm_state_key(
                request_uid=request_uid, agent_id=agent_id
            )
            generation_kwargs["kvcomm_state_key"] = kvcomm_state_key
            ttft_tracer = _TTFTTracer(prompt_length)
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
            ttft_tracer.reset(prompt_length)
            try:
                outputs = self.model.generate(**inputs, **generation_kwargs)
            finally:
                self._cleanup_kvcomm_state(kvcomm_state_key)
            if ttft_tracer.ttft is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_value = 0.0
            else:
                ttft_value = ttft_tracer.ttft
            generated_sequence = outputs.sequences[:, prompt_length:]
            response_message = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True).strip()
            logger.opt(colors=True).debug(
                "<blue>[RESPONSE]</blue> Agent {} Role {} Response:\n{}",
                getattr(self, "node_id", "unknown"),
                getattr(self, "role", "unknown"),
                _escape_loguru_markup(response_message),
            )
            metadata: Dict[str, Any] = {}
            if request_uid:
                metadata["request_uid"] = request_uid
            if agent_id:
                metadata["agent_id"] = agent_id
            if agent_name:
                metadata["agent_name"] = agent_name
            if agent_role:
                metadata["agent_role"] = agent_role
            if return_cache:
                metadata["kv_cache"] = outputs.past_key_values
            metadata["input_char_len"] = input_char_len
            metadata["output_char_len"] = len(response_message) if response_message is not None else 0
            metadata["input_token_len"] = int(prompt_length)
            metadata["output_token_len"] = int(generated_sequence.shape[-1])
            return GenerationResult(text=response_message, mode="default", ttft=ttft_value, metadata=metadata)

    def _build_compression_config(self) -> Dict[str, Any]:
        method = self.compress_method
        supported_methods = {"rkv", "snapkv", "streamingllm", "h2o"}
        if method not in supported_methods:
            raise ValueError(
                f"Unsupported compression method: {self.compress_method}. "
                f"Supported methods: {sorted(supported_methods)}."
            )
        if self.compress_budget <= 0:
            raise ValueError(f"compress_budget must be > 0, got {self.compress_budget}.")
        if self.compress_divide_length <= 0:
            raise ValueError(
                f"compress_divide_length must be > 0, got {self.compress_divide_length}."
            )
        return {
            "method": method,
            "method_config": {
                "budget": self.compress_budget,
                "window_size": 8,
                "kernel_size": 7,
                "mix_lambda": 0.07,
                "retain_ratio": 0.2,
                "retain_direction": "last",
                "first_tokens": 4,
            },
            "update_kv": True,
            "compression": None,
            "compression_content": "all",
            "divide_method": "step_length",
            "divide_length": self.compress_divide_length,
        }

    def _initialize_shared_resources(self):
        with LLMChat._model_lock:
            if LLMChat._shared_model is None:
                if self.compress_mode:
                    model_name_lower = self.model_name.lower()
                    if "llama" in model_name_lower:
                        replace_llama(self._build_compression_config())
                    elif "qwen" in model_name_lower:
                        replace_qwen2(self._build_compression_config())
                    else:
                        logger.warning(
                            "Compression mode is only patched for llama/qwen models. "
                            "Requested model={}, loading without patch.",
                            self.model_name,
                        )
                LLMChat._shared_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                LLMChat._shared_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                LLMChat._shared_compress_mode = self.compress_mode
                LLMChat._shared_compress_method = self.compress_method
                LLMChat._shared_compress_budget = self.compress_budget
                LLMChat._shared_compress_divide_length = self.compress_divide_length
                logger.info("Model {} loaded and shared across instances.", self.model_name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model", None)
        state.pop("tokenizer", None)
        state.pop("lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "_kvcomm_call_seq"):
            self._kvcomm_call_seq = 0
        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self.lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        state = self.__getstate__()
        copied_state = copy.deepcopy(state, memo)
        result.__setstate__(copied_state)
        return result
