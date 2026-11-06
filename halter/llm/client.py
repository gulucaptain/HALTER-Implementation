"""LLM backends for HALTER's reasoning (paper: Qwen3.5-397B, one model, three prompts).

backend = "api"          OpenAI-compatible cloud API (Alibaba Cloud Model Studio / DashScope, or any provider)
backend = "local_server" Locally deployed model behind an OpenAI-compatible server (vLLM / SGLang),
                         started with scripts/serve_qwen_vllm.sh
backend = "local_hf"     In-process HuggingFace transformers (for small Qwen models / debugging only)
backend = "mock"         Deterministic rule-based stand-in, for tests without any model

All backends expose `chat(messages) -> str`.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)
Messages = List[Dict[str, str]]


@dataclass
class LLMConfig:
    backend: str = "api"
    model: str = "qwen3.5-397b-a17b"
    base_url: Optional[str] = None
    api_key_env: str = "DASHSCOPE_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048
    enable_thinking: bool = False       # Qwen3.5 hybrid thinking; off by default for latency
    timeout_s: float = 120.0
    max_retries: int = 3
    extra_body: Dict = field(default_factory=dict)
    # local_hf only
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"

    @staticmethod
    def from_dict(d: Dict) -> "LLMConfig":
        return LLMConfig(**(d or {}))


class LLMClient(ABC):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.n_calls = 0
        self.total_latency = 0.0

    @abstractmethod
    def _chat(self, messages: Messages) -> str: ...

    def chat(self, messages: Messages) -> str:
        t0 = time.time()
        if not isinstance(self, MockLLMClient):
            messages = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        out = self._chat(messages)
        self.n_calls += 1
        self.total_latency += time.time() - t0
        return out


class OpenAICompatClient(LLMClient):
    """Works for both the DashScope API and a local vLLM/SGLang server (same protocol)."""

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        from openai import OpenAI
        key = os.environ.get(cfg.api_key_env, "EMPTY") if cfg.api_key_env else "EMPTY"
        self.client = OpenAI(base_url=cfg.base_url, api_key=key, timeout=cfg.timeout_s)
        self.is_local = cfg.backend == "local_server"

    def _extra_body(self) -> Dict:
        body = dict(self.cfg.extra_body)
        if self.is_local:
            # vLLM / SGLang: thinking is toggled through the chat template
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = self.cfg.enable_thinking
        else:
            # DashScope: top-level `enable_thinking`
            body.setdefault("enable_thinking", self.cfg.enable_thinking)
        return body

    def _chat(self, messages: Messages) -> str:
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                kw = dict(model=self.cfg.model, messages=messages, temperature=self.cfg.temperature,
                          extra_body=self._extra_body())
                if self.cfg.enable_thinking and not self.is_local:
                    # DashScope thinking mode requires streaming
                    kw.update(stream=True, max_completion_tokens=self.cfg.max_tokens)
                    chunks = []
                    for ev in self.client.chat.completions.create(**kw):
                        if ev.choices and ev.choices[0].delta.content:
                            chunks.append(ev.choices[0].delta.content)
                    return "".join(chunks)
                kw["max_tokens"] = self.cfg.max_tokens
                resp = self.client.chat.completions.create(**kw)
                return resp.choices[0].message.content or ""
            except Exception as e:  # network / rate limit
                last_err = e
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


class HFLocalClient(LLMClient):
    """In-process transformers inference. 397B will not fit on a workstation; use e.g. a small Qwen to debug."""

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=getattr(torch, cfg.torch_dtype), device_map=cfg.device_map).eval()

    def _chat(self, messages: Messages) -> str:
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=self.cfg.enable_thinking)
        except TypeError:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.cfg.max_tokens,
                                      do_sample=self.cfg.temperature > 0,
                                      temperature=max(self.cfg.temperature, 1e-5))
        txt = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in txt:
            txt = txt.split("</think>", 1)[1]
        return txt.strip()


class MockLLMClient(LLMClient):
    """Calls a python function instead of a model. See halter.reasoning.mock_policy for the default."""

    def __init__(self, cfg: LLMConfig, fn: Optional[Callable[[Messages], str]] = None):
        super().__init__(cfg)
        if fn is None:
            from halter.reasoning.mock_policy import mock_llm
            fn = mock_llm
        self.fn = fn

    def _chat(self, messages: Messages) -> str:
        return self.fn(messages)


def make_llm(cfg: LLMConfig, mock_fn=None) -> LLMClient:
    if cfg.backend in ("api", "local_server"):
        return OpenAICompatClient(cfg)
    if cfg.backend == "local_hf":
        return HFLocalClient(cfg)
    if cfg.backend == "mock":
        return MockLLMClient(cfg, mock_fn)
    raise ValueError(f"unknown LLM backend {cfg.backend}")
