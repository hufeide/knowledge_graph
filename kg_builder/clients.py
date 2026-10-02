"""Thin OpenAI-compatible clients for chat completion and embeddings.

We use plain ``requests`` (no SDK dependency) so the code works against any
OpenAI-compatible server (llama.cpp, vLLM, Ollama, etc.).
"""
from __future__ import annotations

import threading

import requests

from .config import Config, config
from .utils import extract_json


class LLMClient:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or config
        self._headers = {
            "Authorization": f"Bearer {self.cfg.llm_api_key}",
            "Content-Type": "application/json",
        }
        # Thread-local session: safe for concurrent chunk processing while
        # still reusing a connection pool per worker thread.
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self._headers)
            self._local.session = s
        return s

    @session.setter
    def session(self, value) -> None:
        if not hasattr(self, "_local"):
            self._local = threading.local()
        self._local.session = value

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            return self._post(payload)
        except Exception:
            if json_mode:
                # Some servers reject response_format; retry without it.
                payload.pop("response_format", None)
                return self._post(payload)
            raise

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        text = self.complete(
            prompt, system=system, temperature=temperature, json_mode=True, max_tokens=max_tokens
        )
        return extract_json(text)

    def _post(self, payload: dict) -> str:
        resp = self.session.post(
            self.cfg.llm_host.rstrip("/") + "/chat/completions",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class EmbeddingClient:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or config
        self._headers = {
            "Authorization": f"Bearer {self.cfg.emb_api_key}",
            "Content-Type": "application/json",
        }
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self._headers)
            self._local.session = s
        return s

    @session.setter
    def session(self, value) -> None:
        if not hasattr(self, "_local"):
            self._local = threading.local()
        self._local.session = value

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        resp = self.session.post(
            self.cfg.emb_host.rstrip("/") + "/embeddings",
            json={"model": self.cfg.emb_model, "input": texts},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in data]

    def embed_query(self, text: str):
        return self.embed([text])[0]
