import json
import time
from langsmith import traceable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from repoops.common.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_STRONG_API_KEY,
    OPENAI_STRONG_BASE_URL,
    OPENAI_STRONG_MODEL,
)


@dataclass
class LLMResponse:
    text: str
    model: str
    base_url: str
    used_strong: bool
    latency_ms: int
    attempts: int


class LLMRouter:
    """
    Production-friendly LLM router for OpenAI-compatible endpoints.

    Supports:
    - Base model (often Ollama OpenAI-compatible endpoint)
    - Optional strong model (e.g. GPT-5) for escalation

    Chat Completions shape:
      POST {base_url}/chat/completions
      { model, messages, ... }
    """

    def __init__(self):
        self.base_url = (OPENAI_BASE_URL or "").rstrip("/")
        self.base_key = OPENAI_API_KEY or ""
        self.base_model = OPENAI_MODEL or ""

        self.strong_url = (OPENAI_STRONG_BASE_URL or "").rstrip("/")
        self.strong_key = OPENAI_STRONG_API_KEY or ""
        self.strong_model = OPENAI_STRONG_MODEL or ""

    # -------------------------
    # Config / selection helpers
    # -------------------------

    def llm_enabled(self) -> bool:
        if not self.base_url or not self.base_model:
            return False

        # If using OpenAI hosted API, require key
        if self.base_url.startswith("https://") and "api.openai.com" in self.base_url:
            return bool(self.base_key)

        # Local OpenAI-compatible servers can accept dummy/no key.
        return True

    def strong_enabled(self) -> bool:
        return bool(self.strong_url) and bool(self.strong_model) and bool(self.strong_key)

    def _select_endpoint(self, prefer_strong: bool) -> Tuple[str, str, str, bool]:
        """
        Returns: (base_url, api_key, model, used_strong)
        """
        if prefer_strong and self.strong_enabled():
            return self.strong_url, self.strong_key, self.strong_model, True
        return self.base_url, self.base_key, self.base_model, False

    # -------------------------
    # Main API
    # -------------------------

    @traceable(run_type="llm", name="LLMRouter.chat")
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        prefer_strong: bool = False,
        timeout_s: int = 120,
        max_retries: int = 2,
        force_json: bool = False,
        escalate_on_json_error: bool = True,
    ) -> Dict[str, Any]:
        if not self.llm_enabled():
            raise RuntimeError("LLM not configured: base_url/model (and key if OpenAI) are required.")

        base_url, api_key, model, used_strong = self._select_endpoint(prefer_strong)

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if force_json:
            payload["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        start_time = time.time()
        attempts = 0

        for attempt in range(max_retries + 1):
            attempts += 1
            try:
                text = self._post_chat(url, headers, payload, timeout_s=timeout_s)
                latency_ms = int((time.time() - start_time) * 1000)
                return {
                    "text": text,
                    "model": model,
                    "base_url": base_url,
                    "used_strong": used_strong,
                    "latency_ms": latency_ms,
                    "attempts": attempts,
                }
            except Exception as err:
                last_err = err
                time.sleep(0.4 * (attempt + 1))

        if force_json and escalate_on_json_error and not used_strong and self.strong_enabled():
            strong_url, strong_key, strong_model, unused_flag = self._select_endpoint(True)
            url2 = strong_url.rstrip("/") + "/chat/completions"
            headers2 = {"Content-Type": "application/json", "Authorization": f"Bearer {strong_key}"}

            payload2 = dict(payload)
            payload2["model"] = strong_model
            payload2["response_format"] = {"type": "json_object"}

            attempts += 1
            try:
                text = self._post_chat(url2, headers2, payload2, timeout_s=timeout_s)
                latency_ms = int((time.time() - start_time) * 1000)
                return {
                    "text": text,
                    "model": strong_model,
                    "base_url": strong_url,
                    "used_strong": True,
                    "latency_ms": latency_ms,
                    "attempts": attempts,
                }
            except Exception as err:
                last_err = err

        raise RuntimeError(f"LLM request failed after retries. Last error: {last_err}")

    # -------------------------
    # Low-level HTTP
    # -------------------------

    def _post_chat(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], *, timeout_s: int) -> str:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)

        if not response.ok:
            preview = (response.text or "")[:1200]
            raise RuntimeError(f"LLM HTTP {response.status_code} from {url}: {preview}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            preview = json.dumps(data, ensure_ascii=False)[:1500]
            raise RuntimeError(f"Unexpected LLM response shape: {preview}")

    # -------------------------
    # Output helpers
    # -------------------------

    @staticmethod
    def strip_fences(text: str) -> str:
        import re
        stripped = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", text)
        stripped = re.sub(r"\n```[\s]*$", "\n", stripped)
        return stripped

    @staticmethod
    def extract_json_object(text: str) -> str:
        cleaned = LLMRouter.strip_fences(text).strip()

        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object start '{' found in model output.")

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            char = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif char == "\\":
                    esc = True
                elif char == '"':
                    in_str = False
                continue
            else:
                if char == '"':
                    in_str = True
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return cleaned[start : i + 1]

        raise ValueError("Could not find a complete JSON object in model output.")

    @staticmethod
    def loads_json_object(text: str) -> Dict[str, Any]:
        raw = LLMRouter.extract_json_object(text)
        return json.loads(raw)
