from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Sequence
import hashlib
import math
import random
import time

import requests

from repoops.common.config import (
    EMBEDDINGS_PROVIDER,
    EMBEDDINGS_BASE_URL,
    EMBEDDINGS_API_KEY,
    EMBEDDINGS_MODEL,
    EMBEDDINGS_TIMEOUT_S,
    EMBEDDINGS_DIM,
)


@dataclass(frozen=True)
class EmbeddingResult:
    vector: List[float]
    model_id: str
    dim: int
    latency_ms: int


class Embedder:
    """
    Production interface:
      - embed_many: preferred (batch)
      - embed: convenience wrapper
    """

    def embed_many(self, texts: List[str]) -> List[EmbeddingResult]:
        raise NotImplementedError

    def embed(self, text: str) -> EmbeddingResult:
        res = self.embed_many([text])
        return res[0]


class HashEmbedder(Embedder):
    def __init__(self, dim: int):
        self._dim = int(dim)

    def _embed_one(self, text: str) -> EmbeddingResult:
        start_time = time.time()
        vector = [0.0] * self._dim
        words = [word for word in (text or "").split() if word]
        for word in words[:4000]:
            hash_bytes = hashlib.sha256(word.encode("utf-8")).digest()
            idx = int.from_bytes(hash_bytes[:4], "little") % self._dim
            sign = 1.0 if (hash_bytes[4] & 1) == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vector = [value / norm for value in vector]
        return EmbeddingResult(vector, f"hash:{self._dim}", self._dim, int((time.time() - start_time) * 1000))

    def embed_many(self, texts: List[str]) -> List[EmbeddingResult]:
        return [self._embed_one(text) for text in (texts or [])]


class OpenAICompatibleEmbedder(Embedder):
    """
    OpenAI-compatible embeddings endpoint:
      POST {base_url}/embeddings
      payload: {model: str, input: str|[str]}
      response: {data: [{embedding: [...], index?: int}, ...]}
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        timeout_s: int,
        *,
        max_batch_size: int = 128,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.dim = int(dim) if dim else 0
        self.timeout_s = int(timeout_s) if timeout_s else 30
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_retries = max(1, int(max_retries))
        self._session = session  # optional reuse

    def embed_many(self, texts: List[str]) -> List[EmbeddingResult]:
        start_time = time.time()
        texts = list(texts or [])
        if not texts:
            return []

        if not self.base_url:
            raise RuntimeError("EMBEDDINGS_BASE_URL is empty; cannot call embeddings endpoint.")
        if not self.model:
            raise RuntimeError("EMBEDDINGS_MODEL is empty; cannot call embeddings endpoint.")

        # Split into provider-safe chunks
        out: List[EmbeddingResult] = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            out.extend(self._embed_batch(batch, batch_offset=i, overall_t0=start_time))
        return out

    def _embed_batch(self, batch: List[str], *, batch_offset: int, overall_t0: float) -> List[EmbeddingResult]:
        url = self.base_url + "/embeddings"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "repoops-autopilot/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {"model": self.model, "input": batch}

        sess = self._session or requests
        last: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = sess.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if not response.ok:
                    # include tiny body snippet for debugging
                    raise RuntimeError(f"Embeddings HTTP {response.status_code}: {(response.text or '')[:800]}")

                data = response.json()
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"Embeddings error: {str(data.get('error'))[:800]}")

                items = data.get("data") if isinstance(data, dict) else None
                if not isinstance(items, list):
                    raise RuntimeError("Embeddings response missing 'data' list.")

                # Reorder by `index` if present (provider may return out-of-order)
                ordered = self._order_items(items, expected=len(batch))

                results: List[EmbeddingResult] = []
                for idx, item in enumerate(ordered):
                    vec_raw = item.get("embedding") if isinstance(item, dict) else None
                    vec = self._coerce_vector(vec_raw, idx=idx)

                    if self.dim and len(vec) != self.dim:
                        raise RuntimeError(f"Embedding dim mismatch at item {idx}: got {len(vec)} expected {self.dim}")

                    results.append(
                        EmbeddingResult(
                            vector=vec,
                            model_id=f"openai_compatible:{self.model}:{len(vec)}",
                            dim=len(vec),
                            latency_ms=int((time.time() - overall_t0) * 1000),
                        )
                    )

                if len(results) != len(batch):
                    raise RuntimeError(f"Embeddings response size mismatch: got {len(results)} expected {len(batch)}")

                return results

            except Exception as exc:
                last = exc
                # exponential backoff + jitter
                sleep_s = (0.25 * (2 ** attempt)) + random.uniform(0.0, 0.15)
                time.sleep(sleep_s)

        raise RuntimeError(f"Embedding request failed after {self.max_retries} retries: {last}")

    @staticmethod
    def _order_items(items: Sequence[Any], *, expected: int) -> List[Dict[str, Any]]:
        # If items are dicts with "index", reorder into exact input order.
        dict_items = [item for item in items if isinstance(item, dict)]
        if len(dict_items) != len(items):
            # mixed/bad shape, just best-effort cast later
            return [item if isinstance(item, dict) else {} for item in items][:expected]

        has_index = all(("index" in item) for item in dict_items)
        if not has_index:
            return dict_items[:expected]

        # Build array by index
        out: List[Optional[Dict[str, Any]]] = [None] * expected
        for item in dict_items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < expected:
                out[idx] = item

        # If any missing, fall back to original order to avoid None entries
        if any(x is None for x in out):
            return dict_items[:expected]

        return [x for x in out if x is not None]

    @staticmethod
    def _coerce_vector(vec_raw: Any, *, idx: int) -> List[float]:
        if not isinstance(vec_raw, list) or not vec_raw:
            raise RuntimeError(f"Embeddings response missing vector at item {idx}")

        out: List[float] = []
        for j, value in enumerate(vec_raw):
            # Some providers return ints; some floats; some strings (yikes)
            try:
                float_value = float(value)
            except Exception:
                raise RuntimeError(f"Embedding contains non-numeric at item {idx}, pos {j}: {type(value)}")
            if math.isnan(float_value) or math.isinf(float_value):
                raise RuntimeError(f"Embedding contains NaN/Inf at item {idx}, pos {j}")
            out.append(float_value)
        return out


_EMBEDDER: Optional[Embedder] = None
_SESSION: Optional[requests.Session] = None


def get_embedder() -> Embedder:
    global _EMBEDDER, _SESSION
    if _EMBEDDER is not None:
        return _EMBEDDER

    prov = (EMBEDDINGS_PROVIDER or "").strip().lower()
    if prov == "hash":
        _EMBEDDER = HashEmbedder(dim=EMBEDDINGS_DIM)
        return _EMBEDDER

    # Reuse TCP connections (faster, fewer random timeouts)
    _SESSION = requests.Session()

    _EMBEDDER = OpenAICompatibleEmbedder(
        base_url=EMBEDDINGS_BASE_URL,
        api_key=EMBEDDINGS_API_KEY,
        model=EMBEDDINGS_MODEL,
        dim=EMBEDDINGS_DIM,
        timeout_s=EMBEDDINGS_TIMEOUT_S,
        max_batch_size=128,
        max_retries=3,
        session=_SESSION,
    )
    return _EMBEDDER


def embed_text(text: str) -> EmbeddingResult:
    return get_embedder().embed(text)


def embed_many(texts: List[str]) -> List[EmbeddingResult]:
    return get_embedder().embed_many(texts)
