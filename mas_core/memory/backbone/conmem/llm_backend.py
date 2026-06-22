"""
ConMem LLM and Embedding Backend.

Provides API clients for LLM inference and embedding computation,
using OpenAI-compatible API (e.g., SiliconFlow).
"""
import json
import hashlib
import logging
import time
from typing import Optional

import numpy as np
import requests

from utils.stats import stats

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM API client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        retry_count: int = 1,
        timeout: float = 120.0,
        max_input_chars: int = 120000,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_count = retry_count
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.max_tokens = max_tokens
        self.retry_count = retry_count
        self.timeout = timeout

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_input_chars: int = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        if max_input_chars is None:
            max_input_chars = self.max_input_chars
        # Truncate oversized prompts to avoid exceeding the vLLM max-model-len.
        total_len = len(system_prompt or "") + len(user_prompt)
        if total_len > max_input_chars:
            sys_len = len(system_prompt or "")
            sys_budget = min(sys_len, max_input_chars // 4)
            usr_budget = max_input_chars - sys_budget
            if system_prompt and sys_len > sys_budget:
                system_prompt = system_prompt[:sys_budget] + "\n...[truncated]"
            if len(user_prompt) > usr_budget:
                user_prompt = user_prompt[:usr_budget] + "\n...[truncated]"
            logger.warning(f"Prompt truncated to ~{max_input_chars} chars to fit context window")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        for attempt in range(1 + self.retry_count):
            try:
                t0 = time.time()
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                elapsed = time.time() - t0
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # Strip <think>...</think> tags if present (Qwen3 thinking mode)
                content = self._strip_think_tags(content)
                # Record stats
                usage = data.get("usage", {})
                stats.record(
                    source="conmem/llm",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    elapsed=elapsed,
                )
                return content.strip()
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count:
                    time.sleep(1)
                else:
                    logger.error(f"LLM call failed after {1 + self.retry_count} attempts")
                    raise

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from Qwen3 responses."""
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_retries: int = 2,
        max_input_chars: int = None,
    ) -> dict | list:
        """Chat and parse the response as JSON. Extracts from XML tags or code blocks."""
        json_system = (system_prompt or "") + '\n\nRespond with valid JSON only.'

        for attempt in range(max_retries + 1):
            raw = self.chat(
                json_system,
                user_prompt,
                temperature=temperature,
                max_input_chars=max_input_chars,
            )

            import re

            # Priority 1: Extract from XML-style tags (<cards>, <relations>, <merged_card>, etc.)
            tag_match = re.search(r'<(?:cards|relations|merged_card|result)[^>]*>(.*?)(?:</(?:cards|relations|merged_card|result)>|$)', raw, re.DOTALL)
            if tag_match:
                raw = tag_match.group(1).strip()

            # Priority 2: Extract from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if json_match:
                raw = json_match.group(1).strip()

            # Try parsing
            # 1. Direct parse
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            # 2. Extract JSON array
            try:
                array_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if array_match:
                    return json.loads(array_match.group())
            except:
                pass

            # 3. Extract JSON object
            try:
                obj_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if obj_match:
                    return json.loads(obj_match.group())
            except:
                pass

            if attempt < max_retries:
                logger.warning(f"JSON parse failed, retrying {attempt + 1}/{max_retries}...")
                user_prompt += '\n\nIMPORTANT: Respond ONLY with valid JSON inside the appropriate tags.'
            else:
                logger.warning(f"Failed to parse LLM response as JSON after {max_retries + 1} attempts: {raw[:200]}")
                return {}


class EmbeddingClient:
    """OpenAI-compatible Embedding API client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        device: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.timeout = timeout
        self.device = device
        self._cache: dict[str, list[float]] = {}  # text_hash -> embedding
        self._local_model = None

    def _uses_local_sentence_transformer(self) -> bool:
        model_name = (self.model or "").strip()
        return model_name.startswith("sentence-transformers/")

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": self.device} if self.device else {}
            self._local_model = SentenceTransformer(self.model, **kwargs)
        return self._local_model

    def embed(self, text: str) -> list[float]:
        """Compute embedding for a single text (with cache)."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        result = self.embed_batch([text])[0]
        self._cache[key] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of texts."""
        if not texts:
            return []

        if self._uses_local_sentence_transformer():
            model = self._get_local_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings.tolist()

        payload = {
            "model": self.model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/embeddings"

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(vectors: list[list[float]], query: list[float]) -> list[float]:
    """Compute cosine similarities between a query vector and a list of vectors."""
    if not vectors:
        return []
    mat = np.array(vectors)
    q = np.array(query)
    norms = np.linalg.norm(mat, axis=1)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return [0.0] * len(vectors)
    norms[norms == 0] = 1.0
    sims = mat @ q / (norms * q_norm)
    return sims.tolist()


def estimate_tokens(text: str) -> int:
    """Estimate token count using the BPE heuristic: ~4 chars per token."""
    return max(1, len(text) // 4)
