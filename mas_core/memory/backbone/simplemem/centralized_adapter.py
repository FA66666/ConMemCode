"""SimpleMem-style centralized memory adapter.

The baseline/SimpleMem project targets dialogue memory with a three-layer index:
semantic vectors, lexical keywords, and symbolic metadata.  This adapter keeps
that shape but maps MAS task trajectories into local JSON-backed memory entries
so the existing benchmark runners can use it as a baseline backend.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from common.registry import registry
from mas_core.base_centralized_memory import BaseCentralizedMemory, Memory

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


def _clip_text(text: Any, max_chars: int = 12000) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n...[truncated]"


def _safe_filename_part(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:160] or fallback


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _feedback_to_text(feedback: Any) -> str:
    if isinstance(feedback, dict):
        parts = []
        for key in (
            "summary",
            "score",
            "test_passed",
            "answer_correct",
            "extracted_answer",
            "full_feedback",
            "observation",
        ):
            value = feedback.get(key)
            if value not in (None, ""):
                parts.append(f"{key}: {value}")
        return "\n".join(parts)
    return str(feedback or "")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _merge_numeric_stats(target: dict, source: dict) -> dict:
    for key, value in (source or {}).items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
            if isinstance(value, float):
                target[key] = round(target[key], 4)
        elif isinstance(value, dict):
            child = target.setdefault(key, {})
            if isinstance(child, dict):
                _merge_numeric_stats(child, value)
    return target


def _load_json_with_backup(path: str, storage_name: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        directory = os.path.dirname(path) or "."
        basename = os.path.basename(path)
        backup_paths = [f"{path}.bak"]
        if os.path.isdir(directory):
            timestamped = [
                os.path.join(directory, name)
                for name in os.listdir(directory)
                if name.startswith(f"{basename}.bak.")
            ]
            backup_paths.extend(sorted(timestamped, key=os.path.getmtime, reverse=True))

        errors = []
        for backup_path in backup_paths:
            if not os.path.exists(backup_path):
                continue
            try:
                with open(backup_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.warning(
                    "%s JSON at %s is invalid (%s); loaded backup %s instead.",
                    storage_name,
                    path,
                    exc,
                    backup_path,
                )
                return data
            except json.JSONDecodeError as backup_exc:
                errors.append(f"{backup_path}: {backup_exc}")
        raise RuntimeError(
            f"{storage_name} JSON is invalid at {path}: {exc}. "
            "The file is likely truncated or corrupted. Stop any process writing the same bank, then run "
            f"`python scripts/repair_memory_json.py {path}` before retrying."
            + (f" Invalid backups: {'; '.join(errors)}" if errors else "")
        ) from exc


def _atomic_dump_json(data: dict, path: str):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    backup_path = f"{path}.bak"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(path):
            os.replace(path, backup_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class _SimpleMemEmbedder:
    """Embedding backend for semantic SimpleMem retrieval."""

    def __init__(
        self,
        backend: str = "api",
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.backend = (backend or "api").lower()
        self.model = model or os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self.api_base = (api_base or os.getenv("EMBED_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.timeout = timeout
        self.encoder = None
        self.disabled = self.backend == "lexical"
        self.usage_stats: dict = {}

        if self.backend == "auto":
            self.backend = "api" if self.api_base else "sentence_transformers"
        if self.backend == "api" and not self.api_base:
            logger.warning("SimpleMem embedding API base is empty; using lexical retrieval.")
            self.disabled = True
        elif self.backend == "sentence_transformers":
            self._load_sentence_transformer()

    def _load_sentence_transformer(self):
        if SentenceTransformer is None:
            logger.warning("sentence_transformers is not installed; using lexical retrieval.")
            self.disabled = True
            return
        try:
            self.encoder = SentenceTransformer(self.model, trust_remote_code=True)
            logger.info("SimpleMem sentence-transformers embedding loaded: %s", self.model)
        except TypeError:
            self.encoder = SentenceTransformer(self.model)
            logger.info("SimpleMem sentence-transformers embedding loaded: %s", self.model)
        except Exception as exc:
            logger.warning("SimpleMem local embedding load failed; using lexical retrieval: %s", exc)
            self.disabled = True

    def encode(self, text: str, *, is_query: bool = False) -> Optional[list[float]]:
        if self.disabled:
            return None
        if self.backend == "api":
            return self._encode_api(text)
        if self.encoder is not None:
            return self._encode_local(text, is_query=is_query)
        return None

    def _encode_local(self, text: str, *, is_query: bool = False) -> Optional[list[float]]:
        start = time.time()
        success = False
        try:
            kwargs = {"show_progress_bar": False, "normalize_embeddings": True}
            if is_query:
                try:
                    embedding = self.encoder.encode([text], prompt_name="query", **kwargs)[0]
                except Exception:
                    embedding = self.encoder.encode([text], **kwargs)[0]
            else:
                embedding = self.encoder.encode([text], **kwargs)[0]
            values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            success = True
            return [float(x) for x in values]
        except Exception as exc:
            logger.warning("SimpleMem local embedding failed; using lexical retrieval for this text: %s", exc)
            return None
        finally:
            self._record_usage(
                source="simplemem_embedding_local",
                prompt_tokens=_estimate_tokens(text),
                completion_tokens=0,
                elapsed=time.time() - start,
                success=success,
            )

    def _encode_api(self, text: str) -> Optional[list[float]]:
        start = time.time()
        success = False
        prompt_tokens = _estimate_tokens(text)
        try:
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data.get("data"), list) and data["data"]:
                ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
                embedding = ordered[0].get("embedding")
            else:
                embeddings = data.get("embeddings", [])
                embedding = embeddings[0] if embeddings else None
            if not embedding:
                raise ValueError("empty embedding response")
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or prompt_tokens)
            success = True
            return [float(x) for x in embedding]
        except Exception as exc:
            logger.warning("SimpleMem embedding API failed; using lexical retrieval for this text: %s", exc)
            return None
        finally:
            self._record_usage(
                source="simplemem_embedding_api",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                elapsed=time.time() - start,
                success=success,
            )

    def _record_usage(
        self,
        *,
        source: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed: float,
        success: bool,
    ):
        delta = {
            "calls": 1,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
            "time_seconds": round(elapsed, 4),
            "failures": 0 if success else 1,
            "by_source": {
                source: {
                    "calls": 1,
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "time_seconds": round(elapsed, 4),
                    "failures": 0 if success else 1,
                }
            },
        }
        _merge_numeric_stats(self.usage_stats, delta)
        try:
            from utils.stats import stats

            stats.record(source, prompt_tokens, completion_tokens, elapsed, success=success)
        except Exception:
            pass


@dataclass
class SimpleMemEntry:
    entry_id: str
    lossless_restatement: str
    keywords: list[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    location: Optional[str] = None
    persons: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topic: Optional[str] = None
    task_id: str = ""
    task_domain: str = ""
    task_description: str = ""
    outcome: str = ""
    embedding: Optional[list[float]] = None
    access_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "lossless_restatement": self.lossless_restatement,
            "keywords": self.keywords,
            "timestamp": self.timestamp,
            "location": self.location,
            "persons": self.persons,
            "entities": self.entities,
            "topic": self.topic,
            "task_id": self.task_id,
            "task_domain": self.task_domain,
            "task_description": self.task_description,
            "outcome": self.outcome,
            "embedding": self.embedding,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleMemEntry":
        return cls(
            entry_id=str(data.get("entry_id", "")),
            lossless_restatement=str(data.get("lossless_restatement", "")),
            keywords=list(data.get("keywords", []) or []),
            timestamp=data.get("timestamp") or None,
            location=data.get("location") or None,
            persons=list(data.get("persons", []) or []),
            entities=list(data.get("entities", []) or []),
            topic=data.get("topic") or None,
            task_id=str(data.get("task_id", "")),
            task_domain=str(data.get("task_domain", "")),
            task_description=str(data.get("task_description", "")),
            outcome=str(data.get("outcome", "")),
            embedding=data.get("embedding"),
            access_count=int(data.get("access_count", 0) or 0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


class SimpleMemStore:
    """JSON-backed SimpleMem multi-view index."""

    def __init__(self, storage_path: str, embedder: _SimpleMemEmbedder):
        self.storage_path = storage_path
        self.embedder = embedder
        self.entries: dict[str, SimpleMemEntry] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.storage_path):
            return
        data = _load_json_with_backup(self.storage_path, "SimpleMem memory bank")
        self.entries = {}
        for item in data.get("entries", []) or []:
            entry = SimpleMemEntry.from_dict(item)
            if entry.entry_id:
                self.entries[entry.entry_id] = entry

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "storage_type": "simplemem_json_multiview",
            "entries": [entry.to_dict() for entry in self.entries.values()],
        }
        _atomic_dump_json(data, self.storage_path)

    def upsert(self, entry: SimpleMemEntry) -> str:
        existing = self.find_by_task_id(entry.task_id) if entry.task_id else None
        if existing is not None:
            entry.entry_id = existing.entry_id
            entry.access_count = existing.access_count
        entry.embedding = self.embedder.encode(entry.lossless_restatement, is_query=False)
        self.entries[entry.entry_id] = entry
        self.save()
        return entry.entry_id

    def find_by_task_id(self, task_id: str) -> Optional[SimpleMemEntry]:
        for entry in self.entries.values():
            if entry.task_id == task_id:
                return entry
        return None

    def retrieve(self, query: str, top_k: int, update_access: bool = True) -> list[SimpleMemEntry]:
        if not self.entries or top_k <= 0:
            return []

        semantic_scores = self._semantic_scores(query)
        lexical_scores = self._lexical_scores(query)
        structured_scores = self._structured_scores(query)

        entries = list(self.entries.values())
        scored: list[tuple[float, int, SimpleMemEntry]] = []
        for idx, entry in enumerate(entries):
            score = (
                0.60 * semantic_scores.get(entry.entry_id, 0.0)
                + 0.30 * lexical_scores.get(entry.entry_id, 0.0)
                + 0.10 * structured_scores.get(entry.entry_id, 0.0)
            )
            recency = (idx + 1) / max(len(entries), 1) * 0.001
            scored.append((score + recency, idx, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [entry for score, _, entry in scored[:top_k] if score > 0]
        if not results:
            results = list(reversed(entries))[:top_k]

        if update_access:
            for entry in results:
                entry.access_count += 1
            self.save()
        return results

    def _semantic_scores(self, query: str) -> dict[str, float]:
        query_embedding = self.embedder.encode(query, is_query=True)
        if not query_embedding:
            return {}
        scores = {}
        for entry in self.entries.values():
            if entry.embedding:
                scores[entry.entry_id] = max(0.0, _cosine(entry.embedding, query_embedding))
        return scores

    def _lexical_scores(self, query: str) -> dict[str, float]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return {}
        scores = {}
        for entry in self.entries.values():
            entry_text = " ".join(
                [
                    entry.lossless_restatement,
                    entry.task_description,
                    " ".join(entry.keywords),
                    entry.topic or "",
                ]
            )
            entry_tokens = set(_tokenize(entry_text))
            overlap = len(query_tokens.intersection(entry_tokens))
            scores[entry.entry_id] = overlap / max(len(query_tokens), 1) + overlap * 0.01
        return scores

    def _structured_scores(self, query: str) -> dict[str, float]:
        query_tokens = set(_tokenize(query))
        scores = {}
        for entry in self.entries.values():
            symbolic = " ".join(
                [
                    entry.task_domain,
                    entry.outcome,
                    entry.location or "",
                    " ".join(entry.persons),
                    " ".join(entry.entities),
                    entry.topic or "",
                ]
            )
            symbolic_tokens = set(_tokenize(symbolic))
            overlap = len(query_tokens.intersection(symbolic_tokens))
            scores[entry.entry_id] = overlap / max(len(query_tokens), 1)
        return scores

    def size(self) -> int:
        return len(self.entries)


@registry.register_memory("simplemem")
class SimpleMemCentralizedMemory(BaseCentralizedMemory):
    """Bridge SimpleMem text memory into the MAS centralized memory interface."""

    def __init__(
        self,
        storage_dir: str,
        task_domain: Optional[str] = None,
        top_k: int = 5,
        embedding_backend: str = "api",
        embedding_model: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_timeout: float = 60.0,
        read_only: bool = False,
        trajectory_dir: Optional[str] = None,
    ):
        super().__init__()
        self.storage_dir = storage_dir
        self.task_domain = task_domain or "unknown"
        self.top_k = int(top_k)
        self.read_only = read_only
        self.trajectory_dir = trajectory_dir or os.getenv("SIMPLEMEM_TRAJECTORY_DIR", "")
        self.embedding_backend = embedding_backend or os.getenv("SIMPLEMEM_EMBEDDING_BACKEND", "api")
        self.embedding_model = embedding_model or os.getenv("SIMPLEMEM_EMBEDDING_MODEL") or os.getenv("EMBED_MODEL", "")
        self.embedding_api_base = (
            embedding_api_base
            or os.getenv("SIMPLEMEM_EMBEDDING_API_BASE")
            or os.getenv("EMBED_BASE_URL", "")
        )
        self.embedding_api_key = (
            embedding_api_key
            or os.getenv("SIMPLEMEM_EMBEDDING_API_KEY")
            or os.getenv("EMBED_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.embedder = _SimpleMemEmbedder(
            backend=self.embedding_backend,
            model=self.embedding_model,
            api_base=self.embedding_api_base,
            api_key=self.embedding_api_key,
            timeout=embedding_timeout,
        )
        self.store_path = os.path.join(self.storage_dir, "simplemem_memory.json")
        self.store = SimpleMemStore(self.store_path, self.embedder)
        self._agent_roles: dict[str, str] = {}
        self._task_retrieval_count = 0
        self._saved_trajectory_files: list[str] = []
        self._runtime_context: dict[str, Optional[str]] = {}

        # Compatibility with existing evaluation logging that expects
        # `conmem.storage.count_active_cards()`.
        self.storage = self

    def register_agents(self, agents_list: list):
        super().register_agents(agents_list)
        for agent in agents_list:
            agent_id = getattr(agent, "id", None) or str(id(agent))
            self._agent_roles[agent_id] = getattr(agent, "role", None) or "default"

    def _agent_role(self, agent=None) -> str:
        if agent is None:
            return "default"
        agent_id = getattr(agent, "id", None) or str(id(agent))
        return self._agent_roles.get(agent_id) or getattr(agent, "role", None) or "default"

    def retrieve_memory(self, task_description: str, agent=None) -> Memory:
        if self.count_active_cards() <= 0:
            return Memory(text_memory=None, extra_fields={"retrieved_count": 0})

        agent_role = self._agent_role(agent)
        query = (
            f"Task domain: {self.task_domain}\n"
            f"Agent role: {agent_role}\n"
            f"Task:\n{task_description or ''}"
        )
        entries = self.store.retrieve(query, self.top_k, update_access=not self.read_only)
        self._task_retrieval_count += len(entries)
        return Memory(
            text_memory=self._format_entries(entries) or None,
            extra_fields={
                "retrieved_count": len(entries),
                "entry_ids": [entry.entry_id for entry in entries],
            },
        )

    def add_memory(
        self,
        trajectory=None,
        task_id: str = "",
        task_description: str = "",
        outcome: str = "success",
        **kwargs,
    ):
        if self.read_only or trajectory is None:
            return None

        task_description = task_description or self._extract_task_description(trajectory)
        steps = self._trajectory_to_steps(trajectory, task_description)
        trajectory_path = self._save_trajectory_record(
            task_id=task_id,
            task_description=task_description,
            outcome=outcome,
            steps=steps,
            source_trajectory=trajectory,
        )
        restatement = self._build_lossless_restatement(
            task_id=task_id,
            task_description=task_description,
            outcome=outcome,
            steps=steps,
            trajectory=trajectory,
        )
        keywords = self._extract_keywords(task_description, restatement)
        entry = SimpleMemEntry(
            entry_id=str(uuid.uuid4()),
            lossless_restatement=restatement,
            keywords=keywords,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            location=None,
            persons=[],
            entities=self._extract_entities(task_description, restatement),
            topic=self._topic_from_task(task_description),
            task_id=task_id,
            task_domain=self.task_domain,
            task_description=task_description,
            outcome=outcome,
            metadata={
                "source": "baseline/SimpleMem",
                "trajectory_file": trajectory_path,
                "step_count": len(steps),
                "runtime_context": copy.deepcopy(self._runtime_context),
            },
        )
        entry_id = self.store.upsert(entry)
        return {"entry_id": entry_id, "trajectory_file": trajectory_path}

    def _extract_task_description(self, trajectory: Any) -> str:
        if isinstance(trajectory, dict):
            return str(trajectory.get("task_description", "") or "")
        return str(getattr(trajectory, "task_init_description", "") or "")

    def _trajectory_to_steps(self, trajectory: Any, task_description: str) -> list[dict]:
        if isinstance(trajectory, dict):
            raw_steps = trajectory.get("steps") or []
            return [self._normalize_step(step, idx) for idx, step in enumerate(raw_steps, 1)]

        steps = []
        for idx, msg_graph in enumerate(getattr(trajectory, "trajectory", []) or [], 1):
            steps.append(
                {
                    "step_index": idx,
                    "agent": "mas",
                    "input": getattr(trajectory, "task_init_description", "") or task_description,
                    "output": getattr(msg_graph, "action", "") or "",
                    "feedback": getattr(msg_graph, "observation", "") or "",
                }
            )
        return steps

    def _normalize_step(self, step: Any, idx: int) -> dict:
        if not isinstance(step, dict):
            return {
                "step_index": idx,
                "agent": "agent",
                "input": "",
                "output": str(step),
                "tool_calls": "",
                "feedback": "",
            }
        return {
            "step_index": step.get("step_index", idx),
            "agent": str(step.get("agent", "agent")),
            "input": str(step.get("input", "") or ""),
            "output": str(step.get("output", "") or ""),
            "tool_calls": str(step.get("tool_calls", "") or ""),
            "feedback": step.get("feedback", ""),
        }

    def _build_lossless_restatement(
        self,
        *,
        task_id: str,
        task_description: str,
        outcome: str,
        steps: list[dict],
        trajectory: Any,
    ) -> str:
        lines = [
            f"SimpleMem task memory for domain {self.task_domain}.",
            f"Task id {task_id} ended with outcome {outcome}.",
            f"Task description: {_clip_text(task_description, 3000)}",
        ]
        for step in steps[:8]:
            feedback = _feedback_to_text(step.get("feedback", ""))
            lines.append(
                " ".join(
                    [
                        f"Step {step.get('step_index')} agent {step.get('agent', 'agent')}.",
                        f"Input: {_clip_text(step.get('input', ''), 1600)}",
                        f"Output: {_clip_text(step.get('output', ''), 2600)}",
                    ]
                )
            )
            if feedback:
                lines.append(f"Feedback: {_clip_text(feedback, 2200)}")
        if isinstance(trajectory, dict):
            for key in ("test_code", "infrastructure_failure"):
                if key in trajectory:
                    lines.append(f"{key}: {_clip_text(trajectory.get(key), 2200)}")
        return "\n".join(lines)

    def _format_entries(self, entries: list[SimpleMemEntry]) -> str:
        if not entries:
            return ""
        blocks = ["[SimpleMem Retrieved Memories]"]
        for idx, entry in enumerate(entries, 1):
            lines = [
                f"{idx}. entry_id={entry.entry_id} task_id={entry.task_id} outcome={entry.outcome}",
                f"Topic: {entry.topic or 'unknown'}",
                f"Keywords: {', '.join(entry.keywords[:12])}",
                f"Restatement: {_clip_text(entry.lossless_restatement, 5000)}",
            ]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _extract_keywords(self, task_description: str, content: str) -> list[str]:
        tokens = _tokenize(f"{task_description} {content}")
        stop = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "you",
            "are",
            "task",
            "input",
            "output",
            "score",
            "agent",
            "simplemem",
            "memory",
            "domain",
            "ended",
            "outcome",
        }
        counts: dict[str, int] = {}
        for token in tokens:
            if len(token) < 3 or token in stop:
                continue
            counts[token] = counts.get(token, 0) + 1
        return [token for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:30]]

    def _extract_entities(self, task_description: str, content: str) -> list[str]:
        values = re.findall(r"\b[A-Z][A-Za-z0-9_/-]{2,}\b", f"{task_description} {content}")
        seen = set()
        entities = []
        for value in values:
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            entities.append(value)
            if len(entities) >= 20:
                break
        return entities

    def _topic_from_task(self, task_description: str) -> str:
        tokens = self._extract_keywords(task_description, task_description)
        return " ".join(tokens[:6]) if tokens else self.task_domain

    def _save_trajectory_record(
        self,
        *,
        task_id: str,
        task_description: str,
        outcome: str,
        steps: list[dict],
        source_trajectory: Any,
    ) -> Optional[str]:
        if not self.trajectory_dir:
            return None

        domain = _safe_filename_part(self.task_domain, "unknown_domain")
        safe_task_id = _safe_filename_part(task_id, "unknown_task")
        out_dir = os.path.join(self.trajectory_dir, domain)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{safe_task_id}.json")
        record = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memory_backend": "simplemem",
            "task_domain": self.task_domain,
            "task_id": task_id,
            "task_description": task_description,
            "outcome": outcome,
            "step_count": len(steps),
            "steps": _jsonable(steps),
            "source_trajectory": _jsonable(source_trajectory),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            self._saved_trajectory_files.append(path)
            return path
        except Exception as exc:
            logger.warning("Failed to save SimpleMem trajectory record %s: %s", path, exc)
            return None

    def get_and_reset_retrieval_count(self) -> int:
        count = self._task_retrieval_count
        self._task_retrieval_count = 0
        return count

    def get_saved_trajectory_files(self) -> list[str]:
        return list(self._saved_trajectory_files)

    def get_usage_stats(self) -> dict:
        return copy.deepcopy(self.embedder.usage_stats)

    def count_active_cards(self) -> int:
        return self.store.size()

    def get_current_round(self) -> int:
        return 0

    def set_runtime_context(
        self,
        model_name: Optional[str] = None,
        mas_architecture: Optional[str] = None,
    ):
        self._runtime_context = {
            "model_name": model_name,
            "mas_architecture": mas_architecture,
        }

    def process_memory(
        self,
        text_memory: str,
        task_description: str = "",
        extra_fields: dict = None,
        agent=None,
        **kwargs,
    ) -> Memory:
        return Memory(text_memory=text_memory, extra_fields=extra_fields or {})

    @classmethod
    def from_config(cls, config: dict, working_dir: str) -> "SimpleMemCentralizedMemory":
        storage_dir = config.get("storage_dir") or config.get("simplemem_storage_dir")
        if not storage_dir:
            storage_dir = os.path.join(working_dir, "simplemem_memory")
        return cls(
            storage_dir=storage_dir,
            task_domain=config.get("task_domain"),
            top_k=int(config.get("top_k", config.get("simplemem_top_k", 5))),
            embedding_backend=config.get("embedding_backend", config.get("simplemem_embedding_backend", "api")),
            embedding_model=config.get("embedding_model", config.get("simplemem_embedding_model")),
            embedding_api_base=config.get("embedding_api_base", config.get("simplemem_embedding_api_base")),
            embedding_api_key=config.get("embedding_api_key", config.get("simplemem_embedding_api_key")),
            embedding_timeout=float(config.get("embedding_timeout", config.get("simplemem_embedding_timeout", 60.0))),
            read_only=bool(config.get("read_only", False)),
            trajectory_dir=config.get("trajectory_dir", config.get("simplemem_trajectory_dir")),
        )
