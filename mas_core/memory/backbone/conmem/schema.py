"""
ConMem data schemas.

The methodology defines a card as a unified strategy unit that carries
state/plan/exec/eval content together with trigger semantics. This module keeps
that representation as the source of truth while preserving legacy
`memory_type`/`content` access for older callers and tests.
"""
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import json
import re
import time
import uuid

import numpy as np

SUMMARY_SECTION_PREVIEW_CHARS = 220
SUMMARY_EVIDENCE_PREVIEW_CHARS = 160


def _serialize_embedding_blob(emb: Any) -> Optional[bytes]:
    """Serialize an embedding vector for storage as a BLOB (L2-normalized float32 bytes).

    Accepts list[float], np.ndarray, or existing bytes (pass-through).
    Returns None for None input.
    """
    if emb is None:
        return None
    if isinstance(emb, (bytes, bytearray, memoryview)):
        return bytes(emb)
    arr = np.asarray(emb, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


def _serialize_embedding_json(emb: Any) -> Optional[str]:
    """Serialize an embedding vector as legacy JSON text."""
    if emb is None:
        return None
    if isinstance(emb, (bytes, bytearray, memoryview)):
        emb = _deserialize_embedding_blob(emb)
    arr = np.asarray(emb, dtype=float)
    return json.dumps(arr.tolist())


def _deserialize_embedding_blob(raw: Any) -> Optional[Any]:
    """Deserialize storage value into an embedding.

    BLOB (bytes) → np.ndarray[float32]. Legacy TEXT/JSON → list[float].
    None or empty → None.
    """
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buf = bytes(raw)
        if not buf:
            return None
        return np.frombuffer(buf, dtype=np.float32).copy()
    # Legacy text row: JSON-encoded list of floats.
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def configure_schema_preview(
    *,
    section_chars: int | None = None,
    evidence_chars: int | None = None,
):
    global SUMMARY_SECTION_PREVIEW_CHARS, SUMMARY_EVIDENCE_PREVIEW_CHARS
    if section_chars is not None:
        SUMMARY_SECTION_PREVIEW_CHARS = max(1, int(section_chars))
    if evidence_chars is not None:
        SUMMARY_EVIDENCE_PREVIEW_CHARS = max(1, int(evidence_chars))


class MemoryType(str, Enum):
    STATE = "state"
    PLAN = "plan"
    EXEC = "exec"
    EVAL = "eval"


class LifecycleState(str, Enum):
    ACTIVE = "active"


class TaskOutcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class EdgeRelation(str, Enum):
    SUPPORTS = "supports"
    CONSTRAINS = "constrains"
    SATISFIES = "satisfies"
    CONFLICTS = "conflicts"


SECTION_ORDER = (
    MemoryType.STATE.value,
    MemoryType.PLAN.value,
    MemoryType.EXEC.value,
    MemoryType.EVAL.value,
)

PATTERN_STOPWORDS = {
    "about", "after", "again", "algorithm", "all", "also", "approach", "array",
    "based", "before", "being", "build", "calculate", "card", "case", "check",
    "code", "constraint", "constraints", "content", "current", "data", "default",
    "edge", "element", "elements", "ensure", "evaluate", "execution", "failed",
    "failure", "feedback", "first", "following", "function", "general",
    "generate", "given", "handle", "implementation", "input", "inputs", "list",
    "maintain", "method", "must", "need", "numbers", "optional", "output",
    "outputs", "pattern", "plan", "precondition", "preconditions", "problem",
    "process", "reusable", "return", "section", "solve", "state", "step",
    "steps", "strategy", "string", "summary", "task", "that", "their", "then",
    "there", "these", "this", "through", "type", "using", "valid", "value",
    "where", "with", "without",
}

PATTERN_FAMILY_KEYWORDS = {
    "array": ("array", "subarray", "subsequence", "list", "vector"),
    "string": ("string", "substring", "character", "characters", "text", "token"),
    "matrix": ("matrix", "grid", "row", "column", "2d"),
    "graph": ("graph", "node", "edge", "path", "vertex", "adjacency"),
    "tree": ("tree", "binary tree", "bst", "trie"),
    "interval": ("interval", "meeting", "schedule", "cooldown", "session"),
    "file_io": ("file", "csv", "json", "path", "directory", "filesystem"),
    "datetime": ("date", "datetime", "timestamp", "calendar", "time"),
    "dataframe": ("pandas", "dataframe", "series", "seaborn", "matplotlib"),
    "math": ("polygon", "prime", "factor", "equation", "arithmetic", "geometry"),
}

PATTERN_METHOD_KEYWORDS = {
    "sliding_window": ("sliding window", "two pointer", "two-pointer"),
    "hash_map": ("hash map", "dictionary", "dict", "mapping", "frequency map"),
    "hash_set": ("set", "hash set", "deduplicate", "unique elements"),
    "counting": ("count frequency", "counting", "frequency", "counter"),
    "prefix_sum": ("prefix sum", "running sum", "cumulative sum"),
    "binary_search": ("binary search", "search on answer"),
    "sorting": ("sort", "sorted", "sorting"),
    "stack": ("stack", "monotonic stack"),
    "queue": ("queue", "deque"),
    "heap": ("heap", "priority queue", "min-heap", "max-heap"),
    "bfs": ("bfs", "breadth first search"),
    "dfs": ("dfs", "depth first search"),
    "dynamic_programming": ("dynamic programming", "dp", "memoization", "tabulation"),
    "recursion": ("recursive", "recursion"),
    "backtracking": ("backtracking", "backtrack"),
    "greedy": ("greedy",),
    "parsing": ("parse", "parsing", "tokenize", "expression", "decode"),
    "regex": ("regex", "regular expression", "pattern matching"),
    "in_place_swap": ("in-place", "swap into correct position"),
}

PATTERN_CONTRACT_KEYWORDS = {
    "case_insensitive": ("case-insensitive", "ignore case", "lowercase both"),
    "sorted_output": ("sorted output", "ascending order", "sort the result"),
    "preserve_order": ("preserve order", "maintain order"),
    "returns_minus_one": ("return -1", "returns -1"),
    "raises_valueerror": ("raise valueerror", "raises valueerror", "valueerror"),
    "raises_typeerror": ("raise typeerror", "raises typeerror", "typeerror"),
    "in_place": ("in-place", "modify the input", "mutate the input"),
    "returns_copy": ("return a new list", "return a new array", "new matrix"),
    "o_n": ("o(n)", "linear time"),
    "o_1_space": ("o(1)", "constant space"),
    "single_loop": ("single loop", "one loop"),
}

PATTERN_LIBRARY_KEYWORDS = {
    "collections": ("deque", "defaultdict", "counter", "collections"),
    "typing": ("optional", "typing", "list[", "tuple[", "dict["),
    "pandas": ("pandas", "dataframe", "series"),
    "numpy": ("numpy",),
    "datetime": ("datetime", "strptime"),
    "csv": ("csv",),
    "json": ("json",),
    "tokenize": ("tokenize",),
    "heapq": ("heapq",),
    "bisect": ("bisect",),
    "re": ("regex", "re.", "re.compile"),
}


@dataclass
class Provenance:
    """Where the memory card came from."""

    source_task_id: str = ""
    source_agent: str = ""
    source_step_indices: list = field(default_factory=list)
    trajectory_outcome: str = ""
    reflection_quality: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Provenance":
        return cls(**data)


@dataclass
class CardMetadata:
    """Auxiliary metadata for a memory card."""

    timestamp: float = 0.0
    lifecycle_state: str = "active"
    access_count: int = 0
    last_access_time: float = 0.0
    admission_score: float = 0.0
    create_round: int = 0
    last_access_round: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CardMetadata":
        return cls(**data)


@dataclass
class CardContent:
    """Unified structured content κ = (state, plan, exec, eval)."""

    state: str = ""
    plan: str = ""
    exec: str = ""
    eval: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "CardContent":
        if not data:
            return cls()

        def _to_str(val) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                return "\n".join(str(item) for item in val)
            return str(val)

        return cls(
            state=_to_str(data.get("state")),
            plan=_to_str(data.get("plan")),
            exec=_to_str(data.get("exec")),
            eval=_to_str(data.get("eval")),
        )

    def section_map(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in SECTION_ORDER}

    def get(self, section: str) -> str:
        return self.section_map().get(section, "")

    def set(self, section: str, value: str):
        if section in SECTION_ORDER:
            setattr(self, section, value or "")

    def non_empty_sections(self) -> dict[str, str]:
        return {name: text for name, text in self.section_map().items() if text.strip()}

    def coverage(self) -> float:
        filled = sum(1 for text in self.section_map().values() if text.strip())
        return filled / float(len(SECTION_ORDER))

    def has_any(self) -> bool:
        return any(text.strip() for text in self.section_map().values())

    def primary_type(self) -> str:
        sections = self.non_empty_sections()
        if not sections:
            return "card"
        return max(sections.items(), key=lambda item: len(item[1]))[0]

    def compose(self) -> str:
        lines = []
        for section in SECTION_ORDER:
            text = self.get(section).strip()
            if text:
                lines.append(f"[{section}] {text}")
        return "\n".join(lines)


@dataclass
class CardQuality:
    """Quality signals Q(c) = lambda_1*C(c) + lambda_2*N(c) + lambda_3*R(c) + lambda_4*U(c).

    Four components aligned with the methodology (Section 4.5):
      - reliability: internal consistency + evidence support
      - novelty: distinctiveness from existing cards
      - relevance: time-based relevance decay
      - utility: outcome-based usefulness + density bonus
    """

    reliability: float = 0.0
    novelty: float = 0.0
    relevance: float = 0.0
    utility: float = 0.0

    # --- Deprecated aliases for backward compatibility ---
    @property
    def consistency(self) -> float:
        return self.reliability

    @consistency.setter
    def consistency(self, value: float):
        self.reliability = value

    @property
    def evidence_support(self) -> float:
        return self.reliability

    @evidence_support.setter
    def evidence_support(self, value: float):
        pass  # subsumed into reliability

    @property
    def density(self) -> float:
        return self.utility

    @density.setter
    def density(self, value: float):
        pass  # subsumed into utility

    def to_dict(self) -> dict:
        return {
            "reliability": self.reliability,
            "novelty": self.novelty,
            "relevance": self.relevance,
            "utility": self.utility,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "CardQuality":
        if not data:
            return cls()
        # Support both new and legacy field names
        reliability = data.get("reliability", 0.0)
        if reliability == 0.0 and "consistency" in data:
            reliability = data.get("consistency", 0.0)
        utility = data.get("utility", 0.0)
        return cls(
            reliability=reliability,
            novelty=data.get("novelty", 0.0),
            relevance=data.get("relevance", 0.0),
            utility=utility,
        )


@dataclass
class MemoryCard:
    """
    A structured memory card c = (z, κ, τ_c, χ) (Section 4.2.1).

    The methodology treats one card as a unified strategy unit containing four
    internal sections (state/plan/exec/eval) plus trigger semantics.

    Legacy fields `memory_type` and `content` are kept for storage compatibility
    but are deprecated — use `structured_content` and `trigger_semantics` instead.
    """

    card_id: str = ""
    task_id: str = ""
    task_domain: str = ""
    task_description: str = ""
    # Deprecated: use structured_content instead
    memory_type: str = ""
    # Deprecated: use structured_content.compose() instead
    content: str = ""
    evidence: str = ""
    structured_content: CardContent = field(default_factory=CardContent)
    trigger_semantics: list[str] = field(default_factory=list)
    summary: str = ""
    quality: CardQuality = field(default_factory=CardQuality)
    provenance: Provenance = field(default_factory=Provenance)
    metadata: CardMetadata = field(default_factory=CardMetadata)
    embedding: Optional[list] = None

    # Runtime flags
    conflict_warning: str = ""
    deferred_reason: str = ""

    # --- Derived feature caches (not serialized) ---
    _cached_pattern_features: Optional[dict] = field(default=None, init=False, repr=False, compare=False)
    _cached_pattern_terms: Optional[frozenset] = field(default=None, init=False, repr=False, compare=False)
    _cached_pattern_cluster_key: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.structured_content, dict):
            self.structured_content = CardContent.from_dict(self.structured_content)
        if isinstance(self.quality, dict):
            self.quality = CardQuality.from_dict(self.quality)
        if isinstance(self.provenance, dict):
            self.provenance = Provenance.from_dict(self.provenance)
        if isinstance(self.metadata, dict):
            self.metadata = CardMetadata.from_dict(self.metadata)
        if isinstance(self.trigger_semantics, str):
            raw = self.trigger_semantics.strip()
            self.trigger_semantics = [raw] if raw else []

        if not self.card_id:
            self.card_id = str(uuid.uuid4())
        if self.metadata.timestamp == 0.0:
            self.metadata.timestamp = time.time()
            self.metadata.last_access_time = self.metadata.timestamp

        # Legacy migration: map old single-type content into structured_content
        self._hydrate_legacy_content()

        if not self.summary:
            self.summary = self._build_summary()
        if not self.content:
            self.content = self.structured_content.compose() or self.summary
        if not self.memory_type:
            self.memory_type = "card" if self.structured_content.has_any() else ""

    def _hydrate_legacy_content(self):
        """Deprecated: exists only for migration from old single-type cards."""
        if self.structured_content.has_any():
            if not self.content:
                self.content = self.structured_content.compose()
            return
        if not self.content:
            return
        section = self.memory_type if self.memory_type in SECTION_ORDER else ""
        if section:
            self.structured_content.set(section, self.content)
        else:
            self.summary = self.summary or self.content

    def _build_summary(self) -> str:
        pieces = []
        if self.task_description:
            pieces.append(self.task_description.strip())
        for section in SECTION_ORDER:
            text = self.structured_content.get(section).strip()
            if text:
                pieces.append(f"{section}: {text[:SUMMARY_SECTION_PREVIEW_CHARS]}")
        if self.evidence:
            pieces.append(f"evidence: {self.evidence[:SUMMARY_EVIDENCE_PREVIEW_CHARS]}")
        return " | ".join(piece for piece in pieces if piece)

    def primary_memory_type(self) -> str:
        if self.memory_type and self.memory_type in SECTION_ORDER:
            return self.memory_type
        return self.structured_content.primary_type()

    def matches_memory_type(self, memory_type: str) -> bool:
        if not memory_type:
            return True
        if self.memory_type == memory_type:
            return True
        return bool(self.structured_content.get(memory_type).strip())

    def get_section(self, section: str) -> str:
        if section == self.memory_type and self.content and not self.structured_content.get(section):
            return self.content
        return self.structured_content.get(section)

    def section_map(self) -> dict[str, str]:
        return self.structured_content.section_map()

    def coverage(self) -> float:
        return self.structured_content.coverage()

    def activation_text(self) -> str:
        parts = [self.task_description, self.summary, self.content]
        if self.trigger_semantics:
            parts.append(" | ".join(self.trigger_semantics))
        return "\n".join(part for part in parts if part)

    def pattern_features(self) -> dict[str, tuple[str, ...]]:
        if self._cached_pattern_features is not None:
            return self._cached_pattern_features
        task_text = " ".join(
            part for part in (
                self.task_description,
                self.summary,
                self.content,
                self.get_section("state"),
                self.get_section("eval"),
            ) if part
        )
        method_text = " ".join(
            part for part in (
                self.summary,
                self.content,
                self.get_section("plan"),
                self.get_section("exec"),
                " ".join(self.trigger_semantics),
            ) if part
        )
        all_text = " ".join(part for part in (task_text, method_text, self.evidence) if part)

        families = self._match_feature_tags(task_text or all_text, PATTERN_FAMILY_KEYWORDS)
        methods = self._match_feature_tags(method_text or all_text, PATTERN_METHOD_KEYWORDS)
        contracts = self._match_feature_tags(all_text, PATTERN_CONTRACT_KEYWORDS)
        libraries = self._match_feature_tags(all_text, PATTERN_LIBRARY_KEYWORDS)

        if not families:
            families = self._fallback_signature_terms(task_text or all_text, limit=2)
        if not methods:
            methods = self._fallback_signature_terms(method_text or all_text, limit=3)

        result = {
            "families": tuple(families),
            "methods": tuple(methods),
            "contracts": tuple(contracts),
            "libraries": tuple(libraries),
        }
        self._cached_pattern_features = result
        return result

    @property
    def pattern_signature(self) -> str:
        features = self.pattern_features()
        parts = [
            f"fam:{','.join(features['families']) or 'generic'}",
            f"met:{','.join(features['methods']) or 'generic'}",
        ]
        if features["contracts"]:
            parts.append(f"ctr:{','.join(features['contracts'])}")
        if features["libraries"]:
            parts.append(f"lib:{','.join(features['libraries'])}")
        return "|".join(parts)

    @property
    def pattern_cluster_key(self) -> str:
        if self._cached_pattern_cluster_key is not None:
            return self._cached_pattern_cluster_key
        features = self.pattern_features()
        family_key = ",".join(features["families"]) or "generic"
        method_key = ",".join(features["methods"]) or "generic"
        key = f"fam:{family_key}|met:{method_key}"
        self._cached_pattern_cluster_key = key
        return key

    def pattern_terms(self) -> set[str]:
        if self._cached_pattern_terms is not None:
            return self._cached_pattern_terms
        features = self.pattern_features()
        terms = frozenset(features["families"]) | frozenset(features["methods"]) | frozenset(features["contracts"]) | frozenset(features["libraries"])
        self._cached_pattern_terms = terms
        return terms

    def refresh_derived_fields(self):
        self.summary = self._build_summary()
        self.content = self.structured_content.compose() or self.summary
        self.memory_type = "card" if self.structured_content.has_any() else ""
        # Invalidate cached pattern features — inputs (summary/content) changed.
        self._cached_pattern_features = None
        self._cached_pattern_terms = None
        self._cached_pattern_cluster_key = None

    def to_payload_dict(self) -> dict:
        return {
            "structured_content": self.structured_content.to_dict(),
            "trigger_semantics": list(self.trigger_semantics),
            "summary": self.summary,
            "quality": self.quality.to_dict(),
            "pattern_signature": self.pattern_signature,
        }

    def to_storage_dict(self) -> dict:
        return {
            "card_id": self.card_id,
            "task_id": self.task_id,
            "task_domain": self.task_domain,
            "task_description": self.task_description,
            "memory_type": self.memory_type or "card",
            "content": self.content,
            "evidence": self.evidence,
            "provenance": json.dumps(self.provenance.to_dict()),
            "metadata": json.dumps(self.metadata.to_dict()),
            "embedding": _serialize_embedding_json(self.embedding),
            "card_payload": json.dumps(self.to_payload_dict()),
        }

    @classmethod
    def from_storage_dict(cls, data: dict) -> "MemoryCard":
        payload_raw = data.get("card_payload")
        payload = json.loads(payload_raw) if payload_raw else {}
        return cls(
            card_id=data.get("card_id", ""),
            task_id=data.get("task_id", ""),
            task_domain=data.get("task_domain", "") or "",
            task_description=data.get("task_description", ""),
            memory_type=data.get("memory_type", ""),
            content=data.get("content", "") or "",
            evidence=data.get("evidence", "") or "",
            structured_content=CardContent.from_dict(payload.get("structured_content")),
            trigger_semantics=payload.get("trigger_semantics", []),
            summary=payload.get("summary", "") or "",
            quality=CardQuality.from_dict(payload.get("quality")),
            provenance=Provenance.from_dict(json.loads(data.get("provenance") or "{}")),
            metadata=CardMetadata.from_dict(json.loads(data.get("metadata") or "{}")),
            embedding=_deserialize_embedding_blob(data.get("embedding")),
        )

    def _match_feature_tags(self, text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
        lowered = (text or "").lower()
        tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9+-]*", lowered))
        matched = []
        for tag, phrases in mapping.items():
            for phrase in phrases:
                phrase_lower = phrase.lower()
                if re.search(r"[^a-z0-9_]", phrase_lower) and phrase_lower in lowered:
                    matched.append(tag)
                    break
                if phrase_lower in tokens:
                    matched.append(tag)
                    break
        return sorted(dict.fromkeys(matched))

    def _fallback_signature_terms(self, text: str, limit: int) -> list[str]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z_0-9+-]{3,}", (text or "").lower())
        counts = Counter(
            token for token in tokens
            if token not in PATTERN_STOPWORDS and not token.startswith(("state", "plan", "exec", "eval"))
        )
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ranked[:limit]]


@dataclass
class TaskRecord:
    """Task-level metadata stored in the task registry."""

    task_id: str = ""
    task_domain: str = ""
    task_description: str = ""
    outcome: str = ""
    completion_round: int = 0
    completion_timestamp: float = 0.0
    trajectory_file: str = ""
    embedding: Optional[list] = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if self.completion_timestamp == 0.0:
            self.completion_timestamp = time.time()


@dataclass
class MemoryEdge:
    """An edge in the card graph."""

    source_card_id: str = ""
    target_card_id: str = ""
    relation: str = ""
    weight: float = 0.0
    rationale: str = ""
