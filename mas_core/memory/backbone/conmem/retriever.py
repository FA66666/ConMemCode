"""Trigger-based retrieval for unified ConMem cards (Section 4.4.1)."""
import logging
import re

from .config import ConMemConfig
from .llm_backend import EmbeddingClient, cosine_similarity
from .schema import MemoryCard, SECTION_ORDER
from .storage import ConMemStorage

logger = logging.getLogger(__name__)


class MemoryRetriever:
    def __init__(self, config: ConMemConfig, embedder: EmbeddingClient, storage: ConMemStorage):
        self.config = config
        self.embedder = embedder
        self.storage = storage

    def retrieve(
        self,
        task_id: str,
        task_description: str,
        agent_role: str = "default",
        current_round: int = 0,
        task_domain: str = None,
        interaction_context: str = "",
    ) -> list[MemoryCard]:
        """
        Trigger-based retrieval pipeline (Section 4.4.1).

        Phase 1: Analyze task needs (plan/exec/eval/state requirements).
        Phase 2: Build triggered set C_t^(0) via binary trigger function.
        Phase 3: Score and rank the triggered cards for selection.
        """
        query_text = self._compose_query_text(task_description, interaction_context)
        try:
            query_emb = self.embedder.embed(query_text)
        except Exception as e:
            logger.warning("Embedding failed during retrieval: %s", e)
            return []

        # Phase 0: Analyze what sections this task needs
        # Use benchmark preset if available, otherwise fallback to keyword analysis
        presets = getattr(self.config, "task_needs_presets", {})
        if task_domain and task_domain in presets:
            task_needs = self._normalize_task_needs(presets[task_domain])
        else:
            task_needs = self._analyze_task_needs(query_text)
        logger.debug("Task %s needs: %s", task_id, task_needs)

        # Phase 1: Build triggered set C_t^(0) = {c ∈ C | trigger(c | z, h) = 1}
        triggered = self._triggered_cards(
            task_id=task_id,
            task_description=task_description,
            query_text=query_text,
            query_emb=query_emb,
            task_needs=task_needs,
            task_domain=task_domain,
        )

        if not triggered:
            logger.info("Retrieval: 0 cards triggered for task %s", task_id)
            return []

        # Phase 2: Score triggered cards for ranking. Use the current
        # query text; ranking on card.task_description can promote
        # high-quality cards that are unrelated to this task.
        scored: list[tuple[MemoryCard, float]] = []
        for card in triggered:
            score = self._score_card(card, query_text, query_emb, task_needs, task_domain=task_domain)
            scored.append((card, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top_cards = self._select_diverse(scored, self.config.retrieval_top_k)
        logger.info("Retrieval: %d cards triggered, %d cards returned for task %s", len(triggered), len(top_cards), task_id)
        for card in top_cards:
            self.storage.record_card_access(card.card_id, current_round)
        return top_cards

    def _triggered_cards(
        self,
        *,
        task_id: str,
        task_description: str,
        query_text: str,
        query_emb: list[float],
        task_needs: dict[str, float],
        task_domain: str | None,
    ) -> list[MemoryCard]:
        fallback_pool = self._candidate_pool(task_domain=task_domain, same_domain_only=False)
        return self._trigger_from_pool(
            fallback_pool, task_id, task_description, query_text, query_emb, task_needs, task_domain
        )

    def _candidate_pool(self, task_domain: str | None, same_domain_only: bool) -> list[MemoryCard]:
        return self.storage.get_all_active_cards()

    def _trigger_from_pool(
        self,
        pool: list[MemoryCard],
        task_id: str,
        task_description: str,
        query_text: str,
        query_emb: list[float],
        task_needs: dict[str, float],
        task_domain: str | None,
    ) -> list[MemoryCard]:
        triggered: list[MemoryCard] = []
        for card in pool:
            if not self._trigger(card, task_id, query_text, query_emb, task_needs, task_domain=task_domain):
                continue
            if self._reject_domain_mismatch(card, task_description, task_domain):
                continue
            triggered.append(card)
        return triggered

    def _should_cross_domain_fallback(self, task_domain: str | None) -> bool:
        return bool(
            self.config.enforce_task_domain_filter
            and self.config.allow_cross_domain_retrieval_fallback
            and task_domain
        )

    def _compose_query_text(self, task_description: str, interaction_context: str) -> str:
        interaction_context = (interaction_context or "").strip()
        if not interaction_context:
            return task_description
        return f"{task_description}\n{interaction_context}".strip()

    def _normalize_task_needs(self, needs: dict[str, float]) -> dict[str, float]:
        normalized = {section: max(0.0, float(needs.get(section, 0.0))) for section in SECTION_ORDER}
        total = sum(normalized.values())
        if total <= 0:
            uniform = 1.0 / len(SECTION_ORDER)
            return {section: uniform for section in SECTION_ORDER}
        return {section: score / total for section, score in normalized.items()}

    def _analyze_task_needs(self, task_description: str) -> dict[str, float]:
        """
        Analyze task to determine what memory sections are needed.
        Returns a normalized task-needs profile π_t over the four sections.
        """
        task_lower = task_description.lower()
        
        # Keywords indicating need for each section type
        section_keywords = {
            "plan": [
                "plan", "strategy", "approach", "how to", "steps", "method",
                "organize", "structure", "design", "prepare", "first", "then",
                "before", "after", "procedure", "process", "way to"
            ],
            "exec": [
                "execute", "implement", "run", "build", "create", "write",
                "code", "solve", "compute", "calculate", "generate", "produce",
                "perform", "do", "make", "fix", "debug", "test"
            ],
            "eval": [
                "evaluate", "check", "verify", "review", "validate", "assess",
                "confirm", "ensure", "correct", "accurate", "test", "examine",
                "inspect", "judge", "critique", "agree", "disagree"
            ],
            "state": [
                "state", "context", "remember", "know", "facts", "information",
                "data", "status", "current", "previous", "history", "background",
                "what is", "who is", "where is", "when did", "which"
            ],
        }
        
        needs = {}
        for section, keywords in section_keywords.items():
            score = 0.0
            for kw in keywords:
                if kw in task_lower:
                    score += self.config.retrieval_task_needs_keyword_increment
            needs[section] = min(score, 1.0)  # Cap raw score before normalization

        if max(needs.values()) < self.config.retrieval_task_needs_default_uniform_threshold:
            uniform = 1.0 / len(SECTION_ORDER)
            return {section: uniform for section in SECTION_ORDER}

        return self._normalize_task_needs(needs)

    def _trigger(
        self,
        card: MemoryCard,
        task_id: str,
        query_text: str,
        query_emb: list[float],
        task_needs: dict[str, float] | None = None,
        task_domain: str | None = None,
    ) -> bool:
        """
        Binary trigger function: trigger(c | z, h) → {0, 1} (Section 4.4.1).

        A card is activated if its semantic similarity and keyword overlap
        with the current task exceeds the activation threshold.
        Also considers task needs - cards with matching sections get boost.
        """
        task_needs = task_needs or {}
        
        relevance = self._relevance(card, query_emb)
        keyword_overlap = self._keyword_overlap(card, query_text)
        if not self._passes_relevance_gate(relevance, keyword_overlap):
            return False

        # Compute trigger score from embedding similarity + keyword overlap
        trigger_score = self._trigger_score(card, query_text, query_emb)
        base_score = max(trigger_score, relevance)

        section_match_boost = 0.0
        if task_needs and base_score >= self.config.retrieval_section_gate_threshold:
            section_match_boost = self._section_needs_score(card, task_needs)

        adjusted_score = base_score + section_match_boost - self._activation_penalty(card, task_domain)
        return adjusted_score >= self.config.activation_threshold

    def _score_card(
        self,
        card: MemoryCard,
        query_text: str,
        query_emb: list[float],
        task_needs: dict[str, float] | None = None,
        task_domain: str | None = None,
    ) -> float:
        """Rank score S_ret(c) using the TeX-aligned 5-term formula."""
        relevance = self._relevance(card, query_emb)
        trigger_score = self._trigger_score(card, query_text, query_emb)
        section_needs_score = 0.0
        if task_needs and max(trigger_score, relevance) >= self.config.retrieval_section_gate_threshold:
            section_needs_score = self._section_needs_score(card, task_needs)

        score = (
            self.config.alpha_relevance * relevance
            + self.config.alpha_trigger * trigger_score
            + self.config.alpha_section_needs * section_needs_score
            + self.config.alpha_credibility * self._credibility(card)
            + self.config.alpha_quality * self._quality(card)
        )
        return score - self._score_penalty(card, task_domain)

    def _relevance(self, card: MemoryCard, query_emb: list[float]) -> float:
        if card.embedding is not None and query_emb is not None and len(query_emb) > 0:
            return max(0.0, cosine_similarity(card.embedding, query_emb))
        return 0.0

    def _role_section_fit(self, card: MemoryCard, role_key: str) -> float:
        weights = self.config.role_section_weights.get(
            role_key, self.config.role_section_weights["default"]
        )
        total = sum(weights.values()) or 1.0
        score = 0.0
        for section in SECTION_ORDER:
            if (card.get_section(section) or "").strip():
                score += weights.get(section, 0.0)
        return score / total

    def _credibility(self, card: MemoryCard) -> float:
        base_credibility = self.config.credibility_table.get(
            card.provenance.trajectory_outcome,
            self.config.retrieval_quality_fallback,
        )
        reflection_quality = max(0.0, min(1.0, float(getattr(card.provenance, "reflection_quality", 1.0))))
        reflection_factor = (
            self.config.credibility_reflection_floor
            + self.config.credibility_reflection_weight * reflection_quality
        )
        return base_credibility * reflection_factor

    def _quality(self, card: MemoryCard) -> float:
        if card.metadata.admission_score > 0:
            return card.metadata.admission_score
        values = [
            card.quality.reliability,
            card.quality.novelty,
            card.quality.relevance,
            card.quality.utility,
        ]
        values = [value for value in values if value > 0]
        return sum(values) / len(values) if values else self.config.retrieval_quality_fallback

    def _section_needs_score(self, card: MemoryCard, task_needs: dict[str, float]) -> float:
        return sum(
            task_needs.get(section, 0.0)
            for section in SECTION_ORDER
            if (card.get_section(section) or "").strip()
        )

    def _trigger_score(self, card: MemoryCard, query_text: str, query_emb: list[float]) -> float:
        similarity = self._relevance(card, query_emb)
        query_terms = self._tokenize(query_text)
        card_terms = self._tokenize(self._trigger_match_text(card))
        overlap = len(query_terms & card_terms) / len(query_terms) if query_terms else 0.0
        return (
            self.config.trigger_similarity_weight * similarity
            + self.config.trigger_keyword_weight * overlap
        )

    def _keyword_overlap(self, card: MemoryCard, query_text: str) -> float:
        query_terms = self._tokenize(query_text)
        if not query_terms:
            return 0.0
        card_terms = self._tokenize(self._trigger_match_text(card))
        return len(query_terms & card_terms) / len(query_terms)

    def _trigger_match_text(self, card: MemoryCard) -> str:
        sections = [self._card_section(card, section) for section in SECTION_ORDER]
        return " ".join(
            list(getattr(card, "trigger_semantics", []))
            + [getattr(card, "summary", ""), getattr(card, "task_description", "")]
            + [text for text in sections if text]
        )

    def _card_section(self, card: MemoryCard, section: str) -> str:
        getter = getattr(card, "get_section", None)
        if callable(getter):
            return getter(section) or ""
        structured = getattr(card, "structured_content", None)
        if structured is not None:
            if isinstance(structured, dict):
                return structured.get(section, "") or ""
            section_getter = getattr(structured, "get", None)
            if callable(section_getter):
                return section_getter(section) or ""
        return getattr(card, "content", "") or ""

    def _passes_relevance_gate(self, relevance: float, keyword_overlap: float) -> bool:
        return (
            relevance >= self.config.retrieval_min_relevance
            and keyword_overlap >= self.config.retrieval_min_keyword_overlap
        )

    def _same_domain(self, card: MemoryCard, task_domain: str | None) -> bool:
        return True

    def _activation_penalty(self, card: MemoryCard, task_domain: str | None) -> float:
        return 0.0 if self._same_domain(card, task_domain) else self.config.cross_domain_activation_penalty

    def _score_penalty(self, card: MemoryCard, task_domain: str | None) -> float:
        return 0.0 if self._same_domain(card, task_domain) else self.config.cross_domain_score_penalty

    def _reject_domain_mismatch(
        self,
        card: MemoryCard,
        task_description: str,
        task_domain: str | None,
    ) -> bool:
        if task_domain != "kodcode" or not self.config.enable_kodcode_contract_gate:
            return False
        return (
            self._reject_kodcode_topic_mismatch(card, task_description)
            or self._reject_kodcode_contract_mismatch(card, task_description)
        )

    def _reject_kodcode_topic_mismatch(self, card: MemoryCard, task_description: str) -> bool:
        """Reject cards whose algorithm vocabulary belongs to another programming task."""
        task_topics = self._kodcode_topics(task_description)
        card_topics = self._kodcode_topics(self._trigger_match_text(card))
        off_topic = card_topics - task_topics
        on_topic = card_topics & task_topics

        # A single generic term like "array" or "operator" is harmless.
        # Drop the card only when specific extra objectives dominate the match.
        return (
            len(off_topic) >= self.config.kodcode_off_topic_min_terms
            and len(off_topic) > len(on_topic) + self.config.kodcode_off_topic_margin
        )

    def _reject_kodcode_contract_mismatch(self, card: MemoryCard, task_description: str) -> bool:
        """
        Reject cards whose input/output contract family is incompatible with the task.

        This catches high-lexical-overlap cases such as "reverse list elements" vs
        "reverse words in a sentence", where topic terms like "reverse" are too generic
        to separate the tasks on their own.
        """
        task_contracts = self._kodcode_contract_terms(task_description)
        card_contracts = self._kodcode_contract_terms(self._trigger_match_text(card))

        if not task_contracts or not card_contracts:
            return False
        return task_contracts.isdisjoint(card_contracts)

    def _kodcode_topics(self, text: str) -> set[str]:
        words = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", text.lower()))
        topics = words & KODCODE_TOPIC_TERMS
        lowered = text.lower()
        phrase_topics = {
            "non_overlapping": ("non overlapping", "non-overlap", "nonoverlapping"),
            "prefix_sum": ("prefix sum", "running sum", "cumulative sum"),
            "sliding_window": ("sliding window", "two pointer", "two-pointer"),
            "product_except_self": ("product except self",),
            "prime_factor": ("prime factor", "factorization", "factorisation"),
            "roman_numeral": ("roman numeral",),
            "time_format": ("am/pm", "12-hour", "24-hour", "military time"),
        }
        for topic, phrases in phrase_topics.items():
            if any(phrase in lowered for phrase in phrases):
                topics.add(topic)
        return topics

    def _kodcode_contract_terms(self, text: str) -> set[str]:
        lowered = text.lower()
        words = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", lowered))
        contracts: set[str] = set()

        natural_text_terms = {
            "sentence", "word", "words", "character", "characters",
            "substring", "substrings", "letter", "letters",
        }
        sequence_terms = {
            "list", "lists", "array", "arrays", "element", "elements",
            "item", "items", "tuple", "tuples",
        }
        mapping_terms = {"dict", "dictionary", "mapping", "hashmap", "key", "keys", "value", "values"}
        grid_terms = {"matrix", "grid", "board", "boards", "row", "rows", "column", "columns"}
        linked_terms = {"node", "nodes", "linked", "linkedlist", "linked_list"}
        tree_terms = {"tree", "trees", "binary_tree", "bst", "graph", "graphs", "vertex", "vertices"}

        if words & natural_text_terms:
            contracts.add("natural_text")
        if words & sequence_terms:
            contracts.add("sequence")
        if words & mapping_terms:
            contracts.add("mapping")
        if words & grid_terms:
            contracts.add("grid")
        if words & linked_terms:
            contracts.add("linked")
        if words & tree_terms:
            contracts.add("tree")

        if any(phrase in lowered for phrase in ("space-separated", "comma-separated", "tab-separated")):
            contracts.add("token_stream")
        if "json" in words or "yaml" in words or "xml" in words:
            contracts.add("serialized_text")
        if "float" in words or "floats" in words or "integer" in words or "integers" in words or "numbers" in words:
            contracts.add("numeric")

        return contracts

    def _select_diverse(
        self,
        scored: list[tuple[MemoryCard, float]],
        top_k: int,
    ) -> list[MemoryCard]:
        if top_k <= 0 or not scored:
            return []

        selected: list[tuple[MemoryCard, float]] = []
        remaining = list(scored)
        mmr_lambda = min(1.0, max(0.0, self.config.retrieval_mmr_lambda))

        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_idx = 0
            best_mmr = float("-inf")
            for idx, (card, score) in enumerate(remaining):
                redundancy = 0.0
                if card.embedding is not None:
                    redundancy = max(
                        (
                            max(0.0, cosine_similarity(card.embedding, chosen.embedding))
                            if chosen.embedding is not None else 0.0
                        )
                        for chosen, _ in selected
                    )
                mmr_score = mmr_lambda * score - (1.0 - mmr_lambda) * redundancy
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return [card for card, _ in selected]

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(self.config.token_regex(), text.lower()))

KODCODE_TOPIC_TERMS = {
    "anagram",
    "array",
    "binary",
    "character",
    "count",
    "decode",
    "delimiter",
    "digit",
    "distance",
    "encode",
    "factor",
    "frequency",
    "graph",
    "integer",
    "kth",
    "letter",
    "manhattan",
    "matrix",
    "number",
    "operator",
    "palindrome",
    "parentheses",
    "permutation",
    "point",
    "points",
    "postfix",
    "prefix",
    "prime",
    "product",
    "queue",
    "schedule",
    "scheduler",
    "regex",
    "roman",
    "rotate",
    "stack",
    "string",
    "subarray",
    "substring",
    "suffix",
    "threshold",
    "interval",
    "cooldown",
    "time",
    "tree",
    "valid",
    "window",
}
