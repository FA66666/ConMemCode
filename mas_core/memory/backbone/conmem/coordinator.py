"""Coordination for unified strategy cards."""
import logging
import re
from collections import defaultdict

from .config import ConMemConfig
from .llm_backend import EmbeddingClient, LLMClient, cosine_similarity
from .schema import MemoryCard, MemoryEdge, SECTION_ORDER
from .storage import ConMemStorage

logger = logging.getLogger(__name__)


class MemoryCoordinator:
    def __init__(
        self,
        config: ConMemConfig,
        llm: LLMClient,
        embedder: EmbeddingClient,
        storage: ConMemStorage,
    ):
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.storage = storage

    def coordinate(
        self,
        activated_cards: list[MemoryCard],
        subgraph_edges: list[MemoryEdge],
        task_description: str = "",
    ) -> list[MemoryCard]:
        if not activated_cards:
            return []
        merged = self._merge_redundant(activated_cards)
        selected = self._select_conflicts(merged, subgraph_edges)
        return self._defer_constraints(selected, subgraph_edges, task_description)

    def _merge_redundant(self, cards: list[MemoryCard]) -> list[MemoryCard]:
        result: list[MemoryCard] = []
        consumed = set()
        for i, card in enumerate(cards):
            if i in consumed:
                continue
            cluster = [card]
            for j in range(i + 1, len(cards)):
                if j in consumed:
                    continue
                other = cards[j]
                if card.embedding is not None and other.embedding is not None and cosine_similarity(card.embedding, other.embedding) >= self.config.coord_merge_threshold:
                    cluster.append(other)
                    consumed.add(j)
            result.append(self._merge_cluster(cluster))
        return result

    def _merge_cluster(self, cluster: list[MemoryCard]) -> MemoryCard:
        if len(cluster) == 1:
            return cluster[0]
        cluster = sorted(cluster, key=lambda card: card.metadata.admission_score, reverse=True)
        base = cluster[0]
        for other in cluster[1:]:
            for section in SECTION_ORDER:
                left = base.get_section(section)
                right = other.get_section(section)
                if right and right not in left:
                    merged = f"{left}\n{right}".strip()
                    base.structured_content.set(section, merged)
            base.trigger_semantics = list(dict.fromkeys(base.trigger_semantics + other.trigger_semantics))[: self.config.trigger_max_semantics]
            if other.evidence and other.evidence not in base.evidence:
                base.evidence = ((base.evidence + " | " + other.evidence).strip(" |"))
        return base

    def _select_conflicts(self, cards: list[MemoryCard], edges: list[MemoryEdge]) -> list[MemoryCard]:
        card_map = {card.card_id: card for card in cards}
        removed = set()
        seen_pairs: set[frozenset[str]] = set()
        for edge in edges:
            if edge.relation != "conflicts":
                continue
            pair = frozenset((edge.source_card_id, edge.target_card_id))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            left = card_map.get(edge.source_card_id)
            right = card_map.get(edge.target_card_id)
            if left is None or right is None:
                continue
            left_score = left.metadata.admission_score
            right_score = right.metadata.admission_score
            if left_score >= right_score:
                removed.add(right.card_id)
                left.conflict_warning = edge.rationale or "Selected over a conflicting card"
            else:
                removed.add(left.card_id)
                right.conflict_warning = edge.rationale or "Selected over a conflicting card"
        return [card for card in cards if card.card_id not in removed]

    def _defer_constraints(
        self,
        cards: list[MemoryCard],
        edges: list[MemoryEdge],
        task_description: str,
    ) -> list[MemoryCard]:
        constrained_targets = defaultdict(list)
        for edge in edges:
            if edge.relation == "constrains":
                constrained_targets[edge.target_card_id].append(edge)

        result = []
        for card in cards:
            card_edges = constrained_targets.get(card.card_id, [])
            if not card_edges:
                result.append(card)
                continue
            trigger_text = " ".join(card.trigger_semantics + [card.summary]).lower()
            token_pattern = self.config.token_regex()
            task_tokens = set(re.findall(token_pattern, task_description.lower()))
            trigger_tokens = set(re.findall(token_pattern, trigger_text))
            if (
                task_tokens
                and len(task_tokens & trigger_tokens) / len(task_tokens)
                < self.config.coord_constraint_defer_overlap
            ):
                card.deferred_reason = card_edges[0].rationale or "Deferred because required constraints were not triggered"
                continue
            result.append(card)
        return result
