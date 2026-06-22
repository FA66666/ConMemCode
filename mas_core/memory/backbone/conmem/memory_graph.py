"""Unified card-graph construction and constrained expansion."""
import logging
import re

from .config import ConMemConfig
from .llm_backend import EmbeddingClient, LLMClient, cosine_similarity
from .prompts import GRAPH_RELATION_SYSTEM, GRAPH_RELATION_USER
from .schema import MemoryCard, MemoryEdge
from .storage import ConMemStorage
from .utils import token_overlap

logger = logging.getLogger(__name__)


class MemoryGraphManager:
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

    def update_graph(self, new_cards: list[MemoryCard]):
        if not new_cards:
            return
        old_cards = self.storage.get_all_active_cards()
        new_ids = {card.card_id for card in new_cards}
        old_cards = [card for card in old_cards if card.card_id not in new_ids]
        for new_card in new_cards:
            same_domain_cards = [
                card for card in old_cards if self._same_domain(new_card, card)
            ]
            candidate_pool = same_domain_cards if same_domain_cards else old_cards
            if not self.config.allow_cross_domain_graph_edges:
                candidate_pool = same_domain_cards
            candidates = self._find_candidates(new_card, candidate_pool)
            if not candidates:
                continue
            edges = self._materialize_edges(self._judge_relations(new_card, candidates))
            if edges:
                self.storage.insert_edges(edges)

    def expand_activation(
        self,
        activated_cards: list[MemoryCard],
        task_description: str,
        max_expanded: int | None = None,
    ) -> tuple[list[MemoryCard], list[MemoryEdge]]:
        if not activated_cards:
            return [], []

        selected = {card.card_id: card for card in activated_cards}
        frontier = set(selected)
        all_edges: dict[tuple[str, str], MemoryEdge] = {}
        max_expanded = max_expanded or 0

        for _ in range(self.config.graph_walk_hops):
            next_frontier = set()
            for edge in self.storage.get_edges_for_cards(frontier):
                all_edges[(edge.source_card_id, edge.target_card_id)] = edge
                if edge.source_card_id not in frontier:
                    continue
                neighbor_id = edge.target_card_id
                if neighbor_id in selected:
                    continue
                if not self._should_expand(edge, task_description, neighbor_id):
                    continue
                neighbor = self.storage.get_card(neighbor_id)
                if neighbor is None:
                    continue
                selected[neighbor.card_id] = neighbor
                next_frontier.add(neighbor.card_id)
                if max_expanded > 0 and len(selected) >= max_expanded:
                    break
            if max_expanded > 0 and len(selected) >= max_expanded:
                break
            if not next_frontier:
                break
            frontier = next_frontier

        return list(selected.values()), list(all_edges.values())

    def get_subgraph(self, card_ids: set[str]) -> list[MemoryEdge]:
        return self.storage.get_edges_for_cards(card_ids)

    def _find_candidates(self, new_card: MemoryCard, old_cards: list[MemoryCard]) -> list[MemoryCard]:
        scored = []
        for old_card in old_cards:
            if not self._same_domain(new_card, old_card):
                continue
            if new_card.embedding is not None and old_card.embedding is not None:
                similarity = cosine_similarity(new_card.embedding, old_card.embedding)
                if similarity >= self.config.graph_similarity_threshold:
                    scored.append((old_card, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [card for card, _ in scored[: self.config.graph_candidate_top_k]]

    def _same_domain(self, left: MemoryCard, right: MemoryCard) -> bool:
        if not self.config.enforce_task_domain_filter:
            return True
        if not left.task_domain or not right.task_domain:
            return True
        return left.task_domain == right.task_domain

    def _judge_relations(self, new_card: MemoryCard, candidates: list[MemoryCard]) -> list[MemoryEdge]:
        if not getattr(self.config, "graph_relation_heuristic_first", True):
            return self._llm_judge_relations(new_card, candidates)

        heuristic_candidates: list[MemoryCard] = []
        llm_candidates: list[MemoryCard] = []
        ambiguous_lo, ambiguous_hi = getattr(self.config, "graph_relation_ambiguous_range", (0.30, 0.70))

        for card in candidates:
            if new_card.embedding is None or card.embedding is None:
                heuristic_candidates.append(card)
                continue
            sim = cosine_similarity(new_card.embedding, card.embedding)
            if ambiguous_lo <= sim <= ambiguous_hi:
                llm_candidates.append(card)
            else:
                heuristic_candidates.append(card)

        heuristic_edges = self._heuristic_relations(new_card, heuristic_candidates)
        if not llm_candidates:
            return heuristic_edges
        return heuristic_edges + self._llm_judge_relations(new_card, llm_candidates)

    def _llm_judge_relations(self, new_card: MemoryCard, candidates: list[MemoryCard]) -> list[MemoryEdge]:
        existing_cards_text = "\n\n".join(
            f"Card {i+1}\nSummary: {card.summary}\nTrigger: {', '.join(card.trigger_semantics)}\nContent:\n{card.content}"
            for i, card in enumerate(candidates)
        )
        try:
            result = self.llm.chat_json(
                GRAPH_RELATION_SYSTEM,
                GRAPH_RELATION_USER.format(
                    new_summary=new_card.summary,
                    new_triggers=", ".join(new_card.trigger_semantics),
                    new_content=new_card.content,
                    existing_cards_text=existing_cards_text,
                ),
            )
            if isinstance(result, list):
                edges = []
                # Create index-to-card mapping (Card 1 -> candidates[0], etc.)
                index_to_card = {f"Card {i+1}": card for i, card in enumerate(candidates)}
                for item in result:
                    card_ref = item.get("existing_card_ref") or item.get("existing_card_id")
                    if card_ref not in index_to_card:
                        continue
                    target_card = index_to_card[card_ref]
                    relation = item.get("relation", "none")
                    if relation == "none":
                        continue
                    edges.append(
                        MemoryEdge(
                            source_card_id=new_card.card_id,
                            target_card_id=target_card.card_id,
                            relation=relation,
                            weight=max(
                                0.0,
                                min(1.0, float(item.get("weight", self.config.graph_relation_default_weight))),
                            ),
                            rationale=item.get("rationale", ""),
                        )
                    )
                if edges:
                    return edges
        except Exception as e:
            logger.warning("Graph relation LLM failed: %s. Falling back to heuristics.", e)
        return self._heuristic_relations(new_card, candidates)

    def _heuristic_relations(self, new_card: MemoryCard, candidates: list[MemoryCard]) -> list[MemoryEdge]:
        edges = []
        for card in candidates:
            similarity = 0.0
            if new_card.embedding is not None and card.embedding is not None:
                similarity = cosine_similarity(new_card.embedding, card.embedding)
            overlap = token_overlap(new_card.content, card.content)
            relation = "none"
            new_failure = self._contains_failure_signal(new_card.get_section("eval"))
            old_failure = self._contains_failure_signal(card.get_section("eval"))
            if (new_failure or old_failure) and overlap >= self.config.graph_heuristic_conflict_overlap:
                relation = "conflicts"
            elif (
                similarity >= self.config.graph_heuristic_support_similarity
                or overlap >= self.config.graph_heuristic_support_overlap
            ):
                relation = "supports"
            elif new_card.get_section("eval") and overlap >= self.config.graph_heuristic_satisfies_overlap:
                relation = "satisfies"
            elif new_card.get_section("state") and overlap >= self.config.graph_heuristic_constrains_overlap:
                relation = "constrains"
            if relation != "none":
                edges.append(
                    MemoryEdge(
                        source_card_id=new_card.card_id,
                        target_card_id=card.card_id,
                        relation=relation,
                        weight=max(similarity, overlap, self.config.graph_heuristic_default_weight),
                        rationale=f"heuristic {relation} relation",
                    )
                )
        return edges

    def _materialize_edges(self, relations: list[MemoryEdge]) -> list[MemoryEdge]:
        """Normalize judged relations into traversal-oriented graph edges.

        The relation judge compares a new card against an existing card. The
        directed walk used at retrieval time should instead move from an
        activated strategy card toward supportive or conditioning neighbors.
        """
        normalized: dict[tuple[str, str], MemoryEdge] = {}
        for edge in relations:
            if edge.relation in {"supports", "conflicts"}:
                variants = [
                    edge,
                    MemoryEdge(
                        source_card_id=edge.target_card_id,
                        target_card_id=edge.source_card_id,
                        relation=edge.relation,
                        weight=edge.weight,
                        rationale=edge.rationale,
                    ),
                ]
            elif edge.relation in {"constrains", "satisfies"}:
                variants = [
                    MemoryEdge(
                        source_card_id=edge.target_card_id,
                        target_card_id=edge.source_card_id,
                        relation=edge.relation,
                        weight=edge.weight,
                        rationale=edge.rationale,
                    )
                ]
            else:
                variants = [edge]

            for variant in variants:
                normalized[(variant.source_card_id, variant.target_card_id)] = variant
        return list(normalized.values())

    def _should_expand(self, edge: MemoryEdge, task_description: str, neighbor_id: str) -> bool:
        """Check if neighbor should be expanded (Section 3.4 Implementation_Details.md).
        
        For constrains relation:
        cond(c_j, z_t) = |tokens(z_t) ∩ tokens(χ_j ∪ summary_j)| / |tokens(z_t)| > 0.1
        """
        if edge.relation == "conflicts":
            return False
        if edge.weight < self.config.graph_walk_weight_threshold:
            return False
        if edge.relation in {"supports", "satisfies"}:
            return True
        if edge.relation == "constrains":
            neighbor = self.storage.get_card(neighbor_id)
            if neighbor is None:
                return False
            # Properly implement: |tokens(z_t) ∩ tokens(χ_j ∪ summary_j)| / |tokens(z_t)|
            token_pattern = self.config.token_regex()
            task_tokens = set(re.findall(token_pattern, task_description.lower()))
            neighbor_text = " ".join(neighbor.trigger_semantics + [neighbor.summary])
            neighbor_tokens = set(re.findall(token_pattern, neighbor_text.lower()))
            if not task_tokens:
                return False
            overlap = len(task_tokens & neighbor_tokens) / len(task_tokens)
            return overlap > self.config.graph_constraint_activation_overlap
        return False

    def _contains_failure_signal(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in ("fail", "error", "incorrect", "bug"))
