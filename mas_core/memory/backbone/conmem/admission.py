"""Admission and post-commit merge for unified strategy cards."""
import logging
import math
import re
from collections import Counter, defaultdict

from .config import ConMemConfig
from .llm_backend import EmbeddingClient, LLMClient, cosine_similarity
from .prompts import MERGE_CONTENT_SYSTEM, MERGE_CONTENT_USER
from .schema import CardContent, MemoryCard, SECTION_ORDER
from .storage import ConMemStorage
from .utils import token_overlap

logger = logging.getLogger(__name__)

TRUNCATION_MARKER_RE = re.compile(
    r"\s*\.{3}\s*\[truncated(?:;[^\]]*)?\]\s*",
    flags=re.IGNORECASE,
)


class AdmissionController:
    def __init__(
        self,
        config: ConMemConfig,
        embedder: EmbeddingClient,
        storage: ConMemStorage,
        llm: LLMClient | None = None,
    ):
        self.config = config
        self.embedder = embedder
        self.storage = storage
        self.llm = llm

    def consistent(self, card: MemoryCard) -> bool:
        return self._reliability(card) >= self.config.admission_consistency_threshold

    def compute_admission_score(self, card: MemoryCard, current_round: int) -> float:
        """Q(c) = λ₁C(c) + λ₂N(c) + λ₃R(c) + λ₄U(c) (Section 4.5)."""
        reliability = self._reliability(card)
        novelty = self._novelty(card)
        relevance = self._round_relevance(card, current_round)
        utility = self._utility(card)

        card.quality.reliability = reliability
        card.quality.novelty = novelty
        card.quality.relevance = relevance
        card.quality.utility = utility

        return (
            self.config.admission_w_reliability * reliability
            + self.config.admission_w_novelty * novelty
            + self.config.admission_w_relevance * relevance
            + self.config.admission_w_utility * utility
        )

    def admit_cards(self, candidates: list[MemoryCard], current_round: int) -> list[MemoryCard]:
        admitted = []
        for card in candidates:
            if card.embedding is None:
                try:
                    card.embedding = self.embedder.embed(card.activation_text())
                except Exception as e:
                    logger.warning("Embedding failed for card %s: %s", card.card_id, e)
            score = self.compute_admission_score(card, current_round)
            card.metadata.admission_score = score
            if self.consistent(card) and score >= self.config.admission_threshold:
                admitted.append(card)
        return admitted

    def _reliability(self, card: MemoryCard) -> float:
        """C(c) — reliability: internal consistency + evidence support (Section 4.5).
        
        Formula from Implementation_Details.md:
        consistency = 0.3 + 0.35 * coverage + 0.35 * avg_overlap
        C(c) = 0.6 * consistency + 0.4 * evidence
        """
        # Internal consistency: section coverage + dependency chain overlap
        sections = {name: (card.get_section(name) or "").strip() for name in SECTION_ORDER}
        coverage = card.coverage()
        pair_scores = []
        for left, right in self.config.type_dependency_chain:
            left_text = sections.get(left, "")
            right_text = sections.get(right, "")
            if left_text and right_text:
                pair_scores.append(token_overlap(left_text, right_text))
        pair_mean = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5
        
        # Strictly follow Implementation_Details.md formula:
        # consistency = 0.3 + 0.35 * coverage + 0.35 * pair_mean
        consistency = max(
            0.0,
            min(
                1.0,
                self.config.reliability_base_constant
                + self.config.reliability_coverage_weight * coverage
                + self.config.reliability_dependency_weight * pair_mean,
            ),
        )

        # Evidence support: outcome-based credibility
        if not card.evidence.strip():
            evidence = self.config.admission_missing_evidence_score
        else:
            evidence = self.config.credibility_table.get(
                card.provenance.trajectory_outcome,
                self.config.retrieval_quality_fallback,
            )

        return (
            self.config.reliability_consistency_weight * consistency
            + self.config.reliability_evidence_weight * evidence
        )

    def _novelty(self, card: MemoryCard) -> float:
        existing = [
            other for other in self.storage.get_all_active_cards()
            if other.card_id != card.card_id and self._pattern_related(card, other)
        ]
        if not existing or card.embedding is None:
            return 1.0 if not existing else self.config.retrieval_quality_fallback
        max_sim = 0.0
        for other in existing:
            if other.embedding is None:
                continue
            max_sim = max(max_sim, cosine_similarity(card.embedding, other.embedding))
        return max(0.0, 1.0 - max_sim)

    def _round_relevance(self, card: MemoryCard, current_round: int) -> float:
        age = max(0, current_round - card.metadata.create_round)
        decay_constant = max(1e-6, self.config.get_round_decay_constant(card.task_domain or None))
        return math.exp(-age / decay_constant)

    def _utility(self, card: MemoryCard) -> float:
        """U(c) — utility: outcome-based usefulness + density bonus (Section 4.5)."""
        memory_type = card.primary_memory_type()
        base = self.config.utility_table.get(
            (memory_type, card.provenance.trajectory_outcome),
            self.config.utility_table.get(
                ("card", card.provenance.trajectory_outcome),
                self.config.retrieval_quality_fallback,
            ),
        )
        outcome_utility = min(1.0, base + self.config.utility_coverage_bonus * card.coverage())

        # Density bonus (folded from former separate density score)
        text = card.activation_text().lower()
        density = 1.0
        if len(text) < self.config.admission_density_short_text_chars:
            density *= self.config.admission_density_short_text_penalty
        if re.search(r"\d", text):
            density += self.config.admission_density_digit_bonus
        if re.search(r"\b(because|due to|caused by|therefore|failed|error)\b", text):
            density += self.config.admission_density_causal_bonus
        density = max(0.0, min(1.0, density))

        return (
            self.config.utility_outcome_weight * outcome_utility
            + self.config.utility_density_weight * density
        )

    def post_commit_merge(self, current_round: int):
        self._merge_redundant_committed()

    def _merge_redundant_committed(self):
        active_cards = sorted(
            self.storage.get_all_active_cards(),
            key=lambda card: card.metadata.admission_score,
            reverse=True,
        )
        if len(active_cards) < 2:
            return

        # Build inverted indexes once: term → card ids, cluster_key → card ids.
        # With cached pattern_features on MemoryCard, building this is O(N).
        term_to_cards: dict[str, set[str]] = defaultdict(set)
        cluster_to_cards: dict[str, set[str]] = defaultdict(set)
        cards_by_id: dict[str, MemoryCard] = {}
        rank: dict[str, int] = {}
        for i, card in enumerate(active_cards):
            cards_by_id[card.card_id] = card
            rank[card.card_id] = i
            cluster_to_cards[card.pattern_cluster_key].add(card.card_id)
            for term in card.pattern_terms():
                term_to_cards[term].add(card.card_id)

        min_overlap = self.config.cross_task_signature_min_overlap
        removed_ids: set[str] = set()

        for i, winner in enumerate(active_cards):
            if winner.card_id in removed_ids or winner.metadata.lifecycle_state != "active":
                continue

            # Candidates with ≥ min_overlap shared pattern terms.
            counts: Counter = Counter()
            for term in winner.pattern_terms():
                for cid in term_to_cards.get(term, ()):
                    counts[cid] += 1
            cand_ids = {cid for cid, n in counts.items() if n >= min_overlap}
            # Plus same-cluster cards (captures the pattern_cluster_key == branch).
            cand_ids |= cluster_to_cards.get(winner.pattern_cluster_key, set())
            # Only losers strictly later in sort order, and still active.
            winner_rank = i
            cand_ids = {
                cid for cid in cand_ids
                if cid != winner.card_id
                and cid not in removed_ids
                and rank.get(cid, -1) > winner_rank
            }
            if not cand_ids:
                continue

            for loser_id in cand_ids:
                loser = cards_by_id.get(loser_id)
                if loser is None or loser.card_id in removed_ids:
                    continue
                if loser.metadata.lifecycle_state != "active":
                    continue
                if not self._should_merge_cards(winner, loser):
                    continue
                if not self._merge_into(winner, loser):
                    continue
                self.storage.update_card(winner)
                self.storage.delete_card(loser.card_id)
                removed_ids.add(loser.card_id)

    def _merge_into(self, winner: MemoryCard, loser: MemoryCard) -> bool:
        """Rewrite two redundant cards into one bounded card with the LLM.

        This deliberately does not concatenate section text. If no LLM is
        available or the LLM returns an invalid payload, the merge is skipped so
        we do not delete the loser card and lose information.
        """
        if self.llm is None:
            logger.warning("Skipping card merge rewrite for %s <- %s: no LLM client", winner.card_id, loser.card_id)
            return False

        payload = self._llm_rewrite_merge(winner, loser)
        if payload is None:
            logger.warning("Skipping card merge rewrite for %s <- %s: invalid LLM payload", winner.card_id, loser.card_id)
            return False

        winner.structured_content = payload["structured_content"]
        winner.trigger_semantics = payload["trigger_semantics"]
        winner.evidence = payload["evidence"]
        winner.refresh_derived_fields()
        winner.summary = payload["summary"]
        self._invalidate_pattern_cache(winner)
        self._refresh_embedding(winner)
        return True

    def _llm_rewrite_merge(self, winner: MemoryCard, loser: MemoryCard) -> dict | None:
        summary_limit = max(80, int(getattr(self.config, "extract_existing_summary_preview_chars", 160)))
        max_input_chars = max(600, int(getattr(self.config, "llm_max_input_chars", 120000) // 8))

        cards_text = "\n\n".join(
            [
                self._format_card_for_merge("winner", winner, max_input_chars),
                self._format_card_for_merge("candidate", loser, max_input_chars),
            ]
        )
        user_prompt = MERGE_CONTENT_USER.format(cards_text=cards_text)

        try:
            raw = self.llm.chat_json(
                MERGE_CONTENT_SYSTEM,
                user_prompt,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("LLM card merge rewrite failed for %s <- %s: %s", winner.card_id, loser.card_id, exc)
            return None

        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("merged_card"), dict):
            raw = raw["merged_card"]

        structured_raw = raw.get("structured_content")
        if not isinstance(structured_raw, dict):
            return None

        sections = {}
        for section in SECTION_ORDER:
            text = self._bounded_text(structured_raw.get(section, ""), 0)
            if text:
                sections[section] = text
        if not sections:
            return None

        triggers_raw = raw.get("trigger_semantics")
        if not isinstance(triggers_raw, list):
            return None
        triggers = []
        for item in triggers_raw:
            text = self._bounded_text(item, 80)
            if text and text not in triggers:
                triggers.append(text)
            if len(triggers) >= self.config.trigger_max_semantics:
                break
        if not triggers:
            return None

        summary = self._bounded_text(raw.get("summary", ""), summary_limit)
        if not summary:
            return None

        return {
            "structured_content": CardContent.from_dict(sections),
            "trigger_semantics": triggers,
            "summary": summary,
            "evidence": self._sanitize_evidence(raw.get("evidence", ""), winner.task_domain or loser.task_domain),
        }

    def _format_card_for_merge(self, role: str, card: MemoryCard, max_chars: int) -> str:
        text = "\n".join(
            [
                f"<card role=\"{role}\" id=\"{card.card_id}\">",
                f"Summary: {self._bounded_text(card.summary, 240)}",
                f"Triggers: {'; '.join(card.trigger_semantics[: self.config.trigger_max_semantics])}",
                "Content:",
                self._bounded_text(card.content, max_chars),
                f"Evidence: {self._bounded_text(card.evidence, 240)}",
                "</card>",
            ]
        )
        return text

    def _bounded_text(self, value, limit: int) -> str:
        text = TRUNCATION_MARKER_RE.sub(" ", str(value or "")).strip()
        text = re.sub(r"\s+", " ", text)
        if limit <= 0:
            return text
        if len(text) <= limit:
            return text
        return text[:limit].rstrip()

    def _sanitize_evidence(self, value, task_domain: str = "") -> str:
        text = self._bounded_text(value, 0)
        if not text:
            return ""
        text = " ".join(part.strip() for part in text.split("|") if part.strip())
        if task_domain in {"triviaqa", "popqa"}:
            lowered = text.lower()
            if "failure" in lowered and "success" in lowered:
                return (
                    "success: the strategy produced a source-verified answer; "
                    "failure: unsupported or conflicting evidence requires query refinement."
                )
            if "failure" in lowered:
                return "failure: the attempt lacked sufficient source verification; recovery requires query refinement."
            return "success: the strategy produced a source-verified answer without storing the answer itself."
        return self._clip_sentence(text, 360)

    def _clip_sentence(self, text: str, limit: int) -> str:
        text = self._bounded_text(text, 0)
        if limit <= 0 or len(text) <= limit:
            return text
        clipped = text[:limit].rstrip()
        match = re.search(r"^(.+[.!?])\s+[^.!?]*$", clipped)
        return match.group(1).strip() if match else clipped

    def _refresh_embedding(self, card: MemoryCard):
        try:
            card.embedding = self.embedder.embed(card.activation_text())
        except Exception as exc:
            logger.warning("Failed to refresh embedding after card merge rewrite for %s: %s", card.card_id, exc)
            card.embedding = None

    def _invalidate_pattern_cache(self, card: MemoryCard):
        card._cached_pattern_features = None
        card._cached_pattern_terms = None
        card._cached_pattern_cluster_key = None

    def _should_merge_cards(self, winner: MemoryCard, loser: MemoryCard) -> bool:
        if winner.card_id == loser.card_id:
            return False
        if winner.embedding is None or loser.embedding is None:
            return False
        if not self._pattern_related(winner, loser):
            return False

        similarity = cosine_similarity(winner.embedding, loser.embedding)
        if similarity < self.config.cross_task_merge_threshold:
            return False

        trigger_overlap = token_overlap(
            " ".join(winner.trigger_semantics + [winner.summary]),
            " ".join(loser.trigger_semantics + [loser.summary]),
        )
        section_overlap = token_overlap(winner.content, loser.content)
        shared_terms = len(winner.pattern_terms() & loser.pattern_terms())

        return (
            similarity >= self.config.commit_merge_threshold
            or trigger_overlap >= self.config.admission_merge_trigger_overlap_threshold
            or section_overlap >= self.config.admission_merge_section_overlap_threshold
            or shared_terms >= self.config.cross_task_signature_min_overlap
        )

    def _pattern_related(self, left: MemoryCard, right: MemoryCard) -> bool:
        if not self._same_domain(left, right):
            return False
        if left.pattern_cluster_key == right.pattern_cluster_key:
            return True
        overlap = left.pattern_terms() & right.pattern_terms()
        return len(overlap) >= self.config.cross_task_signature_min_overlap

    def _same_domain(self, left: MemoryCard, right: MemoryCard) -> bool:
        if not self.config.enforce_task_domain_filter:
            return True
        if not left.task_domain or not right.task_domain:
            return True
        return left.task_domain == right.task_domain

    def _idle_round(self, card: MemoryCard, current_round: int) -> int:
        if card.metadata.last_access_round > 0:
            return current_round - card.metadata.last_access_round
        return current_round - card.metadata.create_round

    def check_graph_explosion(self, current_round: int):
        if self.storage.count_active_cards() > self.config.max_graph_nodes:
            self.post_commit_merge(current_round)
