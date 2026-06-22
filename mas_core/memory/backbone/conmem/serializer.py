"""Compose coordinated unified cards into reasoning context."""
import logging

from .config import ConMemConfig
from .llm_backend import estimate_tokens
from .schema import MemoryCard, SECTION_ORDER
from .utils import normalize_role

logger = logging.getLogger(__name__)


SECTION_TAGS = {
    "state": "task_state",
    "plan": "strategy_plan",
    "exec": "execution_path",
    "eval": "evaluation",
}


class MemorySerializer:
    def __init__(self, config: ConMemConfig):
        self.config = config
        self.last_rendered_hint_count: int = 0
        self.last_serialized_token_count: int = 0
        self.last_considered_card_count: int = 0
        self.last_skipped_over_budget_count: int = 0
        self.last_compacted_card_count: int = 0

    def serialize(self, selected_cards: list[MemoryCard], agent_role: str = "default") -> str:
        """Serialize each selected card once as compact, optional guidance."""
        self.last_rendered_hint_count = 0
        self.last_serialized_token_count = 0
        self.last_considered_card_count = 0
        self.last_skipped_over_budget_count = 0
        self.last_compacted_card_count = 0
        if not selected_cards:
            return ""

        role_key = normalize_role(agent_role)
        cards = sorted(selected_cards, key=lambda card: card.metadata.admission_score, reverse=True)
        lines = [
            '<conmem_hints use="optional">',
            '  <rule>Current task and environment feedback are authoritative. Ignore any hint that does not match this task.</rule>',
        ]
        total_tokens = estimate_tokens("\n".join(lines) + "\n</conmem_hints>")
        max_hints = max(1, self.config.serializer_max_cards_per_section)
        rendered = 0
        considered = 0
        skipped_over_budget = 0
        compacted = 0

        for card in cards:
            if rendered >= max_hints:
                break
            block = self._render_card_hint(card, role_key)
            if not block:
                continue
            considered += 1
            block_tokens = estimate_tokens(block)
            if total_tokens + block_tokens > self.config.token_budget:
                compact_block = self._render_card_hint(card, role_key, compact=True)
                block_tokens = estimate_tokens(compact_block)
                if not compact_block or total_tokens + block_tokens > self.config.token_budget:
                    skipped_over_budget += 1
                    continue
                block = compact_block
                compacted += 1
            lines.append(block)
            total_tokens += block_tokens
            rendered += 1

        self.last_considered_card_count = considered
        self.last_skipped_over_budget_count = skipped_over_budget
        self.last_compacted_card_count = compacted
        if rendered == 0:
            return ""
        self.last_rendered_hint_count = rendered
        lines.append('</conmem_hints>')
        serialized = "\n".join(lines)
        self.last_serialized_token_count = estimate_tokens(serialized)
        return serialized

    def _render_card_hint(self, card: MemoryCard, role_key: str, compact: bool = False) -> str:
        max_chars = (
            self.config.serializer_max_card_chars
            if not compact else min(self.config.serializer_max_card_chars, self.config.serializer_compact_card_chars)
        )
        section_budget = (
            self.config.serializer_full_section_budget
            if not compact else self.config.serializer_compact_section_budget
        )
        trigger_chars = (
            self.config.serializer_full_trigger_chars
            if not compact else self.config.serializer_compact_trigger_chars
        )
        when_chars = (
            self.config.serializer_full_when_chars
            if not compact else self.config.serializer_compact_when_chars
        )
        check_chars = (
            self.config.serializer_full_check_chars
            if not compact else self.config.serializer_compact_check_chars
        )
        triggers = self._truncate_text(
            '; '.join(card.trigger_semantics[: self.config.serializer_trigger_examples]) or card.summary,
            trigger_chars,
        )
        when = self._first_nonempty(triggers, card.summary, card.task_description)
        when = self._truncate_text(when, when_chars)

        warning_only = self._warning_only_card(card)
        do_items = [] if warning_only else self._select_guidance_sections(card, role_key, section_budget, max_chars)
        check_text = "" if warning_only else self._truncate_text(card.get_section("eval") or "", check_chars)
        avoid_text = self._avoid_text(card, compact)

        if not do_items and not check_text and not avoid_text:
            fallback = self._truncate_text(card.summary or card.content, max_chars)
            if fallback:
                if warning_only:
                    avoid_text = fallback
                else:
                    do_items = [fallback]
        if not do_items and not check_text and not avoid_text:
            return ""

        attrs = [f'id="{self._escape_xml(card.card_id[: self.config.serializer_card_id_chars])}"']
        outcome = (card.provenance.trajectory_outcome or "").strip()
        if outcome:
            attrs.append(f'outcome="{self._escape_xml(outcome)}"')
        score = card.metadata.admission_score
        if score:
            attrs.append(f'score="{score:.2f}"')

        lines = [f'  <hint {" ".join(attrs)}>']
        if when:
            lines.append(f'    <when>{self._escape_xml(when)}</when>')
        for item in do_items:
            lines.append(f'    <do>{self._escape_xml(item)}</do>')
        if check_text and check_text not in do_items:
            lines.append(f'    <check>{self._escape_xml(check_text)}</check>')
        if avoid_text:
            lines.append(f'    <avoid>{self._escape_xml(avoid_text)}</avoid>')
        if card.conflict_warning:
            lines.append(
                f'    <warning>{self._escape_xml(self._truncate_text(card.conflict_warning, self.config.serializer_warning_chars))}</warning>'
            )
        lines.append('  </hint>')
        return "\n".join(lines)

    def _select_guidance_sections(self, card: MemoryCard, role_key: str, limit: int, max_chars: int) -> list[str]:
        allocation = self.config.role_budget_allocation.get(
            role_key, self.config.role_budget_allocation["default"]
        )
        priority = sorted(SECTION_ORDER, key=lambda name: allocation.get(name, 0), reverse=True)
        items: list[str] = []
        for section in priority:
            if section == "eval":
                continue
            text = (card.get_section(section) or "").strip()
            if not text:
                continue
            text = self._truncate_text(text, max_chars)
            if text and text not in items:
                items.append(text)
            if len(items) >= limit:
                break
        return items

    def _avoid_text(self, card: MemoryCard, compact: bool) -> str:
        eval_text = (card.get_section("eval") or "").strip()
        if not eval_text:
            return ""
        lowered = eval_text.lower()
        has_negative = any(token in lowered for token in ("failure", "fail", "avoid", "do not", "incorrect", "bug", "error", "recovery"))
        if not has_negative and not self._warning_only_card(card) and card.provenance.trajectory_outcome != "failure":
            return ""
        check_chars = (
            self.config.serializer_full_check_chars
            if not compact else self.config.serializer_compact_check_chars
        )
        return self._truncate_text(eval_text, check_chars)

    def _warning_only_card(self, card: MemoryCard) -> bool:
        return False

    def _first_nonempty(self, *texts: str) -> str:
        for text in texts:
            cleaned = (text or "").strip()
            if cleaned:
                return cleaned
        return ""

    def _render_section(self, section: str, cards: list[MemoryCard], budget: int) -> str:
        """Deprecated compatibility hook; serialize() now emits card-centric hints."""
        return self._render_section_xml(section, cards, budget)

    def _render_section_xml(self, section: str, cards: list[MemoryCard], budget: int) -> str:
        """Deprecated section renderer retained for callers/tests that exercise it directly."""
        section_tag = SECTION_TAGS.get(section, section)
        lines = [f"  <{section_tag}>"]
        used = estimate_tokens(lines[0]) + estimate_tokens(f"  </{section_tag}>")
        cards = sorted(cards, key=lambda card: card.metadata.admission_score, reverse=True)
        rendered_count = 0
        max_cards = max(1, self.config.serializer_max_cards_per_section)

        for card in cards:
            if rendered_count >= max_cards:
                break
            text = (card.get_section(section) or "").strip()
            if not text:
                continue
            text = self._truncate_text(text, self.config.serializer_max_card_chars)
            block_lines = ["    <card>", f"      <content>{self._escape_xml(text)}</content>"]
            if card.trigger_semantics:
                triggers = '; '.join(card.trigger_semantics[: self.config.serializer_trigger_examples])
                block_lines.append(
                    f"      <trigger>{self._escape_xml(self._truncate_text(triggers, self.config.serializer_full_trigger_chars))}</trigger>"
                )
            if card.conflict_warning:
                block_lines.append(f"      <warning>{self._escape_xml(card.conflict_warning)}</warning>")
            block_lines.append("    </card>")
            rendered_block = "\n".join(block_lines)
            block_tokens = estimate_tokens(rendered_block)
            if used + block_tokens > budget:
                continue
            lines.append(rendered_block)
            used += block_tokens
            rendered_count += 1

        lines.append(f"  </{section_tag}>")
        return "\n".join(lines) if len(lines) > 2 else ""

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
        return (truncated or text[:max_chars]).rstrip() + "..."

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
