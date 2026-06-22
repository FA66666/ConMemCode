"""Trajectory aggregation into unified strategy cards (Section 4.2.3).

Aggregate(τ_new, C) — builds candidate cards from trajectory,
aligned with existing card set C.
"""
import logging
import re
import time
import uuid

from .config import ConMemConfig
from .llm_backend import LLMClient, estimate_tokens
from .prompts import (
    TRAJECTORY_COMPRESS_SYSTEM,
    TRAJECTORY_COMPRESS_USER,
    MEMORY_EXTRACT_SYSTEM,
    MEMORY_EXTRACT_USER,
    FAILURE_REFLECT_SYSTEM,
    FAILURE_REFLECT_USER,
)
from .schema import CardContent, CardMetadata, MemoryCard, Provenance

logger = logging.getLogger(__name__)


def trajectory_to_text(trajectory_data: dict) -> str:
    lines = [
        f"Task: {trajectory_data.get('task_description', 'N/A')}",
        f"Outcome: {trajectory_data.get('outcome', 'unknown')}",
        "",
    ]
    for step in trajectory_data.get("steps", []):
        lines.append(f"--- Step {step.get('step_index', '?')} ---")
        if step.get("agent"):
            lines.append(f"Agent: {step['agent']}")
        if step.get("input"):
            lines.append(f"Input: {step['input']}")
        if step.get("output"):
            lines.append(f"Output: {step['output']}")
        if step.get("tool_calls"):
            lines.append(f"Tool Calls: {step['tool_calls']}")
        if step.get("feedback"):
            lines.append(f"Feedback: {step['feedback']}")
        for item in step.get("agent_interactions", []):
            lines.append(f"  [{item.get('agent', '?')}]: {item.get('response', '')}")
        lines.append("")
    return "\n".join(lines)


def mas_trajectory_to_dict(trajectory, outcome: str = "") -> dict:
    if not outcome:
        if trajectory.label is True:
            outcome = "success"
        elif trajectory.label is False:
            outcome = "failure"
        else:
            outcome = "partial"

    steps = []
    for i, mg in enumerate(trajectory.trajectory or []):
        step = {
            "step_index": i + 1,
            "agent": "",
            "input": mg.state or "",
            "output": mg.action or "",
            "tool_calls": "",
            "feedback": mg.observation or "",
            "agent_interactions": [],
        }
        if mg.mas_message_graph is not None:
            for node_id in mg.mas_message_graph.nodes():
                node_data = mg.mas_message_graph.nodes[node_id]
                msg = node_data.get("message")
                if not msg:
                    continue
                step["agent_interactions"].append(
                    {
                        "agent": str(node_id),
                        "response": msg.response or "",
                        "state": msg.state or {},
                    }
                )
        steps.append(step)

    return {
        "task_description": trajectory.task_init_description or "",
        "outcome": outcome,
        "steps": steps,
    }


class TrajectoryInterpreter:
    def __init__(self, config: ConMemConfig, llm: LLMClient, task_domain: str | None = None):
        self.config = config
        self.llm = llm
        self.task_domain = task_domain

    def interpret(
        self,
        trajectory_data: dict,
        task_id: str,
        current_round: int,
        existing_cards: list[MemoryCard] | None = None,
    ) -> list[MemoryCard]:
        """Aggregate(τ_new, C) — extract cards aligned with existing set (Section 4.2.3)."""
        task_desc = trajectory_data.get("task_description", "")
        outcome = trajectory_data.get("outcome", "unknown")
        traj_text = trajectory_to_text(trajectory_data)
        if estimate_tokens(traj_text) > self.config.trajectory_max_tokens:
            traj_text = self._compress_trajectory(task_desc, traj_text)

        # Failure reflection: analyze root cause before card extraction
        reflection = ""
        reflection_quality = 1.0
        if self.config.enable_failure_reflection and outcome in ("failure", "partial"):
            reflection = self._reflect_on_failure(task_desc, traj_text)
            if reflection:
                traj_text = f"{traj_text}\n\n--- Failure Reflection ---\n{reflection}"
            reflection_quality = self._reflection_quality(reflection)

        raw_cards = self._extract_cards(task_desc, outcome, traj_text, existing_cards)
        if not raw_cards:
            return []

        now = time.time()
        cards = []
        for i, raw in enumerate(raw_cards):
            logger.debug(f"Processing raw card {i}: {raw.keys() if isinstance(raw, dict) else 'not dict'}")
            sections = CardContent.from_dict(raw.get("structured_content"))
            logger.debug(f"Card {i} sections has_any={sections.has_any()}")
            if not sections.has_any():
                logger.warning(f"Card {i} rejected: empty sections (structured_content={raw.get('structured_content')})")
                continue
            raw_triggers = raw.get("trigger_semantics", [])
            if isinstance(raw_triggers, str):
                raw_triggers = [raw_triggers]
            trigger_semantics = self._normalize_triggers(raw_triggers, task_desc, sections.section_map())
            if len(trigger_semantics) < max(1, min(2, self.config.trigger_max_semantics)):
                derived_triggers = self._derive_triggers(task_desc, sections.section_map(), outcome)
                trigger_semantics = self._normalize_triggers(
                    trigger_semantics + derived_triggers,
                    task_desc,
                    sections.section_map(),
                )
            if not trigger_semantics:
                trigger_semantics = [self._fallback_trigger(sections.section_map(), raw.get("summary", ""))]
            cards.append(
                MemoryCard(
                    card_id=str(uuid.uuid4()),
                    task_id=task_id,
                    task_domain=self.task_domain or "",
                    task_description=task_desc,
                    structured_content=sections,
                    trigger_semantics=trigger_semantics,
                    summary=self._clean_one_line(raw.get("summary", "")),
                    evidence=self._sanitize_evidence(raw.get("evidence", ""), outcome),
                    provenance=Provenance(
                        source_task_id=task_id,
                        source_agent=raw.get("source_agent", ""),
                        source_step_indices=raw.get("source_steps", []),
                        trajectory_outcome=outcome,
                        reflection_quality=reflection_quality,
                    ),
                    metadata=CardMetadata(
                        timestamp=now,
                        lifecycle_state="active",
                        access_count=0,
                        last_access_time=now,
                        admission_score=0.0,
                        create_round=current_round,
                        last_access_round=current_round,
                    ),
                )
            )
        return cards

    def _compress_trajectory(self, task_desc: str, traj_text: str) -> str:
        try:
            result = self.llm.chat(
                TRAJECTORY_COMPRESS_SYSTEM.format(max_tokens=self.config.trajectory_max_tokens),
                TRAJECTORY_COMPRESS_USER.format(task_description=task_desc, trajectory_text=traj_text),
            )
            # Extract from <compressed_trajectory> tag if present
            tag_match = re.search(r'<compressed_trajectory>(.*?)(?:</compressed_trajectory>|$)', result, re.DOTALL)
            if tag_match:
                result = tag_match.group(1)
            return result.strip()
        except Exception:
            return self._structured_truncate(traj_text, self.config.trajectory_max_tokens)

    def _structured_truncate(self, traj_text: str, max_tokens: int) -> str:
        chars_per_token = max(1, int(self.config.trajectory_chars_per_token_estimate))
        return traj_text[: max_tokens * chars_per_token]

    def _reflect_on_failure(self, task_desc: str, traj_text: str) -> str:
        """LLM reflection on why the task failed. Returns reflection text or empty string."""
        try:
            domain_guidance = self._domain_reflection_guidance()
            if domain_guidance:
                task_desc = f"{task_desc}\n[Domain guidance] {domain_guidance}"
            result = self.llm.chat(
                FAILURE_REFLECT_SYSTEM,
                FAILURE_REFLECT_USER.format(task_description=task_desc, trajectory_text=traj_text),
            )
            tag_match = re.search(r'<reflection>(.*?)(?:</reflection>|$)', result, re.DOTALL)
            if tag_match:
                result = tag_match.group(1)
            return result.strip()
        except Exception as e:
            logger.warning("Failure reflection failed: %s", e)
            return ""

    def _reflection_quality(self, reflection: str) -> float:
        """Lightweight structural score for generated failure reflections."""
        if not reflection.strip():
            return 0.0
        text = reflection.lower()
        required_markers = (
            "root cause:",
            "what went wrong:",
            "what should have been done:",
            "general lesson:",
        )
        structure_score = sum(marker in text for marker in required_markers) / len(required_markers)
        length_score = min(1.0, len(reflection.split()) / 40.0)
        return 0.8 * structure_score + 0.2 * length_score

    def _extract_cards(
        self,
        task_desc: str,
        outcome: str,
        traj_text: str,
        existing_cards: list[MemoryCard] | None = None,
    ) -> list[dict]:
        # Build existing cards summary for alignment with C
        existing_cards_section = ""
        if existing_cards:
            summaries = []
            for i, card in enumerate(existing_cards[: self.config.extract_existing_card_examples]):
                sections = card.structured_content.non_empty_sections()
                parts = [
                    f"  - {k}: {v[: self.config.extract_existing_section_preview_chars]}"
                    for k, v in sections.items()
                ]
                summaries.append(
                    f"Card {i+1}: {card.summary[: self.config.extract_existing_summary_preview_chars]}\n"
                    + "\n".join(parts)
                )
            existing_cards_section = (
                "\n## Existing Cards (avoid duplicating these)\n"
                + "\n\n".join(summaries)
                + "\n"
            )
        domain_guidance = self._domain_card_guidance()
        if domain_guidance:
            existing_cards_section += "\n## Domain Guidance\n" + domain_guidance + "\n"

        try:
            result = self.llm.chat_json(
                MEMORY_EXTRACT_SYSTEM,
                MEMORY_EXTRACT_USER.format(
                    task_description=task_desc,
                    outcome=outcome,
                    trajectory_text=traj_text,
                    existing_cards_section=existing_cards_section,
                ),
            )
            logger.info(f"LLM extraction result type={type(result).__name__}, content={str(result)[:200]}")
            if isinstance(result, dict):
                result = [result]
            if isinstance(result, list):
                valid_cards = []
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    sc = item.get("structured_content")
                    if isinstance(sc, str) and sc.strip():
                        logger.warning("Skipping extracted card with string structured_content; expected a section dict.")
                        continue
                    if not isinstance(sc, dict):
                        continue
                    if self._reject_overly_specific_card(task_desc, item):
                        logger.info("Skipping overly task-specific extracted card for domain %s", self.task_domain)
                        continue
                    valid_cards.append(item)
                logger.info(f"Validated {len(valid_cards)} cards from {len(result)} items")
                return valid_cards
        except Exception as e:
            logger.warning("Unified extraction failed: %s. No cards extracted.", e)
            return []

    def _domain_card_guidance(self) -> str:
        if self.task_domain in {"triviaqa", "popqa"}:
            return (
                "- Do not store the answer, entity identity, or source-specific facts as reusable memory.\n"
                "- Prefer cards about query formulation, retrieval sequencing, evidence verification, and answer normalization.\n"
                "- If a lesson only helps this exact question, omit it."
            )
        if self.task_domain == "pddl":
            return (
                "- Prefer reusable cards about goal decomposition, valid-action discovery, precondition checks, and recovery from invalid actions.\n"
                "- Abstract away object names and exact literals; preserve the planning policy."
            )
        return ""

    def _domain_reflection_guidance(self) -> str:
        if self.task_domain in {"triviaqa", "popqa"}:
            return (
                "Explain retrieval or verification mistakes without revealing the gold answer. "
                "Focus on transferable search and evidence-checking lessons."
            )
        if self.task_domain == "pddl":
            return (
                "Explain missed preconditions, state cues, or action-selection heuristics. "
                "Do not turn the reflection into an object-specific plan transcript."
            )
        return ""

    def _reject_overly_specific_card(self, task_desc: str, raw_card: dict) -> bool:
        if self.task_domain not in {"triviaqa", "popqa"}:
            return False
        sections = CardContent.from_dict(raw_card.get("structured_content"))
        text = " ".join(
            part for part in (
                raw_card.get("summary", ""),
                raw_card.get("evidence", ""),
                sections.compose(),
            )
            if part
        ).lower()
        if not text:
            return False
        if any(marker in text for marker in ("the answer is", "final answer", "<answer>", "correct answer")):
            return True
        strategy_markers = (
            "search", "retrieve", "evidence", "verify", "verification", "normalize",
            "alias", "disambigu", "source", "query", "document", "lookup",
        )
        if any(marker in text for marker in strategy_markers):
            return False
        question_terms = set(re.findall(self.config.token_regex(), task_desc.lower()))
        card_terms = set(re.findall(self.config.token_regex(), text))
        if question_terms and len(question_terms & card_terms) / len(question_terms) >= 0.6:
            return True
        return False

    def _normalize_triggers(
        self,
        raw_triggers: list[str],
        task_desc: str,
        sections: dict[str, str],
    ) -> list[str]:
        triggers = []
        for item in raw_triggers or []:
            text = self._clean_one_line(item).strip(" -;,.")
            if not text:
                continue
            if len(text) > 100:
                continue
            if not self._looks_like_use_condition(text):
                continue
            if self._is_task_specific_trigger(text, task_desc):
                continue
            if self._is_section_sentence(text, sections):
                continue
            if text not in triggers:
                triggers.append(text)
            if len(triggers) >= self.config.trigger_max_semantics:
                break
        return triggers

    def _looks_like_use_condition(self, text: str) -> bool:
        lowered = text.lower().strip()
        if "?" in text:
            return False
        if re.match(r"^(first|then|next|finally|success|failure|verify|check|use|initialize|return)\b", lowered):
            return False
        condition_markers = (
            "when ", "for ", "if ", "while ", "during ", "question about ",
            "questions about ", "task requiring ", "tasks requiring ",
            "problem involving ", "problems involving ", "case where ", "cases where ",
        )
        if lowered.startswith(condition_markers):
            return True
        # Allow concise noun phrases when they are more specific than a single keyword.
        return 2 <= len(self._tokenize(text)) <= 8

    def _is_task_specific_trigger(self, trigger: str, task_desc: str) -> bool:
        trigger_norm = self._normalize_for_compare(trigger)
        task_norm = self._normalize_for_compare(task_desc)
        if not trigger_norm:
            return True
        if trigger_norm == task_norm or trigger_norm in task_norm:
            return True
        trigger_tokens = set(self._tokenize(trigger))
        task_tokens = set(self._tokenize(task_desc))
        if len(trigger_tokens) >= 4 and task_tokens:
            if len(trigger_tokens & task_tokens) / len(trigger_tokens) >= 0.65:
                return True
        return False

    def _is_section_sentence(self, trigger: str, sections: dict[str, str]) -> bool:
        trigger_norm = self._normalize_for_compare(trigger)
        if not trigger_norm:
            return False
        for text in sections.values():
            for sentence in re.split(r"(?<=[.!?;])\s+", text or ""):
                sentence_norm = self._normalize_for_compare(sentence)
                if sentence_norm and trigger_norm == sentence_norm:
                    return True
        return False

    def _sanitize_evidence(self, value, outcome: str) -> str:
        text = self._clean_one_line(value)
        if not text:
            return ""
        text = " ".join(part.strip() for part in text.split("|") if part.strip())
        if self.task_domain in {"triviaqa", "popqa"}:
            lowered = text.lower()
            if "failure" in lowered and "success" in lowered:
                return (
                    "success: the strategy produced a source-verified answer; "
                    "failure: unsupported or conflicting evidence requires query refinement."
                )
            if outcome in {"failure", "partial"} or "failure" in lowered:
                return "failure: the attempt lacked sufficient source verification; recovery requires query refinement."
            return "success: the strategy produced a source-verified answer without storing the answer itself."
        return self._clip_sentence(text, 360)

    def _fallback_trigger(self, sections: dict[str, str], summary: str) -> str:
        for key in ("state", "plan", "eval"):
            text = self._clean_one_line(sections.get(key, ""))
            if text:
                if text.lower().startswith("when "):
                    return self._clip_sentence(text, 100).strip(" -;,.")
                return self._clip_sentence(f"when {text[0].lower()}{text[1:]}", 100).strip(" -;,.")
        summary = self._clean_one_line(summary)
        if summary:
            return self._clip_sentence(f"when applying this strategy: {summary}", 100).strip(" -;,.")
        return "when a similar reusable strategy is needed"

    def _clean_one_line(self, value) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _clip_sentence(self, text: str, limit: int) -> str:
        text = self._clean_one_line(text)
        if limit <= 0 or len(text) <= limit:
            return text
        clipped = text[:limit].rstrip()
        match = re.search(r"^(.+[.!?])\s+[^.!?]*$", clipped)
        return match.group(1).strip() if match else clipped

    def _normalize_for_compare(self, text: str) -> str:
        text = self._clean_one_line(text).lower()
        text = re.sub(r"[^a-z0-9_ ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(self.config.token_regex(), self._clean_one_line(text).lower())

    def _derive_triggers(self, task_desc: str, sections: dict[str, str], outcome: str) -> list[str]:
        """Derive generalizable trigger semantics (Section 3.1 Implementation_Details.md).

        Formula:
        chi = [generalize(z_first_clause)] plus top-k generalized section sentences
        from plan, eval, and state when the sentence length is at least ell_min.
        """
        triggers = []
        candidates = []  # (priority, trigger) tuples for ranking

        # Collect only section sentences that already describe a use condition.
        section_priority = dict(getattr(self.config, "trigger_section_priority", {}))
        for key in ("state", "plan", "eval"):
            text = (sections.get(key) or "").strip()
            priority = float(section_priority.get(key, 0.0))
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = sentence.strip()
                if len(sentence) >= self.config.trigger_min_sentence_chars:
                    generalized = self._generalize_text(sentence[: self.config.trigger_sentence_clip_chars])
                    if generalized and self._looks_like_use_condition(generalized):
                        norm = max(1, self.config.trigger_sentence_length_norm_chars)
                        score = priority + min(len(sentence), norm) / float(norm)
                        candidates.append((score, generalized))

        # 3. Sort by score and select top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, trigger in candidates[: self.config.trigger_max_semantics]:
            if trigger not in triggers:
                triggers.append(trigger)
            if len(triggers) >= self.config.trigger_max_semantics:
                break

        return list(dict.fromkeys(triggers))[: self.config.trigger_max_semantics]

    def _generalize_text(self, text: str) -> str:
        """Strip task-specific identifiers to make text more generalizable."""
        # Remove quoted strings (specific values)
        text = re.sub(r'"[^"]*"', '"..."', text)
        text = re.sub(r"'[^']*'", "'...'", text)
        # Remove file paths
        text = re.sub(r'[/\\][\w./\\-]+\.\w+', '<file>', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '<url>', text)
        # Collapse extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
