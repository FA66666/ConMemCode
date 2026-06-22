"""
ConMem Task Identification (Section 4.2.4).

Hierarchical task identification:
  Step 1 — Explicit ID priority
  Step 2 — LLM-based task extraction
  Step 3 — Embedding-based deduplication
  Step 4 — Task completion detection
"""
import logging
import uuid
from typing import Optional

from .config import ConMemConfig
from .llm_backend import LLMClient, EmbeddingClient, cosine_similarity
from .prompts import TASK_EXTRACT_SYSTEM, TASK_EXTRACT_USER
from .schema import TaskRecord
from .storage import ConMemStorage

logger = logging.getLogger(__name__)


class TaskIdentifier:
    """Hierarchical task identification mechanism."""

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

    def identify_task(
        self,
        event_text: str,
        explicit_task_id: Optional[str] = None,
        explicit_task_desc: Optional[str] = None,
        task_domain: str | None = None,
    ) -> tuple[str, str]:
        """
        Identify the current task. Returns (task_id, task_description).

        Step 1: If explicit_task_id provided, use it directly.
        Step 2: Extract task description via LLM.
        Step 3: Deduplicate against historical tasks via embedding similarity.
        """
        # Step 1 — Explicit ID priority
        if explicit_task_id:
            task = self.storage.get_task(explicit_task_id)
            if task:
                return task.task_id, task.task_description
            desc = explicit_task_desc or event_text
            return explicit_task_id, desc

        # Step 2 — Use explicit description if available, skip LLM extraction
        task_desc = explicit_task_desc
        if not task_desc:
            # Only call LLM if no description provided at all
            task_desc = self._extract_task_description(event_text)

        # Step 3 — Embedding-based deduplication
        task_id = self._deduplicate_task(task_desc, task_domain=task_domain)
        return task_id, task_desc

    def _extract_task_description(self, event_text: str) -> str:
        """ExtractTask(e_t) — LLM-based task extraction (Step 2)."""
        import re
        desc = self.llm.chat(
            TASK_EXTRACT_SYSTEM,
            TASK_EXTRACT_USER.format(event_text=event_text),
        )
        # Extract from <task_description> tag if present
        tag_match = re.search(r'<task_description>(.*?)(?:</task_description>|$)', desc, re.DOTALL)
        if tag_match:
            desc = tag_match.group(1)
        return desc.strip()

    def _deduplicate_task(self, task_desc: str, task_domain: str | None = None) -> str:
        """
        Embedding-based deduplication (Step 3).

        If max similarity >= nu_task, reuse existing task_id;
        otherwise create a new task.
        """
        try:
            query_emb = self.embedder.embed(task_desc)
        except Exception as e:
            logger.warning(f"Embedding failed during task dedup: {e}. Creating new task.")
            return str(uuid.uuid4())

        tasks = self.storage.get_all_tasks(task_domain=task_domain)
        if not tasks:
            return str(uuid.uuid4())

        best_sim = -1.0
        best_task_id = None
        for task in tasks:
            if task.embedding is None:
                continue
            sim = cosine_similarity(query_emb, task.embedding)
            if sim > best_sim:
                best_sim = sim
                best_task_id = task.task_id

        if best_sim >= self.config.task_dedup_threshold and best_task_id:
            logger.info(
                f"Task deduplicated: reusing task {best_task_id} (sim={best_sim:.3f})"
            )
            return best_task_id

        return str(uuid.uuid4())

    def register_task(
        self,
        task_id: str,
        task_description: str,
        outcome: str = "",
        current_round: int = 0,
        task_domain: str | None = None,
    ) -> TaskRecord:
        """Register a new task or update an existing one in the task registry."""
        existing = self.storage.get_task(task_id)
        if existing:
            if task_domain and not existing.task_domain:
                existing.task_domain = task_domain
            if outcome:
                existing.outcome = outcome
                existing.completion_round = current_round
            self.storage.insert_task(existing)
            return existing

        try:
            emb = self.embedder.embed(task_description)
        except Exception as e:
            logger.warning(f"Embedding failed for task registration: {e}")
            emb = None

        task = TaskRecord(
            task_id=task_id,
            task_domain=task_domain or "",
            task_description=task_description,
            outcome=outcome,
            completion_round=current_round,
            embedding=emb,
        )
        self.storage.insert_task(task)
        return task
