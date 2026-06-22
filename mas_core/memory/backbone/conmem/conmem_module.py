"""
ConMem Module — Main Orchestrator (Section 4.6 Host MAS Interface).

Implements the full pipeline:
  Update-side: store → interpret → admit → commit → graph update
  Use-side:    retrieve → coordinate → serialize → augment

Host MAS Interface:
  - on_task_start:    task identification + retrieval + coordination + serialization
  - on_agent_step:    optional per-step memory refresh
  - on_task_complete:  trajectory storage + card extraction + admission + commit + graph update
  - query:            manual retrieval interface for debugging
"""
import logging
import time
from typing import Optional

from utils.stats import stats

from .admission import AdmissionController
from .config import ConMemConfig
from .completion_detector import TaskCompletionDetector, CompletionSignal
from .coordinator import MemoryCoordinator
from .interpreter import TrajectoryInterpreter, mas_trajectory_to_dict
from .llm_backend import LLMClient, EmbeddingClient
from .memory_graph import MemoryGraphManager
from .retriever import MemoryRetriever
from .schema import MemoryCard
from .serializer import MemorySerializer
from .storage import ConMemStorage
from .task_identifier import TaskIdentifier

logger = logging.getLogger(__name__)


class ConMemModule:
    """
    Trajectory-to-Memory Conditioning Module (Section 4).

    A pluggable memory layer deployed on top of an existing host MAS.
    M = (T, M, G, Lambda, Phi)
    """

    def __init__(self, config: ConMemConfig, storage_dir: str, task_domain: str = None):
        self.config = config
        self.task_domain = task_domain
        self.config.apply_task_domain_profile(task_domain)
        self.config.sync_runtime_settings()
        self.storage = ConMemStorage(storage_dir)
        self._runtime_model_name = config.llm_model
        self._runtime_mas_architecture = "single"

        # LLM & Embedding backends
        self.llm = LLMClient(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            retry_count=config.llm_retry_count,
            timeout=config.llm_timeout,
            max_input_chars=config.llm_max_input_chars,
        )
        self.embedder = EmbeddingClient(
            api_key=config.embed_api_key,
            base_url=config.embed_base_url,
            model=config.embed_model,
            device=config.embed_device or None,
        )

        # Sub-components
        self.task_identifier = TaskIdentifier(config, self.llm, self.embedder, self.storage)
        self.interpreter = TrajectoryInterpreter(config, self.llm, task_domain=task_domain)
        self.graph_manager = MemoryGraphManager(config, self.llm, self.embedder, self.storage)
        self.retriever = MemoryRetriever(config, self.embedder, self.storage)
        self.coordinator = MemoryCoordinator(config, self.llm, self.embedder, self.storage)
        self.serializer = MemorySerializer(config)
        self.admission = AdmissionController(config, self.embedder, self.storage, self.llm)
        self.completion_detector = TaskCompletionDetector(config)

        # Pending trajectories for auto-completion (task_id -> list of step dicts)
        self._pending_trajectories: dict[str, list[dict]] = {}

        # Track card count from last retrieval
        self._last_card_count: int = 0
        self._last_retrieval_usage: dict = {}
        self._retrieval_usage_buffer: list[dict] = []
        self._last_update_usage: dict = {}

        # Counter for throttling post_commit_merge (see admission_post_commit_merge_every_k_tasks).
        self._completed_task_counter: int = 0

        # Startup consistency check (Section 4.8)
        self._startup_consistency_check()

    @classmethod
    def from_env(cls, storage_dir: str, env_path: str = None) -> "ConMemModule":
        """Create ConMemModule with configuration loaded from environment."""
        config = ConMemConfig.from_env(env_path)
        return cls(config, storage_dir)

    def set_runtime_context(
        self,
        model_name: Optional[str] = None,
        mas_architecture: Optional[str] = None,
    ):
        if model_name:
            self._runtime_model_name = model_name
        if mas_architecture:
            self._runtime_mas_architecture = mas_architecture

    def _enrich_trajectory_metadata(self, trajectory_data: dict) -> dict:
        enriched = dict(trajectory_data)
        enriched.setdefault("model_name", self._runtime_model_name)
        enriched.setdefault("mas_architecture", self._runtime_mas_architecture)
        if self.task_domain:
            enriched.setdefault("task_domain", self.task_domain)
        return enriched

    # ======================= Host MAS Interface (Section 4.6) =======================

    def on_task_start(
        self,
        task_description: str,
        agent_role: str = "default",
        task_id: Optional[str] = None,
        event_text: Optional[str] = None,
        interaction_context: str = "",
    ) -> str:
        """
        Called when a task starts. Returns serialized structured memory context.

        Pipeline: identify task → retrieve → coordinate → serialize.
        If no relevant memory, returns empty string.
        """
        t_start = time.perf_counter()
        current_round = self.storage.get_current_round()

        # Step 1: Task identification
        effective_event = event_text or task_description
        t_stage = time.perf_counter()
        tid, tdesc = self.task_identifier.identify_task(
            event_text=effective_event,
            explicit_task_id=task_id,
            explicit_task_desc=task_description,
            task_domain=self.task_domain,
        )
        identify_latency_ms = (time.perf_counter() - t_stage) * 1000

        # Ensure task is registered
        t_stage = time.perf_counter()
        self.task_identifier.register_task(
            tid, tdesc, current_round=current_round, task_domain=self.task_domain
        )
        register_latency_ms = (time.perf_counter() - t_stage) * 1000

        # Step 2: Retrieve related memory cards
        t_stage = time.perf_counter()
        activated_cards = self.retriever.retrieve(
            task_id=tid,
            task_description=tdesc,
            agent_role=agent_role,
            current_round=current_round,
            task_domain=self.task_domain,
            interaction_context=interaction_context,
        )
        retrieve_latency_ms = (time.perf_counter() - t_stage) * 1000

        if not activated_cards:
            self._last_card_count = 0
            self._record_retrieval_usage({
                "task_id": tid,
                "agent_role": agent_role,
                "task_domain": self.task_domain,
                "retrieval_calls": 1,
                "retrieved_cards": 0,
                "expanded_cards": 0,
                "coordinated_cards": 0,
                "injected_cards": 0,
                "injected_tokens_est": 0,
                "subgraph_edges": 0,
                "considered_cards_for_serialization": 0,
                "skipped_over_budget_cards": 0,
                "compacted_cards": 0,
                "token_budget": self.config.token_budget,
                "budget_utilization": 0.0,
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
                "identify_latency_ms": round(identify_latency_ms, 3),
                "register_latency_ms": round(register_latency_ms, 3),
                "retrieve_latency_ms": round(retrieve_latency_ms, 3),
                "expand_latency_ms": 0.0,
                "coordinate_latency_ms": 0.0,
                "serialize_latency_ms": 0.0,
            })
            return ""

        # Step 3: Graph-based subgraph walk (Section 4.4.2)
        t_stage = time.perf_counter()
        if self.config.enable_graph_expansion:
            expanded_cards, subgraph_edges = self.graph_manager.expand_activation(
                activated_cards,
                task_description=tdesc,
                max_expanded=self.config.graph_expansion_max_cards,
            )
            expanded_cards = self._cap_expanded_cards(expanded_cards, activated_cards)
        else:
            expanded_cards = activated_cards
            subgraph_edges = self.graph_manager.get_subgraph(
                {c.card_id for c in activated_cards}
            )
        expand_latency_ms = (time.perf_counter() - t_stage) * 1000

        # Step 4: Coordinate (can be disabled for ablation)
        t_stage = time.perf_counter()
        if self.config.enable_coordination:
            coordinated_cards = self.coordinator.coordinate(
                expanded_cards, subgraph_edges, task_description=tdesc
            )
        else:
            coordinated_cards = expanded_cards
        coordinate_latency_ms = (time.perf_counter() - t_stage) * 1000

        # Step 5: Serialize
        t_stage = time.perf_counter()
        memory_context = self.serializer.serialize(coordinated_cards, agent_role)
        serialize_latency_ms = (time.perf_counter() - t_stage) * 1000
        self._last_card_count = self.serializer.last_rendered_hint_count
        injected_tokens = self.serializer.last_serialized_token_count if memory_context else 0
        token_budget = max(int(self.config.token_budget or 0), 0)
        self._record_retrieval_usage({
            "task_id": tid,
            "agent_role": agent_role,
            "task_domain": self.task_domain,
            "retrieval_calls": 1,
            "retrieved_cards": len(activated_cards),
            "expanded_cards": len(expanded_cards),
            "coordinated_cards": len(coordinated_cards),
            "injected_cards": self.serializer.last_rendered_hint_count,
            "injected_tokens_est": injected_tokens,
            "subgraph_edges": len(subgraph_edges),
            "considered_cards_for_serialization": self.serializer.last_considered_card_count,
            "skipped_over_budget_cards": self.serializer.last_skipped_over_budget_count,
            "compacted_cards": self.serializer.last_compacted_card_count,
            "token_budget": self.config.token_budget,
            "budget_utilization": round((injected_tokens / token_budget), 6) if token_budget else 0.0,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
            "identify_latency_ms": round(identify_latency_ms, 3),
            "register_latency_ms": round(register_latency_ms, 3),
            "retrieve_latency_ms": round(retrieve_latency_ms, 3),
            "expand_latency_ms": round(expand_latency_ms, 3),
            "coordinate_latency_ms": round(coordinate_latency_ms, 3),
            "serialize_latency_ms": round(serialize_latency_ms, 3),
        })

        return memory_context

    def _record_retrieval_usage(self, record: dict):
        self._last_retrieval_usage = record
        self._retrieval_usage_buffer.append(dict(record))

    def get_and_reset_retrieval_usage(self) -> list[dict]:
        records = list(self._retrieval_usage_buffer)
        self._retrieval_usage_buffer = []
        return records

    def on_agent_step(
        self,
        task_description: str,
        agent_role: str = "default",
        task_id: Optional[str] = None,
        agent_message: str = "",
        observation: str = "",
    ) -> str:
        """
        Optional per-step memory refresh (Section 4.6 Agent Step).

        Also performs weak-signal task completion detection (Section 4.2.4 Step 4).
        If completion is detected and the host has not explicitly called
        on_task_complete(), triggers auto-completion with accumulated trajectory.

        Args:
            task_description: Current task description.
            agent_role: Role of the requesting agent.
            task_id: Task identifier.
            agent_message: The agent's output for this step (for completion detection).
            observation: Environment feedback for this step (for completion detection).

        Returns:
            Serialized memory context string.
        """
        # Accumulate step data for potential auto-completion
        if task_id:
            if task_id not in self._pending_trajectories:
                self._pending_trajectories[task_id] = []
            step_data = {
                "step_index": len(self._pending_trajectories[task_id]) + 1,
                "agent": agent_role,
                "input": task_description,
                "output": agent_message,
                "tool_calls": "",
                "feedback": observation,
            }
            self._pending_trajectories[task_id].append(step_data)

            # Weak-signal completion detection (Section 4.2.4 Step 4)
            signal = self.completion_detector.observe_step(
                task_id=task_id,
                agent_message=agent_message,
                observation=observation,
            )
            if signal.is_complete:
                self._auto_complete_task(
                    task_id=task_id,
                    task_description=task_description,
                    signal=signal,
                )

        return self.on_task_start(
            task_description=task_description,
            agent_role=agent_role,
            task_id=task_id,
            interaction_context="\n".join(part for part in (agent_message, observation) if part).strip(),
        )

    def _auto_complete_task(
        self,
        task_id: str,
        task_description: str,
        signal: CompletionSignal,
    ):
        """
        Auto-complete a task based on weak-signal detection (Section 4.2.4 Step 4).

        Constructs a trajectory from accumulated step data and triggers
        the full on_task_complete pipeline.
        """
        steps = self._pending_trajectories.pop(task_id, [])
        if not steps:
            return

        trajectory_data = {
            "task_description": task_description,
            "outcome": signal.detected_outcome or "partial",
            "steps": steps,
            "auto_detected": True,
            "detection_reason": signal.reason,
            "detection_confidence": signal.confidence,
        }

        logger.info(
            f"Auto-completing task {task_id} "
            f"(reason={signal.reason}, outcome={signal.detected_outcome}, "
            f"confidence={signal.confidence:.2f})"
        )

        self.on_task_complete(
            task_id=task_id,
            task_description=task_description,
            trajectory=trajectory_data,
            outcome=signal.detected_outcome or "partial",
        )
        self.completion_detector.reset_task(task_id)

    def on_task_complete(
        self,
        task_id: str,
        task_description: str,
        trajectory,
        outcome: str = "success",
    ):
        """
        Called when a task completes (Section 4.6 Task Complete).

        Pipeline:
          store trajectory → interpret → admission → commit → graph update.

        Args:
            task_id: Task identifier.
            task_description: Task description.
            trajectory: Either a utils.message.Trajectory object or a pre-converted dict.
            outcome: One of {success, partial, failure}.
        """
        t_update_start = time.perf_counter()
        api_before = stats.to_dict()

        def finish_update(
            status: str,
            candidate_cards: int = 0,
            admitted_cards: int = 0,
            committed_cards: int = 0,
        ):
            api_after = stats.to_dict()
            before_source = api_before.get("by_source", {}).get("conmem/llm", {})
            after_source = api_after.get("by_source", {}).get("conmem/llm", {})

            def delta(field: str):
                return after_source.get(field, 0) - before_source.get(field, 0)

            self._last_update_usage = {
                "task_id": task_id,
                "task_domain": self.task_domain,
                "outcome": outcome,
                "status": status,
                "candidate_cards": candidate_cards,
                "admitted_cards": admitted_cards,
                "committed_cards": committed_cards,
                "llm_calls": int(delta("calls")),
                "prompt_tokens": int(delta("prompt_tokens")),
                "completion_tokens": int(delta("completion_tokens")),
                "total_tokens": int(delta("prompt_tokens") + delta("completion_tokens")),
                "llm_time_seconds": round(float(delta("time")), 3),
                "latency_ms": round((time.perf_counter() - t_update_start) * 1000, 3),
            }

        current_round = self.storage.increment_round()

        # Clean up pending trajectory tracking (host explicitly notified)
        self._pending_trajectories.pop(task_id, None)
        self.completion_detector.reset_task(task_id)

        # Convert trajectory if it's a MAS Trajectory object
        if isinstance(trajectory, dict):
            traj_data = trajectory
        else:
            traj_data = mas_trajectory_to_dict(trajectory, outcome=outcome)
        traj_data = self._enrich_trajectory_metadata(traj_data)

        # Step 1: Store trajectory
        filepath = self.storage.store_trajectory(task_id, traj_data)
        logger.info(f"Stored trajectory for task {task_id} at {filepath}")

        # Update task record
        self.task_identifier.register_task(
            task_id,
            task_description,
            outcome=outcome,
            current_round=current_round,
            task_domain=self.task_domain,
        )
        task_record = self.storage.get_task(task_id)
        if task_record:
            task_record.trajectory_file = filepath
            task_record.outcome = outcome
            task_record.completion_round = current_round
            self.storage.insert_task(task_record)

        if self._should_skip_card_extraction(traj_data, outcome):
            logger.info(
                f"Skipping card extraction for task {task_id}: failed trajectory has no informative agent output."
            )
            finish_update("skipped_uninformative_failure")
            return
        if outcome == "failure" and not self.config.enable_failure_admission:
            logger.info("Skipping failure-card admission for task %s due to ablation flag.", task_id)
            finish_update("skipped_failure_admission_disabled")
            return

        # Step 2: Interpret trajectory → candidate cards (Aggregate(τ_new, C))
        existing_cards = self.storage.get_cards_by_task(
            task_id, active_only=True, task_domain=self.task_domain
        )
        candidates = self.interpreter.interpret(
            traj_data, task_id, current_round, existing_cards=existing_cards
        )

        if not candidates:
            logger.info(f"No cards extracted for task {task_id}. Trajectory stored only.")
            finish_update("no_candidate_cards")
            return

        # Step 3: Admission
        admitted = self.admission.admit_cards(candidates, current_round)

        if not admitted:
            logger.info(f"All candidates rejected for task {task_id}.")
            finish_update("all_candidates_rejected", candidate_cards=len(candidates))
            return

        # Step 4: Commit to storage
        self.storage.insert_cards(admitted)
        logger.info(f"Committed {len(admitted)} cards for task {task_id}")

        # Step 5: Graph update
        self.graph_manager.update_graph(admitted)

        # Step 6: post-commit merge / graph-size safeguard
        # Throttled: only run every K completed tasks to avoid per-task O(N²) scans.
        self._completed_task_counter += 1
        every_k = max(1, getattr(self.config, "admission_post_commit_merge_every_k_tasks", 1))
        if self._completed_task_counter % every_k == 0:
            self.admission.post_commit_merge(current_round)
        self.admission.check_graph_explosion(current_round)
        finish_update(
            "committed",
            candidate_cards=len(candidates),
            admitted_cards=len(admitted),
            committed_cards=len(admitted),
        )

    def _cap_expanded_cards(
        self,
        expanded_cards: list[MemoryCard],
        activated_cards: list[MemoryCard],
    ) -> list[MemoryCard]:
        limit = max(self.config.retrieval_top_k, self.config.graph_expansion_max_cards)
        if limit <= 0 or len(expanded_cards) <= limit:
            return expanded_cards

        activated_ids = {card.card_id for card in activated_cards}
        protected = [card for card in expanded_cards if card.card_id in activated_ids]
        neighbors = [card for card in expanded_cards if card.card_id not in activated_ids]
        neighbors.sort(key=lambda card: card.metadata.admission_score, reverse=True)
        return (protected + neighbors)[:limit]

    def _should_skip_card_extraction(self, traj_data: dict, outcome: str) -> bool:
        if not self.config.skip_uninformative_failed_trajectories or outcome != "failure":
            return False
        if traj_data.get("infrastructure_failure"):
            return True
        steps = traj_data.get("steps") or []
        if not steps:
            return True
        has_agent_output = any((step.get("output") or "").strip() for step in steps)
        return not has_agent_output

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        agent_role: str = "default",
    ) -> list[MemoryCard]:
        """
        Manual query interface (Section 4.6 Query).

        Allows direct query by text. Returns matching cards for debugging.
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k

        current_round = self.storage.get_current_round()

        # Use a temporary task_id for query
        tid, _ = self.task_identifier.identify_task(
            event_text=query_text,
            explicit_task_desc=query_text,
            task_domain=self.task_domain,
        )

        cards = self.retriever.retrieve(
            task_id=tid,
            task_description=query_text,
            agent_role=agent_role,
            current_round=current_round,
        )

        return cards[:top_k]

    # ======================= Storage Consistency (Section 4.8) =======================

    def _startup_consistency_check(self):
        """
        Startup consistency check (Section 4.8).

        If vector storage (embeddings) and relational storage (cards/tasks) are
        inconsistent, the system uses relational storage as the source of truth
        and rebuilds missing embeddings.

        Also cleans up orphaned edges referencing deleted cards.
        """
        # 1. Rebuild missing card embeddings
        cards_missing = self.storage.get_cards_missing_embeddings()
        if cards_missing:
            logger.info(
                f"Consistency check: {len(cards_missing)} active cards missing embeddings. "
                "Rebuilding..."
            )
            rebuilt = 0
            for card in cards_missing:
                try:
                    card.embedding = self.embedder.embed(card.activation_text())
                    self.storage.update_card(card)
                    rebuilt += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to rebuild embedding for card {card.card_id}: {e}"
                    )
            logger.info(f"Rebuilt {rebuilt}/{len(cards_missing)} card embeddings")

        # 2. Rebuild missing task embeddings
        tasks_missing = self.storage.get_tasks_missing_embeddings()
        if tasks_missing:
            logger.info(
                f"Consistency check: {len(tasks_missing)} tasks missing embeddings. "
                "Rebuilding..."
            )
            rebuilt = 0
            for task in tasks_missing:
                try:
                    task.embedding = self.embedder.embed(task.task_description)
                    self.storage.insert_task(task)
                    rebuilt += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to rebuild embedding for task {task.task_id}: {e}"
                    )
            logger.info(f"Rebuilt {rebuilt}/{len(tasks_missing)} task embeddings")

        # 3. Clean up orphaned edges
        deleted = self.storage.delete_orphaned_edges()
        if deleted:
            logger.info(f"Consistency check: removed {deleted} orphaned edges")

    # ======================= Augmentation (Section 4.3.4) =======================

    def augment(self, original_context: str, memory_context: str) -> str:
        """
        Augment(x_t^(i), tilde_m_t^(i)) → hat_x_t^(i).

        Default direct-call strategy: prefix memory text to the caller-provided context. The centralized-memory adapter injects memory through the host MAS prompt fields instead.
        """
        if not memory_context:
            return original_context
        return memory_context + "\n\n" + original_context
