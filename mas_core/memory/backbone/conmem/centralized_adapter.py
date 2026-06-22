"""
ConMem Centralized Memory Adapter.

Bridges ConMemModule into the MAS framework through BaseCentralizedMemory.
"""
import logging
import re
from typing import Optional

from common.registry import registry
from mas_core.base_centralized_memory import BaseCentralizedMemory, Memory

from .conmem_module import ConMemModule
from .config import ConMemConfig
from .interpreter import mas_trajectory_to_dict
from .storage import resolve_conmem_storage_dir
from common.utils.factual_qa import is_factual_qa_domain

logger = logging.getLogger(__name__)


@registry.register_memory("conmem")
class ConMemCentralizedMemory(BaseCentralizedMemory):
    """
    Adapts ConMemModule to the BaseCentralizedMemory interface
    used by MAS structures (AutoGen, Camel, MacNet).

    Usage:
        config = ConMemConfig.from_env()
        memory = ConMemCentralizedMemory(config, storage_dir="/path/to/storage")

        # Or via from_config:
        memory = ConMemCentralizedMemory.from_config(config_dict, working_dir)
    """

    def __init__(self, config: ConMemConfig, storage_dir: str, task_domain: str = None):
        super().__init__()
        self.conmem = ConMemModule(config, storage_dir, task_domain=task_domain)
        self.conmem.set_runtime_context(model_name=config.llm_model, mas_architecture="unknown")
        self._agent_roles: dict[str, str] = {}  # agent_uuid -> role
        self._task_retrieval_card_count: int = 0  # accumulated card count per task
        self.disable_retrieval: bool = False

    def register_agents(self, agents_list: list):
        """Register agents and map their UUIDs to roles."""
        super().register_agents(agents_list)
        for agent in agents_list:
            if hasattr(agent, "id") and hasattr(agent, "role"):
                agent_id = agent.id or str(id(agent))
                self._agent_roles[agent_id] = self._infer_agent_role(agent)

    def _infer_agent_role(self, agent) -> str:
        role = (getattr(agent, "role", None) or "default").lower()
        prompt_text = " ".join([
            str(getattr(agent, "system_prompt_template", "") or ""),
            str(getattr(agent, "user_prompt_template", "") or ""),
        ]).lower()

        def has_term(text: str, terms: tuple[str, ...]) -> bool:
            return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)

        if has_term(role, ("critic", "reviewer", "judge", "evaluator", "summarizer")):
            return "evaluator"
        if has_term(role, ("actor", "executor", "coder", "developer")):
            return "executor"
        if has_term(role, ("assistant", "planner")):
            return "planner"

        if has_term(role, ("proxy",)):
            if any(k in prompt_text for k in ("write correct python", "implement the task", "complete python function", "return only one", "code from")):
                return "executor"
            if has_term(prompt_text, ("validator", "verify")):
                return "evaluator"
            if has_term(prompt_text, ("plan", "strategy", "requirement")):
                return "planner"
            return "default"

        if has_term(prompt_text, ("critic", "reviewer", "judge", "evaluate", "verify", "summarizer", "finalize")):
            return "evaluator"
        if any(k in prompt_text for k in ("write correct python", "implement the task", "complete python function", "return only one", "code from")):
            return "executor"
        if has_term(prompt_text, ("actor", "executor", "coder", "developer", "implement")):
            return "executor"
        if has_term(prompt_text, ("plan", "strategy", "assistant", "proxy", "requirement")):
            return "planner"
        return "default"

    def add_memory(self, trajectory=None, task_id: str = "", task_description: str = "",
                   outcome: str = "success", **kwargs):
        """
        Add memory from a completed task trajectory.

        Accepts either a MAS Trajectory object or a pre-converted dict.
        """
        if trajectory is None:
            return

        if not task_description and hasattr(trajectory, "task_init_description"):
            task_description = trajectory.task_init_description or ""

        self.conmem.on_task_complete(
            task_id=task_id,
            task_description=task_description,
            trajectory=trajectory,
            outcome=outcome,
        )

    def retrieve_memory(self, task_description: str, agent=None) -> Memory:
        """
        Retrieve memory for a specific agent given a task description.

        Returns Memory with text_memory set to the serialized ConMem context.
        """
        if self.disable_retrieval:
            self.conmem._last_card_count = 0
            return Memory(text_memory=None)

        agent_role = "default"
        if agent is not None:
            agent_id = getattr(agent, "id", None) or str(id(agent))
            agent_role = self._agent_roles.get(agent_id) or self._infer_agent_role(agent)

        if (
            self.conmem.config.disable_factual_qa_evaluator_memory
            and agent_role == "evaluator"
            and is_factual_qa_domain(getattr(self.conmem, "task_domain", None))
        ):
            return Memory(text_memory=None)

        memory_context = self.conmem.on_task_start(
            task_description=task_description,
            agent_role=agent_role,
        )

        # Accumulate card count from this retrieval
        self._task_retrieval_card_count += self.conmem._last_card_count

        return Memory(
            text_memory=memory_context if memory_context else None,
        )

    def get_and_reset_retrieval_count(self) -> int:
        """Return accumulated card count for current task and reset counter."""
        count = self._task_retrieval_card_count
        self._task_retrieval_card_count = 0
        return count

    def get_and_reset_retrieval_usage(self) -> list[dict]:
        """Return detailed ConMem retrieval/injection records for current task."""
        return self.conmem.get_and_reset_retrieval_usage()

    def set_runtime_context(
        self,
        model_name: Optional[str] = None,
        mas_architecture: Optional[str] = None,
    ):
        self.conmem.set_runtime_context(
            model_name=model_name,
            mas_architecture=mas_architecture,
        )

    def process_memory(self, text_memory: str, task_description: str = "",
                       extra_fields: dict = None, agent=None, **kwargs) -> Memory:
        """
        Process text memory into Memory object.

        For ConMem, text_memory is already the serialized context from retrieve_memory.
        No additional latent processing is needed.
        """
        return Memory(
            text_memory=text_memory,
            extra_fields=extra_fields or {},
        )

    @classmethod
    def from_config(cls, config: dict, working_dir: str) -> "ConMemCentralizedMemory":
        """
        Create instance from configuration dictionary.

        Expected config keys:
            - storage_dir (str): legacy directory for ConMem storage
            - shared_storage_dir (str): shared directory for cross-task ConMem storage
            - memory_storage_dir (str): alias of shared_storage_dir
            - env_path (str, optional): path to .env file for LLM/embedding API keys
            - Any ConMemConfig field overrides (e.g., admission_threshold, retrieval_top_k)
        """
        storage_dir = resolve_conmem_storage_dir(
            shared_storage_dir=config.get("shared_storage_dir") or config.get("memory_storage_dir"),
            fallback_storage_dir=config.get("storage_dir", f"{working_dir}/conmem"),
        )
        env_path = config.get("env_path", None)

        conmem_config = ConMemConfig.from_env(env_path)

        # Apply any config overrides
        config_fields = {f.name for f in conmem_config.__dataclass_fields__.values()}
        for key, value in config.items():
            if key in config_fields:
                setattr(conmem_config, key, value)

        return cls(conmem_config, storage_dir)


@registry.register_memory("none")
class NullCentralizedMemory(BaseCentralizedMemory):
    """No-op centralized memory used for no-memory baselines and config-driven MAS runs."""

    def __init__(self):
        super().__init__()

    def add_memory(self, **kwargs):
        return None

    def retrieve_memory(self, task_description: str, agent=None) -> Memory:
        return Memory(text_memory=None)

    def process_memory(self, text_memory: str, task_description: str = "", extra_fields: dict = None, agent=None, **kwargs) -> Memory:
        return Memory(text_memory=text_memory, extra_fields=extra_fields or {})

    @classmethod
    def from_config(cls, config: dict, working_dir: str) -> "NullCentralizedMemory":
        return cls()

    def get_and_reset_retrieval_count(self) -> int:
        return 0

    def get_and_reset_retrieval_usage(self) -> list[dict]:
        return []

    def set_runtime_context(
        self,
        model_name: Optional[str] = None,
        mas_architecture: Optional[str] = None,
    ):
        return None
