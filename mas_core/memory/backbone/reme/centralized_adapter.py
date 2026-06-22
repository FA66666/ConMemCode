"""ReMe Centralized Memory Adapter.

This adapter keeps the host MAS side dependency-light: the evaluation process
talks to a separately running ReMe HTTP service and does not import ReMe itself.
"""

from __future__ import annotations

import logging
import copy
import json
import os
import re
import time
from typing import Any, Optional

import requests

from common.registry import registry
from mas_core.base_centralized_memory import BaseCentralizedMemory, Memory

logger = logging.getLogger(__name__)


def _clip_text(text: Any, max_chars: int = 12000) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n...[truncated]"


def _feedback_to_text(feedback: Any) -> str:
    if isinstance(feedback, dict):
        parts = []
        for key in ("summary", "score", "test_passed", "answer_correct", "extracted_answer", "full_feedback", "observation"):
            if key in feedback and feedback[key] not in (None, ""):
                parts.append(f"{key}: {feedback[key]}")
        return "\n".join(parts)
    return str(feedback or "")


def _merge_numeric_stats(target: dict, source: dict) -> dict:
    for key, value in (source or {}).items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
            if isinstance(value, float):
                target[key] = round(target[key], 4)
        elif isinstance(value, dict):
            child = target.setdefault(key, {})
            if isinstance(child, dict):
                _merge_numeric_stats(child, value)
    return target


def _safe_filename_part(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:160] or fallback


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


@registry.register_memory("reme")
class ReMeCentralizedMemory(BaseCentralizedMemory):
    """Bridge ReMe task memory HTTP APIs into the MAS memory interface."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8003/",
        workspace_id: str = "conmem_reme",
        task_domain: Optional[str] = None,
        top_k: int = 5,
        timeout: float = 120.0,
        read_only: bool = False,
        trajectory_dir: Optional[str] = None,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.workspace_id = workspace_id
        self.task_domain = task_domain or "unknown"
        self.top_k = top_k
        self.timeout = timeout
        self.read_only = read_only
        self.trajectory_dir = trajectory_dir or os.getenv("REME_TRAJECTORY_DIR", "")
        self._agent_roles: dict[str, str] = {}
        self._task_retrieval_count = 0
        self._last_count_active_cards = 0
        self._usage_stats: dict = {}
        self._last_service_usage_total: dict = {}
        self._saved_trajectory_files: list[str] = []

        # Compatibility with existing evaluation logging that expects
        # `conmem.storage.count_active_cards()`.
        self.storage = self

    def register_agents(self, agents_list: list):
        super().register_agents(agents_list)
        for agent in agents_list:
            agent_id = getattr(agent, "id", None) or str(id(agent))
            self._agent_roles[agent_id] = getattr(agent, "role", None) or "default"

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _post(self, path: str, payload: dict) -> dict:
        response = requests.post(self._endpoint(path), json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        self._record_usage_from_response(data)
        return data

    def _record_usage_from_response(self, data: dict):
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        if not isinstance(metadata, dict):
            return
        usage_delta = metadata.get("reme_usage_delta")
        if isinstance(usage_delta, dict):
            _merge_numeric_stats(self._usage_stats, usage_delta)
        usage_total = metadata.get("reme_usage_total")
        if isinstance(usage_total, dict):
            self._last_service_usage_total = usage_total

    def _agent_role(self, agent=None) -> str:
        if agent is None:
            return "default"
        agent_id = getattr(agent, "id", None) or str(id(agent))
        return self._agent_roles.get(agent_id) or getattr(agent, "role", None) or "default"

    def retrieve_memory(self, task_description: str, agent=None) -> Memory:
        if self.count_active_cards() <= 0:
            return Memory(text_memory=None, extra_fields={"retrieved_count": 0})

        agent_role = self._agent_role(agent)
        query = (
            f"Task domain: {self.task_domain}\n"
            f"Agent role: {agent_role}\n"
            f"Task:\n{task_description or ''}"
        )
        payload = {
            "workspace_id": self.workspace_id,
            "query": query,
            "top_k": self.top_k,
        }
        try:
            data = self._post("retrieve_task_memory", payload)
        except Exception as exc:
            logger.warning("ReMe retrieve_memory failed: %s", exc)
            return Memory(text_memory=None)

        memory_list = data.get("metadata", {}).get("memory_list", [])
        retrieved_count = len(memory_list) if isinstance(memory_list, list) else 0
        self._task_retrieval_count += retrieved_count
        text = (data.get("answer") or "").strip()
        return Memory(text_memory=text or None, extra_fields={"retrieved_count": retrieved_count})

    def add_memory(
        self,
        trajectory=None,
        task_id: str = "",
        task_description: str = "",
        outcome: str = "success",
        **kwargs,
    ):
        if self.read_only or trajectory is None:
            return None

        messages = self._trajectory_to_messages(trajectory, task_description)
        if not messages:
            return None

        score = {"success": 1.0, "partial": 0.5, "failure": 0.0}.get(str(outcome).lower(), 0.0)
        payload = {
            "workspace_id": self.workspace_id,
            "trajectories": [
                {
                    "messages": messages,
                    "score": score,
                    "metadata": {
                        "query": task_description,
                        "task_id": task_id,
                        "task_domain": self.task_domain,
                        "outcome": outcome,
                    },
                }
            ],
        }
        self._save_trajectory_record(
            task_id=task_id,
            task_description=task_description,
            outcome=outcome,
            score=score,
            messages=messages,
            source_trajectory=trajectory,
            payload=payload,
        )
        try:
            data = self._post("summary_task_memory", payload)
            inserted = data.get("metadata", {}).get("update_result", {}).get("inserted_count")
            if isinstance(inserted, int):
                self._last_count_active_cards += inserted
            return data
        except Exception as exc:
            logger.warning("ReMe add_memory failed: %s", exc)
            return None

    def _save_trajectory_record(
        self,
        *,
        task_id: str,
        task_description: str,
        outcome: str,
        score: float,
        messages: list[dict],
        source_trajectory: Any,
        payload: dict,
    ) -> Optional[str]:
        if not self.trajectory_dir:
            return None

        domain = _safe_filename_part(self.task_domain, "unknown_domain")
        safe_task_id = _safe_filename_part(task_id, "unknown_task")
        out_dir = os.path.join(self.trajectory_dir, domain)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{safe_task_id}.json")
        record = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workspace_id": self.workspace_id,
            "task_domain": self.task_domain,
            "task_id": task_id,
            "task_description": task_description,
            "outcome": outcome,
            "score": score,
            "message_count": len(messages),
            "messages": messages,
            "reme_payload": payload,
            "source_trajectory": _jsonable(source_trajectory),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            self._saved_trajectory_files.append(path)
            return path
        except Exception as exc:
            logger.warning("Failed to save ReMe trajectory record %s: %s", path, exc)
            return None

    def _trajectory_to_messages(self, trajectory: Any, task_description: str) -> list[dict]:
        if isinstance(trajectory, dict):
            task_description = task_description or trajectory.get("task_description", "")
            steps = trajectory.get("steps") or []
        else:
            steps = []
            for idx, msg_graph in enumerate(getattr(trajectory, "trajectory", []) or [], 1):
                steps.append(
                    {
                        "step_index": idx,
                        "agent": "mas",
                        "input": getattr(trajectory, "task_init_description", "") or task_description,
                        "output": getattr(msg_graph, "action", "") or "",
                        "feedback": getattr(msg_graph, "observation", "") or "",
                    }
                )

        messages: list[dict] = [
            {
                "role": "user",
                "content": _clip_text(f"Task:\n{task_description}", 4000),
            }
        ]
        for step in steps:
            agent = step.get("agent", "agent") if isinstance(step, dict) else "agent"
            step_input = step.get("input", "") if isinstance(step, dict) else ""
            step_output = step.get("output", "") if isinstance(step, dict) else ""
            feedback = _feedback_to_text(step.get("feedback", "") if isinstance(step, dict) else "")

            user_content = (
                f"Agent: {agent}\n"
                f"Input:\n{_clip_text(step_input, 6000)}"
            )
            assistant_content = (
                f"Output:\n{_clip_text(step_output, 8000)}"
                + (f"\n\nFeedback:\n{_clip_text(feedback, 4000)}" if feedback else "")
            )
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})

        return messages

    def get_and_reset_retrieval_count(self) -> int:
        count = self._task_retrieval_count
        self._task_retrieval_count = 0
        return count

    def get_usage_stats(self) -> dict:
        return copy.deepcopy(self._usage_stats)

    def get_last_service_usage_total(self) -> dict:
        return copy.deepcopy(self._last_service_usage_total)

    def get_saved_trajectory_files(self) -> list[str]:
        return list(self._saved_trajectory_files)

    def fetch_service_usage_stats(self, reset: bool = False) -> dict:
        try:
            if reset:
                response = requests.post(
                    self._endpoint("reme_usage_stats"),
                    json={"action": "reset"},
                    timeout=self.timeout,
                )
            else:
                response = requests.get(self._endpoint("reme_usage_stats"), timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                self._last_service_usage_total = data
            return data
        except Exception as exc:
            logger.warning("ReMe fetch_service_usage_stats failed: %s", exc)
            return {}

    def count_active_cards(self) -> int:
        payload = {"workspace_id": self.workspace_id, "action": "list"}
        try:
            data = self._post("vector_store", payload)
            action_result = data.get("metadata", {}).get("action_result", [])
            if isinstance(action_result, list):
                self._last_count_active_cards = len(action_result)
        except Exception:
            pass
        return self._last_count_active_cards

    def get_current_round(self) -> int:
        return 0

    def set_runtime_context(
        self,
        model_name: Optional[str] = None,
        mas_architecture: Optional[str] = None,
    ):
        return None

    def process_memory(
        self,
        text_memory: str,
        task_description: str = "",
        extra_fields: dict = None,
        agent=None,
        **kwargs,
    ) -> Memory:
        return Memory(text_memory=text_memory, extra_fields=extra_fields or {})

    @classmethod
    def from_config(cls, config: dict, working_dir: str) -> "ReMeCentralizedMemory":
        return cls(
            base_url=config.get("base_url", config.get("reme_url", "http://127.0.0.1:8003/")),
            workspace_id=config.get("workspace_id", config.get("reme_workspace", "conmem_reme")),
            task_domain=config.get("task_domain"),
            top_k=int(config.get("top_k", config.get("reme_top_k", 5))),
            timeout=float(config.get("timeout", config.get("reme_timeout", 120.0))),
            read_only=bool(config.get("read_only", False)),
            trajectory_dir=config.get("trajectory_dir", config.get("reme_trajectory_dir")),
        )
