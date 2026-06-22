"""
ConMem Task Completion Detection (Section 4.2.4 Step 4).

Weak-signal detection for when the host MAS does not explicitly notify
task completion. Three detection heuristics:

1. Termination markers: patterns in agent messages indicating completion
2. Environment success status: observable success signals in observations
3. Max round threshold: forced completion after exceeding a round limit
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .config import ConMemConfig

logger = logging.getLogger(__name__)


@dataclass
class CompletionSignal:
    """Result of completion detection."""
    is_complete: bool = False
    reason: str = ""            # "termination_marker" | "env_success" | "max_rounds" | ""
    detected_outcome: str = ""  # "success" | "partial" | "failure" | ""
    confidence: float = 0.0     # [0.0, 1.0]


class TaskCompletionDetector:
    """
    Detects task completion from weak signals when the host MAS
    does not explicitly call on_task_complete().
    """

    # Default termination markers — patterns that indicate an agent considers the task done
    TERMINATION_PATTERNS = [
        re.compile(r"\b(?:task\s+(?:is\s+)?(?:complete|done|finished|accomplished))\b", re.IGNORECASE),
        re.compile(r"\b(?:final\s+answer|solution\s+(?:is|:))\b", re.IGNORECASE),
        re.compile(r"\b(?:TERMINATE|FINISH|DONE)\b"),
        re.compile(r"\b(?:successfully\s+(?:completed|solved|resolved))\b", re.IGNORECASE),
        re.compile(r"\b(?:no\s+(?:further|more)\s+(?:action|steps?)\s+(?:needed|required))\b", re.IGNORECASE),
    ]

    # Patterns indicating success in environment observations
    ENV_SUCCESS_PATTERNS = [
        re.compile(r"\b(?:all\s+tests?\s+pass(?:ed)?)\b", re.IGNORECASE),
        re.compile(r"\b(?:build\s+succeed(?:ed)?|compilation\s+success)\b", re.IGNORECASE),
        re.compile(r"\b(?:correct|accepted|approved)\b", re.IGNORECASE),
        re.compile(r"\b(?:exit\s+code\s*[:=]\s*0)\b", re.IGNORECASE),
        re.compile(r"\bSUCCESS\b"),
    ]

    # Patterns indicating failure
    ENV_FAILURE_PATTERNS = [
        re.compile(r"\b(?:all\s+tests?\s+fail(?:ed)?)\b", re.IGNORECASE),
        re.compile(r"\b(?:fatal\s+error|unrecoverable)\b", re.IGNORECASE),
        re.compile(r"\b(?:rejected|denied)\b", re.IGNORECASE),
        re.compile(r"\bFAILED\b"),
    ]

    def __init__(self, config: ConMemConfig):
        self.config = config
        # Track per-task state
        self._task_rounds: dict[str, int] = {}
        self._max_rounds: int = getattr(config, "completion_max_rounds", 30)

    def observe_step(
        self,
        task_id: str,
        agent_message: str = "",
        observation: str = "",
    ) -> CompletionSignal:
        """
        Observe a single agent step and detect completion signals.

        Args:
            task_id: The task being tracked.
            agent_message: The agent's output/response for this step.
            observation: Environment feedback/observation for this step.

        Returns:
            CompletionSignal with detection results.
        """
        # Track round count
        self._task_rounds[task_id] = self._task_rounds.get(task_id, 0) + 1
        current_rounds = self._task_rounds[task_id]

        # Check 1: Termination markers in agent message
        if agent_message:
            for pattern in self.TERMINATION_PATTERNS:
                if pattern.search(agent_message):
                    # Determine outcome based on message content
                    outcome = self._infer_outcome_from_text(agent_message)
                    logger.info(
                        f"Completion detected for task {task_id}: "
                        f"termination marker (round {current_rounds})"
                    )
                    return CompletionSignal(
                        is_complete=True,
                        reason="termination_marker",
                        detected_outcome=outcome,
                        confidence=0.8,
                    )

        # Check 2: Environment success/failure signals
        if observation:
            for pattern in self.ENV_SUCCESS_PATTERNS:
                if pattern.search(observation):
                    logger.info(
                        f"Completion detected for task {task_id}: "
                        f"environment success signal (round {current_rounds})"
                    )
                    return CompletionSignal(
                        is_complete=True,
                        reason="env_success",
                        detected_outcome="success",
                        confidence=0.9,
                    )

            for pattern in self.ENV_FAILURE_PATTERNS:
                if pattern.search(observation):
                    logger.info(
                        f"Completion detected for task {task_id}: "
                        f"environment failure signal (round {current_rounds})"
                    )
                    return CompletionSignal(
                        is_complete=True,
                        reason="env_success",
                        detected_outcome="failure",
                        confidence=0.85,
                    )

        # Check 3: Max round threshold
        if current_rounds >= self._max_rounds:
            logger.warning(
                f"Completion forced for task {task_id}: "
                f"max rounds ({self._max_rounds}) exceeded"
            )
            return CompletionSignal(
                is_complete=True,
                reason="max_rounds",
                detected_outcome="partial",
                confidence=0.6,
            )

        return CompletionSignal(is_complete=False)

    def _infer_outcome_from_text(self, text: str) -> str:
        """Infer task outcome from the completion message text."""
        text_lower = text.lower()

        success_signals = ["successfully", "correct", "solved", "accomplished", "done"]
        failure_signals = ["failed", "unable", "cannot", "error", "incorrect"]

        success_count = sum(1 for s in success_signals if s in text_lower)
        failure_count = sum(1 for s in failure_signals if s in text_lower)

        if success_count > failure_count:
            return "success"
        elif failure_count > success_count:
            return "failure"
        return "partial"

    def reset_task(self, task_id: str):
        """Reset tracking state for a task."""
        self._task_rounds.pop(task_id, None)

    def get_round_count(self, task_id: str) -> int:
        """Get the current round count for a task."""
        return self._task_rounds.get(task_id, 0)
