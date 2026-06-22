"""Shared utility functions for ConMem components."""
import re


def normalize_role(role: str) -> str:
    """Map an agent role string to one of the canonical role keys."""
    role = role.lower()
    if any(k in role for k in ("plan", "proxy", "strategist", "requirement")):
        return "planner"
    if any(k in role for k in ("actor", "exec", "coder", "developer", "tool", "implement")):
        return "executor"
    if any(k in role for k in ("critic", "eval", "judge", "review", "summar")):
        return "evaluator"
    return "default"


def token_overlap(a: str, b: str) -> float:
    """Compute Jaccard overlap of 4+-letter word tokens between two strings."""
    ta = set(re.findall(r"\b[a-zA-Z_]{4,}\b", a.lower()))
    tb = set(re.findall(r"\b[a-zA-Z_]{4,}\b", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
