"""
ConMem — Trajectory-to-Memory Conditioning Module.

A pluggable memory layer for multi-agent systems that converts
historical task trajectories into structured, retrievable memory cards.
"""
from .conmem_module import ConMemModule
from .config import ConMemConfig
from .centralized_adapter import ConMemCentralizedMemory, NullCentralizedMemory
from .schema import (
    MemoryCard,
    MemoryEdge,
    TaskRecord,
    MemoryType,
    LifecycleState,
    TaskOutcome,
    EdgeRelation,
    Provenance,
    CardMetadata,
)

__all__ = [
    "ConMemModule",
    "ConMemConfig",
    "ConMemCentralizedMemory",
    "NullCentralizedMemory",
    "MemoryCard",
    "MemoryEdge",
    "TaskRecord",
    "MemoryType",
    "LifecycleState",
    "TaskOutcome",
    "EdgeRelation",
    "Provenance",
    "CardMetadata",
]
