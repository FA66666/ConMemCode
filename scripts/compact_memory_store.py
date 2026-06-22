#!/usr/bin/env python3
"""Offline compaction for an existing ConMem SQLite store.

Runs the post-commit merge pass repeatedly until the active-card count stops
shrinking. This is useful after enabling cross-task aggregation logic on an
already-populated memory store.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_core.memory.backbone.conmem.admission import AdmissionController
from mas_core.memory.backbone.conmem.config import ConMemConfig
from mas_core.memory.backbone.conmem.storage import ConMemStorage


class _NoopEmbedder:
    def embed(self, text):
        raise RuntimeError("Embedding should not be used during offline compaction.")


def main():
    parser = argparse.ArgumentParser(description="Compact an existing ConMem memory store.")
    parser.add_argument("--storage_dir", type=str, required=True, help="Directory containing conmem.db")
    parser.add_argument("--env_path", type=str, default=None, help="Optional .env path for config loading")
    parser.add_argument("--max_passes", type=int, default=5, help="Maximum merge passes (default: 5)")
    args = parser.parse_args()

    config = ConMemConfig.from_env(args.env_path if args.env_path and os.path.exists(args.env_path) else None)
    storage = ConMemStorage(args.storage_dir)
    controller = AdmissionController(config, _NoopEmbedder(), storage, llm=None)

    current_round = storage.get_current_round()
    before = storage.count_active_cards()
    print(f"Active cards before compaction: {before}")

    previous = before
    for idx in range(1, max(args.max_passes, 1) + 1):
        controller.post_commit_merge(current_round)
        after = storage.count_active_cards()
        print(f"Pass {idx}: {previous} -> {after}")
        if after >= previous:
            break
        previous = after

    print(f"Active cards after compaction: {storage.count_active_cards()}")


if __name__ == "__main__":
    main()
