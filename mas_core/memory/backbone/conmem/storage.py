"""
ConMem storage layer.

The storage keeps legacy columns for compatibility and stores the unified card
payload in an additional JSON column. Existing databases are migrated lazily by
adding the missing column.
"""
import json
import logging
import os
import re
import sqlite3
import threading
from typing import Optional

from .schema import (
    MemoryCard,
    MemoryEdge,
    TaskRecord,
    _serialize_embedding_blob,
    _deserialize_embedding_blob,
)

logger = logging.getLogger(__name__)

DEFAULT_SHARED_STORAGE_DIRNAME = "conmem_shared_storage"


def default_shared_storage_dir(project_root: Optional[str] = None) -> str:
    """Return the default shared ConMem storage directory."""
    base_dir = project_root or os.getcwd()
    return os.path.abspath(os.path.join(base_dir, DEFAULT_SHARED_STORAGE_DIRNAME))


def resolve_conmem_storage_dir(
    *,
    shared_storage_dir: Optional[str] = None,
    project_root: Optional[str] = None,
    fallback_storage_dir: Optional[str] = None,
) -> str:
    """
    Resolve the physical directory used by ConMem storage.

    Priority:
      1. Explicit shared storage directory.
      2. Legacy fallback storage directory.
      3. Project-level shared storage directory.
    """
    if shared_storage_dir:
        return os.path.abspath(shared_storage_dir)
    if fallback_storage_dir:
        return os.path.abspath(fallback_storage_dir)
    return default_shared_storage_dir(project_root)


class ConMemStorage:
    """Unified storage layer for the ConMem module."""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.db_path = os.path.join(storage_dir, "conmem.db")
        self.trajectory_dir = os.path.join(storage_dir, "trajectories")
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(self.trajectory_dir, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_cards (
                card_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                task_domain TEXT,
                task_description TEXT,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                evidence TEXT,
                provenance TEXT,
                metadata TEXT,
                embedding TEXT,
                card_payload TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_edges (
                source_card_id TEXT NOT NULL,
                target_card_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 0.0,
                rationale TEXT,
                PRIMARY KEY (source_card_id, target_card_id)
            );
            CREATE TABLE IF NOT EXISTS task_records (
                task_id TEXT PRIMARY KEY,
                task_domain TEXT,
                task_description TEXT,
                outcome TEXT,
                completion_round INTEGER DEFAULT 0,
                completion_timestamp REAL DEFAULT 0.0,
                trajectory_file TEXT,
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cards_task ON memory_cards(task_id);
            CREATE INDEX IF NOT EXISTS idx_cards_domain ON memory_cards(task_domain);
            CREATE INDEX IF NOT EXISTS idx_cards_type ON memory_cards(memory_type);
            CREATE INDEX IF NOT EXISTS idx_tasks_domain ON task_records(task_domain);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_card_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_card_id);
            """
        )
        self._ensure_column("memory_cards", "task_domain", "TEXT")
        self._ensure_column("memory_cards", "card_payload", "TEXT")
        self._ensure_column("task_records", "task_domain", "TEXT")
        conn.execute(
            "INSERT OR IGNORE INTO system_state (key, value) VALUES (?, ?)",
            ("current_round", "0"),
        )
        conn.commit()
        self._migrate_embeddings_to_blob()

    def _migrate_embeddings_to_blob(self):
        """One-shot migration: convert legacy JSON-TEXT embeddings into BLOB bytes.

        Idempotent — guarded by a row in `system_state`. New inserts always write
        BLOBs, so after the migration runs once this function is a no-op.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = 'embedding_blob_migration'"
        ).fetchone()
        if row and row["value"] == "done":
            return

        with self._write_lock:
            converted_cards = 0
            # Scan card embeddings; any non-NULL str value needs migration.
            card_rows = conn.execute(
                "SELECT card_id, embedding FROM memory_cards WHERE embedding IS NOT NULL"
            ).fetchall()
            for r in card_rows:
                raw = r["embedding"]
                if isinstance(raw, (bytes, bytearray)):
                    continue
                blob = _serialize_embedding_blob(_deserialize_embedding_blob(raw))
                conn.execute(
                    "UPDATE memory_cards SET embedding = ? WHERE card_id = ?",
                    (blob, r["card_id"]),
                )
                converted_cards += 1

            converted_tasks = 0
            task_rows = conn.execute(
                "SELECT task_id, embedding FROM task_records WHERE embedding IS NOT NULL"
            ).fetchall()
            for r in task_rows:
                raw = r["embedding"]
                if isinstance(raw, (bytes, bytearray)):
                    continue
                blob = _serialize_embedding_blob(_deserialize_embedding_blob(raw))
                conn.execute(
                    "UPDATE task_records SET embedding = ? WHERE task_id = ?",
                    (blob, r["task_id"]),
                )
                converted_tasks += 1

            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                ("embedding_blob_migration", "done"),
            )
            conn.commit()
            if converted_cards or converted_tasks:
                logger.info(
                    "Migrated embeddings to BLOB: %d cards, %d tasks",
                    converted_cards,
                    converted_tasks,
                )

    def _ensure_column(self, table: str, column: str, definition: str):
        conn = self._get_conn()
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        columns = {row[1] for row in rows}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    # ======================= Round Counter =======================

    def get_current_round(self) -> int:
        row = self._get_conn().execute(
            "SELECT value FROM system_state WHERE key = 'current_round'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def increment_round(self) -> int:
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE system_state SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
                "WHERE key = 'current_round'"
            )
            conn.commit()
            return self.get_current_round()

    # ======================= Card CRUD =======================

    def insert_card(self, card: MemoryCard):
        self.insert_cards([card])

    def insert_cards(self, cards: list[MemoryCard]):
        with self._write_lock:
            conn = self._get_conn()
            for card in cards:
                d = card.to_storage_dict()
                conn.execute(
                    "INSERT OR REPLACE INTO memory_cards "
                    "(card_id, task_id, task_domain, task_description, memory_type, content, evidence, provenance, metadata, embedding, card_payload) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        d["card_id"],
                        d["task_id"],
                        d.get("task_domain"),
                        d["task_description"],
                        d["memory_type"],
                        d["content"],
                        d["evidence"],
                        d["provenance"],
                        d["metadata"],
                        _serialize_embedding_blob(card.embedding),
                        d.get("card_payload"),
                    ),
                )
            conn.commit()

    def get_card(self, card_id: str) -> Optional[MemoryCard]:
        row = self._get_conn().execute(
            "SELECT * FROM memory_cards WHERE card_id = ?", (card_id,)
        ).fetchone()
        return MemoryCard.from_storage_dict(dict(row)) if row else None

    def get_cards_by_task(
        self,
        task_id: str,
        active_only: bool = True,
        task_domain: str | None = None,
    ) -> list[MemoryCard]:
        query = "SELECT * FROM memory_cards WHERE task_id = ?"
        params: list[str] = [task_id]
        if task_domain:
            query += " AND (task_domain = ? OR task_domain IS NULL OR task_domain = '')"
            params.append(task_domain)
        rows = self._get_conn().execute(query, tuple(params)).fetchall()
        cards = [MemoryCard.from_storage_dict(dict(r)) for r in rows]
        if active_only:
            cards = [c for c in cards if c.metadata.lifecycle_state == "active"]
        return cards

    def get_cards_by_task_and_type(
        self, task_id: str, memory_type: str, active_only: bool = True
    ) -> list[MemoryCard]:
        cards = self.get_cards_by_task(task_id, active_only=active_only)
        return [c for c in cards if c.matches_memory_type(memory_type)]

    def get_all_active_cards(self, task_domain: str | None = None) -> list[MemoryCard]:
        query = "SELECT * FROM memory_cards"
        params: tuple = ()
        if task_domain:
            query += " WHERE task_domain = ?"
            params = (task_domain,)
        rows = self._get_conn().execute(query, params).fetchall()
        cards = [MemoryCard.from_storage_dict(dict(r)) for r in rows]
        return [c for c in cards if c.metadata.lifecycle_state == "active"]

    def update_card(self, card: MemoryCard):
        self.insert_card(card)

    def record_card_access(self, card_id: str, current_round: int):
        import time

        card = self.get_card(card_id)
        if card:
            card.metadata.access_count += 1
            card.metadata.last_access_time = time.time()
            card.metadata.last_access_round = current_round
            self.update_card(card)

    def count_active_cards(self) -> int:
        return len(self.get_all_active_cards())

    def delete_card(self, card_id: str):
        with self._write_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM memory_cards WHERE card_id = ?", (card_id,))
            conn.execute(
                "DELETE FROM memory_edges WHERE source_card_id = ? OR target_card_id = ?",
                (card_id, card_id),
            )
            conn.commit()

    # ======================= Edge CRUD =======================

    def insert_edge(self, edge: MemoryEdge):
        self.insert_edges([edge])

    def insert_edges(self, edges: list[MemoryEdge]):
        with self._write_lock:
            conn = self._get_conn()
            for e in edges:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_edges "
                    "(source_card_id, target_card_id, relation, weight, rationale) VALUES (?,?,?,?,?)",
                    (e.source_card_id, e.target_card_id, e.relation, e.weight, e.rationale),
                )
            conn.commit()

    def get_edges_for_card(self, card_id: str) -> list[MemoryEdge]:
        rows = self._get_conn().execute(
            "SELECT * FROM memory_edges WHERE source_card_id = ? OR target_card_id = ?",
            (card_id, card_id),
        ).fetchall()
        return [MemoryEdge(**dict(r)) for r in rows]

    def get_edges_for_cards(self, card_ids: set[str]) -> list[MemoryEdge]:
        if not card_ids:
            return []
        placeholders = ",".join("?" for _ in card_ids)
        ids = list(card_ids)
        rows = self._get_conn().execute(
            f"SELECT * FROM memory_edges WHERE source_card_id IN ({placeholders}) "
            f"OR target_card_id IN ({placeholders})",
            ids + ids,
        ).fetchall()
        return [MemoryEdge(**dict(r)) for r in rows]

    def get_conflict_edges_for_cards(self, card_ids: set[str]) -> list[MemoryEdge]:
        return [e for e in self.get_edges_for_cards(card_ids) if e.relation == "conflicts"]

    # ======================= Task CRUD =======================

    def insert_task(self, task: TaskRecord):
        with self._write_lock:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO task_records "
                "(task_id, task_domain, task_description, outcome, completion_round, completion_timestamp, trajectory_file, embedding) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    task.task_id,
                    task.task_domain,
                    task.task_description,
                    task.outcome,
                    task.completion_round,
                    task.completion_timestamp,
                    task.trajectory_file,
                    _serialize_embedding_blob(task.embedding),
                ),
            )
            self._get_conn().commit()

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        row = self._get_conn().execute(
            "SELECT * FROM task_records WHERE task_id = ?", (task_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["embedding"] = _deserialize_embedding_blob(d.get("embedding"))
        return TaskRecord(**d)

    def get_all_tasks(self, task_domain: str | None = None) -> list[TaskRecord]:
        query = "SELECT * FROM task_records"
        params: tuple = ()
        if task_domain:
            query += " WHERE task_domain = ?"
            params = (task_domain,)
        rows = self._get_conn().execute(query, params).fetchall()
        tasks = []
        for r in rows:
            d = dict(r)
            d["embedding"] = _deserialize_embedding_blob(d.get("embedding"))
            tasks.append(TaskRecord(**d))
        return tasks

    # ======================= Trajectory File Storage =======================

    @staticmethod
    def _sanitize_filename_part(value: Optional[str], fallback: str) -> str:
        text = (value or "").strip()
        if not text:
            text = fallback
        text = text.replace(os.sep, "_")
        if os.altsep:
            text = text.replace(os.altsep, "_")
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        return text.strip("._-") or fallback

    def store_trajectory(self, task_id: str, trajectory_data: dict) -> str:
        model_name = self._sanitize_filename_part(
            trajectory_data.get("model_name"), "unknown_model"
        )
        mas_architecture = self._sanitize_filename_part(
            trajectory_data.get("mas_architecture"), "unknown_mas"
        )
        task_domain = self._sanitize_filename_part(
            trajectory_data.get("task_domain"), ""
        )
        safe_task_id = self._sanitize_filename_part(task_id, "unknown_task")
        trajectory_dir = (
            os.path.join(self.trajectory_dir, task_domain)
            if task_domain
            else self.trajectory_dir
        )
        os.makedirs(trajectory_dir, exist_ok=True)
        filepath = os.path.join(
            trajectory_dir,
            f"{mas_architecture}__{model_name}__{safe_task_id}.json",
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, ensure_ascii=False, indent=2)
        return filepath

    def load_trajectory(self, filepath: str) -> Optional[dict]:
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ======================= Consistency Check =======================

    def get_cards_missing_embeddings(self) -> list[MemoryCard]:
        rows = self._get_conn().execute(
            "SELECT * FROM memory_cards WHERE embedding IS NULL OR embedding = 'null' OR LENGTH(embedding) = 0"
        ).fetchall()
        cards = [MemoryCard.from_storage_dict(dict(r)) for r in rows]
        return [c for c in cards if c.metadata.lifecycle_state == "active"]

    def get_tasks_missing_embeddings(self) -> list[TaskRecord]:
        rows = self._get_conn().execute(
            "SELECT * FROM task_records WHERE embedding IS NULL OR embedding = 'null' OR LENGTH(embedding) = 0"
        ).fetchall()
        tasks = []
        for r in rows:
            d = dict(r)
            d["embedding"] = None
            tasks.append(TaskRecord(**d))
        return tasks

    def get_all_card_ids(self) -> set[str]:
        rows = self._get_conn().execute("SELECT card_id FROM memory_cards").fetchall()
        return {r["card_id"] for r in rows}

    def get_orphaned_edges(self) -> list[MemoryEdge]:
        rows = self._get_conn().execute(
            """
            SELECT e.* FROM memory_edges e
            LEFT JOIN memory_cards c1 ON e.source_card_id = c1.card_id
            LEFT JOIN memory_cards c2 ON e.target_card_id = c2.card_id
            WHERE c1.card_id IS NULL OR c2.card_id IS NULL
            """
        ).fetchall()
        return [MemoryEdge(**dict(r)) for r in rows]

    def delete_orphaned_edges(self) -> int:
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                DELETE FROM memory_edges
                WHERE source_card_id NOT IN (SELECT card_id FROM memory_cards)
                   OR target_card_id NOT IN (SELECT card_id FROM memory_cards)
                """
            )
            conn.commit()
            return cursor.rowcount
