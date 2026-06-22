#!/usr/bin/env python3
"""
ConMem card viewer.

This script intentionally reads the SQLite store directly with Python's
standard library. Viewing cards should not require installing the full runtime
dependency stack.

Examples:
    python scripts/view_cards.py
    python scripts/view_cards.py --list-stores
    python scripts/view_cards.py --storage conmem_shared_storage
    python scripts/view_cards.py --storage conmem_shared_storage --graph
    python scripts/view_cards.py --storage conmem_shared_storage --card 6c35 --full
    python scripts/view_cards.py --storage conmem_shared_storage --graph --graph-format dot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_NAME = "conmem.db"
SECTION_ORDER = ("state", "plan", "exec", "eval")
DEFAULT_STORE_DIRS = (
    PROJECT_ROOT / "conmem_shared_storage",
    PROJECT_ROOT / "conmem_storage",
)
STORE_SEARCH_ROOTS = (
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "conmem_shared_storage",
    PROJECT_ROOT / "conmem_storage",
)


def eprint(message: str):
    print(message, file=sys.stderr)


def parse_json(raw: Any, fallback: Any) -> Any:
    if raw is None or raw == "":
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return fallback


def row_get(row: dict[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key)
    return default if value is None else value


def compact_one_line(text: str, max_chars: int = 100) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... [truncated; use --full to show all]"


def wrap_block(text: str, indent: str, width: int) -> str:
    text = (text or "").rstrip()
    if not text:
        return ""
    wrapped_lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.append(
            textwrap.fill(
                paragraph,
                width=max(width - len(indent), 40),
                initial_indent=indent,
                subsequent_indent=indent,
                replace_whitespace=True,
                drop_whitespace=True,
            )
        )
    return "\n".join(wrapped_lines)


def wrap_labeled_block(label: str, text: str, width: int) -> str:
    text = (text or "").strip()
    if not text:
        return label.rstrip()
    subsequent = " " * len(label)
    return textwrap.fill(
        text,
        width=max(width, 40),
        initial_indent=label,
        subsequent_indent=subsequent,
        replace_whitespace=True,
        drop_whitespace=True,
    )


def display_summary(card: dict[str, Any]) -> str:
    summary = (card["summary"] or "").strip()
    task_description = (card["task_description"] or "").strip()
    if task_description and summary.startswith(task_description):
        summary = summary[len(task_description) :].lstrip(" |")
    return summary.strip()


def is_section_preview(text: str) -> bool:
    if len(text) < 240:
        return False
    return bool(re.search(r"(^|\s\|\s)(state|plan|exec|eval):", text, re.IGNORECASE))


def find_store_dirs() -> list[Path]:
    stores: dict[Path, float] = {}

    for store_dir in DEFAULT_STORE_DIRS:
        db_path = store_dir / DB_NAME
        if db_path.exists():
            stores[store_dir] = db_path.stat().st_mtime

    for root in STORE_SEARCH_ROOTS:
        if root.is_file() and root.name == DB_NAME:
            stores[root.parent] = root.stat().st_mtime
            continue
        if not root.exists() or not root.is_dir():
            continue
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
            ]
            if DB_NAME in filenames:
                db_path = Path(current_root) / DB_NAME
                stores[db_path.parent] = db_path.stat().st_mtime

    return [item[0] for item in sorted(stores.items(), key=lambda item: item[1], reverse=True)]


def resolve_storage_dir(storage_arg: str | None) -> Path:
    if storage_arg:
        path = Path(storage_arg).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_file() and path.name == DB_NAME:
            return path.parent
        return path

    stores = find_store_dirs()
    if stores:
        return stores[0]

    return DEFAULT_STORE_DIRS[0]


def db_path_for_storage(storage_dir: Path) -> Path:
    if storage_dir.is_file() and storage_dir.name == DB_NAME:
        return storage_dir
    return storage_dir / DB_NAME


def connect(storage_dir: Path) -> sqlite3.Connection:
    db_path = db_path_for_storage(storage_dir)
    if not db_path.exists():
        raise SystemExit(
            f"ConMem database not found: {db_path}\n"
            "Use --list-stores to discover stores, or pass --storage <dir-or-conmem.db>."
        )
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def parse_sectioned_content(content: str) -> dict[str, str]:
    sections = {section: "" for section in SECTION_ORDER}
    current: str | None = None
    buffers = {section: [] for section in SECTION_ORDER}

    for line in (content or "").splitlines():
        match = re.match(r"^\s*\[(state|plan|exec|eval)\]\s*(.*)$", line, re.IGNORECASE)
        if match:
            current = match.group(1).lower()
            first_text = match.group(2).strip()
            if first_text:
                buffers[current].append(first_text)
            continue
        if current:
            buffers[current].append(line)

    for section in SECTION_ORDER:
        sections[section] = "\n".join(buffers[section]).strip()
    return sections


def normalize_sections(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, str]:
    structured = payload.get("structured_content") or {}
    sections = {section: str(structured.get(section) or "").strip() for section in SECTION_ORDER}
    if any(sections.values()):
        return sections

    content = str(row_get(row, "content", "") or "")
    parsed = parse_sectioned_content(content)
    if any(parsed.values()):
        return parsed

    memory_type = str(row_get(row, "memory_type", "") or "").lower()
    if memory_type in SECTION_ORDER:
        sections[memory_type] = content.strip()
    return sections


def load_cards(conn: sqlite3.Connection, include_inactive: bool = False) -> list[dict[str, Any]]:
    if not has_table(conn, "memory_cards"):
        return []
    columns = table_columns(conn, "memory_cards")
    rows = conn.execute("SELECT * FROM memory_cards ORDER BY task_id, card_id").fetchall()
    cards = []

    for sqlite_row in rows:
        row = dict(sqlite_row)
        payload = parse_json(row.get("card_payload") if "card_payload" in columns else None, {})
        metadata = parse_json(row.get("metadata"), {})
        provenance = parse_json(row.get("provenance"), {})
        sections = normalize_sections(row, payload)
        trigger_semantics = payload.get("trigger_semantics", [])
        if isinstance(trigger_semantics, str):
            trigger_semantics = [trigger_semantics] if trigger_semantics.strip() else []
        quality = payload.get("quality") or {}

        lifecycle_state = str(metadata.get("lifecycle_state") or "active")
        if lifecycle_state != "active" and not include_inactive:
            continue

        content = str(row_get(row, "content", "") or "")
        summary = str(payload.get("summary") or row_get(row, "task_description", "") or "")
        if not summary and content:
            summary = compact_one_line(content, 180)

        card = {
            "card_id": str(row_get(row, "card_id", "")),
            "task_id": str(row_get(row, "task_id", "")),
            "task_domain": str(row_get(row, "task_domain", "")),
            "task_description": str(row_get(row, "task_description", "")),
            "memory_type": str(row_get(row, "memory_type", "")),
            "content": content,
            "sections": sections,
            "evidence": str(row_get(row, "evidence", "") or ""),
            "summary": summary,
            "trigger_semantics": [str(item) for item in trigger_semantics],
            "quality": quality,
            "pattern_signature": payload.get("pattern_signature", ""),
            "provenance": provenance,
            "metadata": metadata,
            "has_embedding": bool(row.get("embedding")),
        }
        cards.append(card)

    return cards


def load_tasks(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not has_table(conn, "task_records"):
        return []
    return [dict(row) for row in conn.execute("SELECT * FROM task_records ORDER BY task_id").fetchall()]


def load_edges(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not has_table(conn, "memory_edges"):
        return []
    rows = conn.execute(
        "SELECT source_card_id, target_card_id, relation, weight, rationale "
        "FROM memory_edges ORDER BY source_card_id, relation, target_card_id"
    ).fetchall()
    return [dict(row) for row in rows]


def filter_cards(cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered = cards
    if args.task:
        filtered = [card for card in filtered if card["task_id"] == args.task]
    if args.domain:
        filtered = [card for card in filtered if card["task_domain"] == args.domain]
    if args.type:
        filtered = [
            card
            for card in filtered
            if card["memory_type"] == args.type or bool(card["sections"].get(args.type, "").strip())
        ]
    if args.card:
        needle = args.card.lower()
        filtered = [
            card
            for card in filtered
            if needle in card["card_id"].lower()
            or needle in card["task_id"].lower()
            or needle in card["summary"].lower()
        ]
    if args.limit > 0:
        filtered = filtered[: args.limit]
    return filtered


def card_label(card: dict[str, Any] | None, card_id: str, full_id: bool = False) -> str:
    if card is None:
        return card_id if full_id else f"{card_id[:10]} missing"
    card_part = card["card_id"] if full_id else card["card_id"][:10]
    task = card["task_id"] or "no-task"
    memory_type = card["memory_type"] or primary_type(card)
    return f"{card_part} {task}/{memory_type}"


def primary_type(card: dict[str, Any]) -> str:
    for section in SECTION_ORDER:
        if card["sections"].get(section, "").strip():
            return section
    return "card"


def card_to_json(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": card["card_id"],
        "task_id": card["task_id"],
        "task_domain": card["task_domain"],
        "task_description": card["task_description"],
        "memory_type": card["memory_type"],
        "summary": card["summary"],
        "trigger_semantics": card["trigger_semantics"],
        "sections": card["sections"],
        "evidence": card["evidence"],
        "quality": card["quality"],
        "pattern_signature": card["pattern_signature"],
        "provenance": card["provenance"],
        "metadata": card["metadata"],
        "has_embedding": card["has_embedding"],
    }


def edge_to_json(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_card_id": edge["source_card_id"],
        "target_card_id": edge["target_card_id"],
        "relation": edge["relation"],
        "weight": edge["weight"],
        "rationale": edge["rationale"],
    }


def print_card(card: dict[str, Any], args: argparse.Namespace):
    print(f"Card ID:      {card['card_id']}")
    print(f"Task ID:      {card['task_id']}")
    if card["task_domain"]:
        print(f"Domain:       {card['task_domain']}")
    if card["task_description"]:
        print("Task:")
        print(wrap_block(truncate(card["task_description"], args.max_chars), "  ", args.width))
    summary = display_summary(card)
    if summary and summary != card["task_description"] and not is_section_preview(summary):
        print("Summary:")
        print(wrap_block(truncate(summary, args.max_chars), "  ", args.width))

    if card["trigger_semantics"]:
        print("Triggers:")
        for trigger in card["trigger_semantics"]:
            print(wrap_block(f"- {truncate(trigger, args.max_chars)}", "  ", args.width))

    print("Content:")
    printed_section = False
    for section in SECTION_ORDER:
        text = card["sections"].get(section, "").strip()
        if not text:
            continue
        printed_section = True
        print(f"  [{section}]")
        print(wrap_block(truncate(text, args.max_chars), "    ", args.width))
    if not printed_section:
        fallback = truncate(card["content"], args.max_chars).strip()
        print(wrap_block(fallback or "(empty)", "  ", args.width))

    if card["evidence"]:
        print("Evidence:")
        print(wrap_block(truncate(card["evidence"], args.max_chars), "  ", args.width))

    quality = card["quality"] or {}
    if quality:
        pieces = []
        for key in ("reliability", "novelty", "relevance", "utility"):
            if key in quality:
                pieces.append(f"{key}={float(quality[key]):.3f}")
        if pieces:
            print(f"Quality:      {', '.join(pieces)}")

    metadata = card["metadata"] or {}
    score = metadata.get("admission_score")
    access_count = metadata.get("access_count")
    create_round = metadata.get("create_round")
    lifecycle = metadata.get("lifecycle_state", "active")
    meta_pieces = [f"status={lifecycle}"]
    if score is not None:
        meta_pieces.append(f"score={float(score):.3f}")
    if access_count is not None:
        meta_pieces.append(f"access={access_count}")
    if create_round is not None:
        meta_pieces.append(f"round={create_round}")
    print(f"Metadata:     {', '.join(meta_pieces)}")

    if card["pattern_signature"]:
        print(f"Pattern:      {card['pattern_signature']}")
    print(f"Embedding:    {'yes' if card['has_embedding'] else 'no'}")

    if args.detail:
        provenance = card["provenance"] or {}
        if provenance:
            print("Provenance:")
            for key in ("source_task_id", "source_agent", "source_step_indices", "trajectory_outcome"):
                if key in provenance:
                    print(wrap_block(f"{key}: {provenance[key]}", "  ", args.width))
    print()


def cmd_cards(cards: list[dict[str, Any]], all_edges: list[dict[str, Any]], args: argparse.Namespace):
    if args.json:
        scope = {card["card_id"] for card in cards}
        edges = [
            edge_to_json(edge)
            for edge in all_edges
            if edge["source_card_id"] in scope or edge["target_card_id"] in scope
        ]
        print(json.dumps({"cards": [card_to_json(card) for card in cards], "edges": edges}, indent=2, ensure_ascii=False))
        return

    if not cards:
        print("No matching cards found.")
        return

    print("=" * 80)
    print(f"Memory Cards ({len(cards)})")
    print("=" * 80)
    print()
    for index, card in enumerate(cards, 1):
        print(f"[{index}] " + "-" * 72)
        print_card(card, args)


def cmd_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace):
    if args.json:
        print(json.dumps(tasks, indent=2, ensure_ascii=False))
        return

    print("=" * 80)
    print(f"Tasks ({len(tasks)})")
    print("=" * 80)
    print()
    for task in tasks:
        print(f"Task ID:      {row_get(task, 'task_id', '')}")
        if row_get(task, "task_domain", ""):
            print(f"Domain:       {row_get(task, 'task_domain', '')}")
        print("Description:")
        print(wrap_block(compact_one_line(row_get(task, "task_description", ""), args.max_chars), "  ", args.width))
        print(f"Outcome:      {row_get(task, 'outcome', '(pending)') or '(pending)'}")
        print(f"Round:        {row_get(task, 'completion_round', 0)}")
        if row_get(task, "trajectory_file", ""):
            print(f"Trajectory:   {row_get(task, 'trajectory_file', '')}")
        print()


def graph_scope_edges(
    selected_cards: list[dict[str, Any]],
    all_cards_by_id: dict[str, dict[str, Any]],
    all_edges: list[dict[str, Any]],
    internal_only: bool = False,
) -> list[dict[str, Any]]:
    selected_ids = {card["card_id"] for card in selected_cards}
    if internal_only:
        return [
            edge
            for edge in all_edges
            if edge["source_card_id"] in selected_ids and edge["target_card_id"] in selected_ids
        ]
    return [
        edge
        for edge in all_edges
        if edge["source_card_id"] in selected_ids or edge["target_card_id"] in selected_ids
    ]


def graph_node_ids(selected_cards: list[dict[str, Any]], edges: list[dict[str, Any]]) -> set[str]:
    ids = {card["card_id"] for card in selected_cards}
    for edge in edges:
        ids.add(edge["source_card_id"])
        ids.add(edge["target_card_id"])
    return ids


def cmd_graph(
    selected_cards: list[dict[str, Any]],
    all_cards_by_id: dict[str, dict[str, Any]],
    all_edges: list[dict[str, Any]],
    args: argparse.Namespace,
):
    edges = graph_scope_edges(selected_cards, all_cards_by_id, all_edges, internal_only=args.internal_only)
    node_ids = graph_node_ids(selected_cards, edges)

    if args.graph_format == "json" or args.json:
        payload = {
            "nodes": [
                card_to_json(all_cards_by_id[node_id])
                if node_id in all_cards_by_id
                else {"card_id": node_id, "missing": True}
                for node_id in sorted(node_ids)
            ],
            "edges": [edge_to_json(edge) for edge in edges],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.graph_format == "dot":
        print("digraph ConMemCards {")
        print('  rankdir="LR";')
        for node_id in sorted(node_ids):
            label = card_label(all_cards_by_id.get(node_id), node_id, full_id=False).replace('"', r"\"")
            print(f'  "{node_id}" [label="{label}"];')
        for edge in edges:
            relation = str(edge["relation"]).replace('"', r"\"")
            weight = float(edge["weight"] or 0.0)
            print(
                f'  "{edge["source_card_id"]}" -> "{edge["target_card_id"]}" '
                f'[label="{relation} {weight:.2f}"];'
            )
        print("}")
        return

    print("=" * 80)
    print(f"Memory Graph ({len(node_ids)} nodes, {len(edges)} edges)")
    print("=" * 80)
    print()

    if not selected_cards:
        print("No matching cards found for graph scope.")
        return
    if not edges:
        print("No edges found in the selected graph scope.")
        return

    relation_counts = Counter(edge["relation"] for edge in edges)
    print("Relations:")
    for relation, count in relation_counts.most_common():
        print(f"  {relation:12s} {count}")
    print()

    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["source_card_id"]].append(edge)
        incoming[edge["target_card_id"]].append(edge)

    selected_ids = {card["card_id"] for card in selected_cards}
    display_ids = sorted(
        node_ids,
        key=lambda node_id: (
            0 if node_id in selected_ids else 1,
            (all_cards_by_id.get(node_id) or {}).get("task_id", ""),
            node_id,
        ),
    )

    for node_id in display_ids:
        card = all_cards_by_id.get(node_id)
        if args.selected_only and node_id not in selected_ids:
            continue
        print(card_label(card, node_id, full_id=args.full_ids))
        if card and (card["summary"] or card["task_description"]):
            summary = display_summary(card) or card["summary"] or card["task_description"]
            if is_section_preview(summary):
                summary = card["task_description"] or card["summary"]
            print(wrap_labeled_block("  summary: ", compact_one_line(summary, 180), args.width))

        node_out = outgoing.get(node_id, [])
        node_in = incoming.get(node_id, [])
        if not node_out and not node_in:
            print("  (isolated in scope)")
            print()
            continue

        for edge in node_out:
            target = card_label(all_cards_by_id.get(edge["target_card_id"]), edge["target_card_id"], args.full_ids)
            print(f"  -> {edge['relation']} w={float(edge['weight'] or 0.0):.2f} {target}")
            if edge.get("rationale"):
                print(wrap_block(str(edge["rationale"]), "     rationale: ", args.width))

        if args.incoming:
            for edge in node_in:
                source = card_label(all_cards_by_id.get(edge["source_card_id"]), edge["source_card_id"], args.full_ids)
                print(f"  <- {edge['relation']} w={float(edge['weight'] or 0.0):.2f} {source}")
                if edge.get("rationale"):
                    print(wrap_block(str(edge["rationale"]), "     rationale: ", args.width))
        print()


def cmd_stats(cards: list[dict[str, Any]], tasks: list[dict[str, Any]], edges: list[dict[str, Any]], args: argparse.Namespace):
    by_type = Counter(card["memory_type"] or primary_type(card) for card in cards)
    by_task = Counter(card["task_id"] for card in cards)
    by_domain = Counter(card["task_domain"] or "(none)" for card in cards)
    outcomes = Counter(row_get(task, "outcome", "") for task in tasks if row_get(task, "outcome", ""))
    by_relation = Counter(edge["relation"] for edge in edges)

    if args.json:
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "active_cards": len(cards),
                    "edges": len(edges),
                    "cards_by_type": dict(by_type),
                    "cards_by_task": dict(by_task),
                    "cards_by_domain": dict(by_domain),
                    "task_outcomes": dict(outcomes),
                    "edges_by_relation": dict(by_relation),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print("=" * 80)
    print("ConMem Statistics")
    print("=" * 80)
    print()
    print(f"Tasks:        {len(tasks)}")
    print(f"Active Cards: {len(cards)}")
    print(f"Graph Edges:  {len(edges)}")
    print()

    print("Cards by Type:")
    for key, count in by_type.most_common():
        print(f"  {key:12s} {count}")
    print()

    print("Cards by Domain:")
    for key, count in by_domain.most_common():
        print(f"  {key:20s} {count}")
    print()

    if outcomes:
        print("Task Outcomes:")
        for key, count in outcomes.most_common():
            print(f"  {key:12s} {count}")
        print()

    if by_relation:
        print("Edges by Relation:")
        for key, count in by_relation.most_common():
            print(f"  {key:12s} {count}")
        print()

    print("Cards per Task:")
    for task_id, count in by_task.most_common():
        print(f"  {task_id:40s} {count}")


def cmd_list_stores():
    stores = find_store_dirs()
    if not stores:
        print("No ConMem stores found under the default search roots.")
        return

    print("Discovered ConMem stores:")
    for index, store_dir in enumerate(stores, 1):
        db_path = store_dir / DB_NAME
        size_kb = db_path.stat().st_size / 1024.0
        print(f"  [{index}] {store_dir} ({size_kb:.1f} KiB)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View ConMem cards and card graph structure.")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="ConMem storage directory or conmem.db path. Defaults to the newest discovered store.",
    )
    parser.add_argument("--list-stores", action="store_true", help="List discovered ConMem stores and exit.")
    parser.add_argument("--task", type=str, default="", help="Filter cards by exact task_id.")
    parser.add_argument("--domain", type=str, default="", help="Filter cards by exact task_domain.")
    parser.add_argument("--type", type=str, default="", choices=["", *SECTION_ORDER], help="Filter by section/type.")
    parser.add_argument("--card", type=str, default="", help="Filter by card_id, task_id, or summary substring.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of printed cards; 0 means no limit.")
    parser.add_argument("--include-inactive", action="store_true", help="Include non-active cards.")
    parser.add_argument("--detail", action="store_true", help="Show provenance details.")
    parser.add_argument("--full", action="store_true", help="Show full text without per-field truncation.")
    parser.add_argument("--max-chars", type=int, default=1200, help="Max chars per long field unless --full is used.")
    parser.add_argument("--width", type=int, default=110, help="Output wrap width.")
    parser.add_argument("--full-ids", action="store_true", help="Show full card IDs in graph labels.")
    parser.add_argument("--json", action="store_true", help="Emit JSON for the selected command.")
    parser.add_argument("--tasks", action="store_true", help="Show task records.")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics.")
    parser.add_argument("--graph", action="store_true", help="Show card graph structure for the selected card scope.")
    parser.add_argument(
        "--graph-format",
        choices=("text", "json", "dot"),
        default="text",
        help="Graph output format.",
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="For --graph, only show edges whose source and target are both in the selected card scope.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="For --graph, only print selected cards as adjacency roots while still showing neighbor labels.",
    )
    parser.add_argument("--incoming", action="store_true", help="For --graph, include incoming edges under each node.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.full:
        args.max_chars = 0

    if args.list_stores:
        cmd_list_stores()
        return

    storage_dir = resolve_storage_dir(args.storage)
    conn = connect(storage_dir)
    eprint(f"Using ConMem store: {db_path_for_storage(storage_dir)}")

    all_cards = load_cards(conn, include_inactive=args.include_inactive)
    label_cards = load_cards(conn, include_inactive=True)
    selected_cards = filter_cards(all_cards, args)
    all_cards_by_id = {card["card_id"]: card for card in label_cards}
    tasks = load_tasks(conn)
    edges = load_edges(conn)

    if args.stats:
        cmd_stats(all_cards, tasks, edges, args)
    elif args.tasks:
        cmd_tasks(tasks, args)
    elif args.graph:
        cmd_graph(selected_cards, all_cards_by_id, edges, args)
    else:
        cmd_cards(selected_cards, edges, args)


if __name__ == "__main__":
    main()
