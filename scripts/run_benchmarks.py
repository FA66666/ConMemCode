#!/usr/bin/env python3
"""
Unified ConMem benchmark evaluation script.

Supports four datasets and four MAS modes:
  Datasets: kodcode, triviaqa, popqa, pddl
  MAS:   single, camel, macnet, autogen

Prerequisite: a local vLLM service is running (default: localhost:8100).

Usage:
    # Single benchmark
    conda run -n conmem python scripts/run_benchmarks.py --benchmark kodcode --mas_type macnet --data_split test --num_tasks 20

    # Run all supported benchmarks
    conda run -n conmem python scripts/run_benchmarks.py --benchmark all --mas_type macnet --data_split test --num_tasks 20

    # Run with no-memory baseline comparison (single and MAS modes).
    conda run -n conmem python scripts/run_benchmarks.py --benchmark kodcode --mas_type macnet --data_split test --num_tasks 50 --run_baseline

    # Specify model and API endpoint
    conda run -n conmem python scripts/run_benchmarks.py --benchmark all --mas_type camel \
        --data_split test --api_base http://localhost:8100/v1 --model_name Qwen/Qwen3-4B-Instruct-2507
"""
import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import types

_PROFILE = os.environ.get("CONMEM_PROFILE")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Import setup to avoid chained __init__.py imports.
# ============================================================
_STUB_PACKAGES = [
    "mas_core", "mas_core.structures",
    "mas_core.structures.camel", "mas_core.structures.camel.prompts",
    "mas_core.structures.macnet", "mas_core.structures.macnet.prompts",
    "mas_core.structures.autogen", "mas_core.structures.autogen.prompts",
    "mas_core.memory", "mas_core.memory.backbone", "mas_core.memory.backbone.conmem",
    "mas_core.memory.backbone.reme",
    "mas_core.memory.backbone.simplemem",
]

def _setup_imports():
    for pkg in _STUB_PACKAGES:
        mod = sys.modules.get(pkg)
        if mod is None:
            mod = types.ModuleType(pkg)
            sys.modules[pkg] = mod
        mod.__path__ = [os.path.join(PROJECT_ROOT, pkg.replace(".", "/"))]
        parent_name, _, attr = pkg.rpartition(".")
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], attr, mod)

_setup_imports()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_benchmarks")


def _consume_retrieval_usage(is_mas_mode, centralized_memory, conmem):
    records = []
    if is_mas_mode and hasattr(centralized_memory, "get_and_reset_retrieval_usage"):
        records = centralized_memory.get_and_reset_retrieval_usage() or []
    elif hasattr(conmem, "get_and_reset_retrieval_usage"):
        records = conmem.get_and_reset_retrieval_usage() or []

    legacy_count = None
    if is_mas_mode and hasattr(centralized_memory, "get_and_reset_retrieval_count"):
        legacy_count = centralized_memory.get_and_reset_retrieval_count()
    elif hasattr(conmem, "_last_card_count"):
        legacy_count = int(getattr(conmem, "_last_card_count", 0) or 0)

    injected_count = int(sum(int(record.get("injected_cards", 0) or 0) for record in records))
    if not records and legacy_count is not None:
        injected_count = int(legacy_count or 0)
    return injected_count, records


def _zero_update_usage(task_id: str, task_index: int) -> dict:
    return {
        "task_id": task_id,
        "task_index": task_index,
        "status": "not_run",
        "candidate_cards": 0,
        "admitted_cards": 0,
        "committed_cards": 0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_time_seconds": 0.0,
        "latency_ms": 0.0,
    }


def _consume_update_usage(conmem, task_id: str, task_index: int) -> dict:
    record = dict(getattr(conmem, "_last_update_usage", {}) or {})
    if not record:
        return _zero_update_usage(task_id, task_index)
    record.setdefault("task_id", task_id)
    record["task_index"] = task_index
    if hasattr(conmem, "_last_update_usage"):
        conmem._last_update_usage = {}
    return record


def _summarize_memory_cost_stats(retrieval_records: list[dict], update_records: list[dict], api_stats: dict, evaluated_tasks: int) -> dict:
    flat_retrievals = [
        record
        for task_record in retrieval_records
        for record in task_record.get("retrievals", [])
    ]
    retrieval_calls = len(flat_retrievals)

    def total_retrieval(field: str) -> float:
        return sum(float(record.get(field, 0) or 0) for record in flat_retrievals)

    def avg_per_task(total_value: float) -> float:
        return total_value / evaluated_tasks if evaluated_tasks else 0.0

    def avg_per_retrieval(total_value: float) -> float:
        return total_value / retrieval_calls if retrieval_calls else 0.0

    total_update_calls = sum(int(record.get("llm_calls", 0) or 0) for record in update_records)
    total_update_prompt = sum(int(record.get("prompt_tokens", 0) or 0) for record in update_records)
    total_update_completion = sum(int(record.get("completion_tokens", 0) or 0) for record in update_records)
    total_update_latency = sum(float(record.get("latency_ms", 0) or 0) for record in update_records)

    total_prompt_tokens = int(api_stats.get("total_prompt_tokens", 0) or 0)
    total_completion_tokens = int(api_stats.get("total_completion_tokens", 0) or 0)
    total_api_calls = int(api_stats.get("total_calls", 0) or 0)

    totals = {
        "retrieved_cards": total_retrieval("retrieved_cards"),
        "expanded_cards": total_retrieval("expanded_cards"),
        "coordinated_cards": total_retrieval("coordinated_cards"),
        "injected_cards": total_retrieval("injected_cards"),
        "injected_tokens_est": total_retrieval("injected_tokens_est"),
        "subgraph_edges": total_retrieval("subgraph_edges"),
        "skipped_over_budget_cards": total_retrieval("skipped_over_budget_cards"),
        "compacted_cards": total_retrieval("compacted_cards"),
        "controller_latency_ms": total_retrieval("latency_ms"),
        "retrieve_latency_ms": total_retrieval("retrieve_latency_ms"),
        "expand_latency_ms": total_retrieval("expand_latency_ms"),
        "coordinate_latency_ms": total_retrieval("coordinate_latency_ms"),
        "serialize_latency_ms": total_retrieval("serialize_latency_ms"),
    }

    per_task = []
    update_by_task = {record.get("task_id"): record for record in update_records}
    for task_record in retrieval_records:
        task_retrievals = task_record.get("retrievals", [])
        task_id = task_record.get("task_id", "")
        update = update_by_task.get(task_id, {})

        def task_total(field: str) -> float:
            return sum(float(record.get(field, 0) or 0) for record in task_retrievals)

        per_task.append({
            "task_id": task_id,
            "task_index": task_record.get("task_index"),
            "retrieval_calls": len(task_retrievals),
            "retrieved_cards": task_total("retrieved_cards"),
            "expanded_cards": task_total("expanded_cards"),
            "coordinated_cards": task_total("coordinated_cards"),
            "injected_cards": task_total("injected_cards"),
            "injected_tokens_est": task_total("injected_tokens_est"),
            "memory_controller_latency_ms": task_total("latency_ms"),
            "memory_update_llm_calls": int(update.get("llm_calls", 0) or 0),
            "memory_update_prompt_tokens": int(update.get("prompt_tokens", 0) or 0),
            "memory_update_latency_ms": float(update.get("latency_ms", 0) or 0),
        })

    return {
        "token_count_note": "injected_tokens_est uses the ConMem serializer estimator; total_prompt_tokens uses provider-reported API usage.",
        "retrieval_calls": retrieval_calls,
        "avg_retrieval_calls_per_task": avg_per_task(retrieval_calls),
        "total_retrieved_cards": totals["retrieved_cards"],
        "avg_retrieved_cards_per_task": avg_per_task(totals["retrieved_cards"]),
        "avg_retrieved_cards_per_retrieval": avg_per_retrieval(totals["retrieved_cards"]),
        "total_expanded_cards": totals["expanded_cards"],
        "avg_expanded_cards_per_task": avg_per_task(totals["expanded_cards"]),
        "avg_expanded_cards_per_retrieval": avg_per_retrieval(totals["expanded_cards"]),
        "total_coordinated_cards": totals["coordinated_cards"],
        "avg_coordinated_cards_per_task": avg_per_task(totals["coordinated_cards"]),
        "avg_coordinated_cards_per_retrieval": avg_per_retrieval(totals["coordinated_cards"]),
        "total_injected_cards": totals["injected_cards"],
        "avg_injected_cards_per_task": avg_per_task(totals["injected_cards"]),
        "avg_injected_cards_per_retrieval": avg_per_retrieval(totals["injected_cards"]),
        "total_injected_tokens_est": totals["injected_tokens_est"],
        "avg_injected_tokens_per_task_est": avg_per_task(totals["injected_tokens_est"]),
        "avg_injected_tokens_per_retrieval_est": avg_per_retrieval(totals["injected_tokens_est"]),
        "total_skipped_over_budget_cards": totals["skipped_over_budget_cards"],
        "avg_skipped_over_budget_cards_per_task": avg_per_task(totals["skipped_over_budget_cards"]),
        "total_compacted_cards": totals["compacted_cards"],
        "avg_memory_controller_latency_ms_per_task": avg_per_task(totals["controller_latency_ms"]),
        "avg_memory_controller_latency_ms_per_retrieval": avg_per_retrieval(totals["controller_latency_ms"]),
        "avg_retrieve_latency_ms_per_retrieval": avg_per_retrieval(totals["retrieve_latency_ms"]),
        "avg_expand_latency_ms_per_retrieval": avg_per_retrieval(totals["expand_latency_ms"]),
        "avg_coordinate_latency_ms_per_retrieval": avg_per_retrieval(totals["coordinate_latency_ms"]),
        "avg_serialize_latency_ms_per_retrieval": avg_per_retrieval(totals["serialize_latency_ms"]),
        "memory_update_llm_calls": total_update_calls,
        "avg_memory_update_llm_calls_per_task": avg_per_task(total_update_calls),
        "memory_update_prompt_tokens": total_update_prompt,
        "avg_memory_update_prompt_tokens_per_task": avg_per_task(total_update_prompt),
        "memory_update_completion_tokens": total_update_completion,
        "avg_memory_update_completion_tokens_per_task": avg_per_task(total_update_completion),
        "avg_memory_update_latency_ms_per_task": avg_per_task(total_update_latency),
        "total_prompt_tokens": total_prompt_tokens,
        "avg_total_prompt_tokens_per_task": avg_per_task(total_prompt_tokens),
        "total_completion_tokens": total_completion_tokens,
        "avg_total_completion_tokens_per_task": avg_per_task(total_completion_tokens),
        "total_api_calls": total_api_calls,
        "avg_api_calls_per_task": avg_per_task(total_api_calls),
        "non_update_prompt_tokens": max(total_prompt_tokens - total_update_prompt, 0),
        "avg_non_update_prompt_tokens_per_task": avg_per_task(max(total_prompt_tokens - total_update_prompt, 0)),
        "per_task": per_task,
    }


# ============================================================
# Lazy imports
# ============================================================
def _import_conmem():
    from mas_core.memory.backbone.conmem.conmem_module import ConMemModule
    from mas_core.memory.backbone.conmem.config import ConMemConfig
    from mas_core.memory.backbone.conmem.centralized_adapter import ConMemCentralizedMemory
    return ConMemModule, ConMemConfig, ConMemCentralizedMemory


def _import_reme_centralized_memory():
    from mas_core.memory.backbone.reme.centralized_adapter import ReMeCentralizedMemory
    return ReMeCentralizedMemory


def _import_simplemem_centralized_memory():
    from mas_core.memory.backbone.simplemem.centralized_adapter import SimpleMemCentralizedMemory
    return SimpleMemCentralizedMemory


def _resolve_memory_storage_dir(shared_storage_dir=None):
    from mas_core.memory.backbone.conmem.storage import resolve_conmem_storage_dir

    return resolve_conmem_storage_dir(
        shared_storage_dir=shared_storage_dir,
        project_root=PROJECT_ROOT,
    )


def _seed_memory_storage_dir(seed_storage_dir=None, target_storage_dir=None, overwrite=False):
    """Copy a source ConMem DB into the writable run storage before evaluation."""
    if not seed_storage_dir:
        return
    if not target_storage_dir:
        raise ValueError("--seed_memory_from_storage_dir requires --memory_storage_dir")

    source_dir = os.path.abspath(seed_storage_dir)
    target_dir = os.path.abspath(target_storage_dir)
    if source_dir == target_dir:
        logger.info("Memory seed source and target are the same; skipping seed copy.")
        return

    source_db = os.path.join(source_dir, "conmem.db")
    target_db = os.path.join(target_dir, "conmem.db")
    if not os.path.exists(source_db):
        raise FileNotFoundError(f"Seed ConMem DB not found: {source_db}")

    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(target_db):
        if not overwrite:
            logger.info(
                "Memory target already has conmem.db; skipping seed copy. "
                "Pass --overwrite_seed_memory to refresh it from the source."
            )
            return
        for suffix in ("", "-wal", "-shm"):
            candidate = target_db + suffix
            if os.path.exists(candidate):
                os.remove(candidate)

    src = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    dst = sqlite3.connect(target_db)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    logger.info("Seeded memory DB from %s to %s", source_db, target_db)


def _load_task_positions_file(path: str | None) -> list[int] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("indices") or data.get("task_indices") or data.get("positions")
    if not isinstance(data, list):
        raise ValueError("--task_indices_file must contain a JSON list or an object with indices")
    positions = []
    for value in data:
        idx = int(value)
        if idx < 0:
            raise ValueError(f"Negative task index in --task_indices_file: {idx}")
        positions.append(idx)
    return positions


def _select_task_positions(tasks, positions: list[int]):
    if not positions:
        return tasks.select([]) if hasattr(tasks, "select") else []
    max_pos = len(tasks) - 1
    out_of_range = [idx for idx in positions if idx > max_pos]
    if out_of_range:
        raise ValueError(
            f"--task_indices_file contains positions outside loaded task range "
            f"0..{max_pos}: {out_of_range[:5]}"
        )
    if hasattr(tasks, "select"):
        return tasks.select(positions)
    return [tasks[idx] for idx in positions]


def _import_mas_class(mas_type):
    if mas_type == "camel":
        from mas_core.structures.camel.camel_main import CamelMemoryMAS
        return CamelMemoryMAS
    elif mas_type == "macnet":
        from mas_core.structures.macnet.macnet_main import MacNetMemoryMAS
        return MacNetMemoryMAS
    elif mas_type == "autogen":
        from mas_core.structures.autogen.autogen_main import AutoGenMemoryMAS
        return AutoGenMemoryMAS
    else:
        raise ValueError(f"Unsupported MAS type: {mas_type}")


# ============================================================
# Dataset loading.
# ============================================================
def _split_kodcode_dataset(split_seed: int = 42):
    from data.kodcode.builder import load_kodcode_splits

    return load_kodcode_splits(split_seed=split_seed)


def load_benchmark(
    name: str,
    num_tasks: int,
    start_from: int,
    data_split: str = "test",
    popqa_sampling: str = "ordered",
    popqa_sample_seed: int = 42,
):
    """Load benchmark datasets and return (tasks, dataset_info)."""
    if name == "kodcode":
        split_seed = 42
        splits = _split_kodcode_dataset(split_seed=split_seed)
        if data_split not in splits:
            raise ValueError(f"KodCode only supports data_split in {list(splits)}")
        ds = splits[data_split]
        end = min(start_from + num_tasks, len(ds))
        tasks = ds.select(range(start_from, end))
        return tasks, {"type": "static", "total": len(ds), "split": data_split, "split_seed": split_seed}

    elif name == "triviaqa":
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
        if data_split == "train":
            selected = ds["train"].select(range(5000))
        elif data_split == "valid":
            selected = ds["validation"].select(range(1000))
        elif data_split == "test":
            selected = ds["validation"].select(range(1000, len(ds["validation"])))
        else:
            raise ValueError("TriviaQA only supports data_split in ['train', 'valid', 'test']")
        end = min(start_from + num_tasks, len(selected))
        tasks = selected.select(range(start_from, end))
        return tasks, {"type": "dynamic", "total": len(selected), "split": data_split}

    elif name == "popqa":
        from datasets import load_dataset
        from data.popqa.sampling import select_popqa_tasks

        ds = load_dataset("akariasai/PopQA")
        if data_split == "valid":
            selected = ds["test"].select(range(6000, 7000))
        elif data_split == "test":
            selected = ds["test"].select(range(7000, len(ds["test"])))
        else:
            raise ValueError("PopQA only supports data_split in ['valid', 'test']")
        tasks, _, sampling_info = select_popqa_tasks(
            selected,
            num_tasks=num_tasks,
            start_from=start_from,
            sampling=popqa_sampling,
            seed=popqa_sample_seed,
        )
        return tasks, {"type": "dynamic", "total": len(selected), "split": data_split, **sampling_info}

    elif name == "pddl":
        data_path = os.path.join(PROJECT_ROOT, "data", "pddl", "test.jsonl")
        with open(data_path) as f:
            all_tasks = [json.loads(line) for line in f]
        per_game_indices = {}
        for task in all_tasks:
            additional = task.setdefault("additional_info", {})
            game = additional.get("subtask", "unknown")
            if additional.get("problem_index") is None:
                additional["problem_index"] = per_game_indices.get(game, 0)
            per_game_indices[game] = int(additional["problem_index"]) + 1
        if data_split != "test":
            raise ValueError("PDDL only supports data_split='test'")
        end = min(start_from + num_tasks, len(all_tasks))
        tasks = all_tasks[start_from:end]
        return tasks, {"type": "dynamic", "total": len(all_tasks), "split": data_split}

    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ============================================================
# KodCode evaluation using KodCodeEnv.
# ============================================================
def eval_kodcode_task(code, test_code, test_info):
    from data.kodcode.env import KodCodeEnv

    env = KodCodeEnv({})
    env.set_env({
        "prompt": "",
        "test": test_code,
        "test_info": test_info,
    })

    try:
        _, score, _ = env.step(code or "")
    except Exception as e:
        logger.warning(f"KodCode evaluation failed: {e}")
        return 0.0, f"Error: {e}"

    detail = env.feedback_detail or {}
    summary = detail.get("summary")
    if summary:
        return score, summary
    if score >= 1.0:
        return score, "All tests passed."
    if score > 0:
        return score, f"Partial pass: {score:.1%}"
    return score, "All tests failed."


# ============================================================
# TriviaQA / PopQA evaluation for dynamic environments.
# ============================================================
def eval_qa_task(response, ground_truth_list):
    """Evaluate a QA task by extracting and matching <answer> content."""
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not matches:
        # If no <answer> tag is present, match the full response.
        answer = response.strip().lower()
    else:
        answer = matches[-1].strip().lower()

    for gt in ground_truth_list:
        if gt.lower() in answer:
            return 1.0, f"Correct (matched: {gt})"
    return 0.0, f"Wrong (answer: {answer[:50]})"


# ============================================================
# PDDL evaluation.
# ============================================================
def eval_pddl_task(response, task_data):
    """Evaluate a PDDL task by checking whether the response contains key goal actions."""
    goal = task_data.get("goal", "").lower()
    response_lower = response.lower()
    # Simple matching: check whether goal keywords appear in the response.
    goal_words = set(goal.split())
    if not goal_words:
        return 0.0, "No goal defined"
    matched = sum(1 for w in goal_words if w in response_lower)
    score = min(1.0, matched / max(len(goal_words), 1))
    return score, f"Goal match: {score:.1%}"


# ============================================================
# Single-agent generation
# ============================================================
def generate_single(llm, system_prompt, user_prompt, memory_context=""):
    full_user = ""
    if memory_context:
        full_user += f"{memory_context}\n\n"
    full_user += user_prompt
    return llm.chat(system_prompt, full_user, temperature=0.3, max_tokens=2048)


# ============================================================
# MAS trajectory conversion
# ============================================================
def compose_prompt_record(system_prompt: str, user_prompt: str, memory_context: str = "") -> str:
    user_parts = []
    if memory_context:
        user_parts.append(memory_context)
    if user_prompt:
        user_parts.append(user_prompt)
    full_user = "\n\n".join(part for part in user_parts if part)
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    if full_user:
        prompt_parts.append(full_user)
    return "\n\n".join(prompt_parts)


def _format_message_node_input(msg, fallback_user_prompt: str = "") -> str:
    if msg is None:
        return fallback_user_prompt
    prompt_parts = []
    formatted_system = getattr(msg, "formatted_system_prompt", None) or ""
    formatted_user = getattr(msg, "formatted_user_prompt", None) or ""
    if formatted_system:
        prompt_parts.append(formatted_system)
    if formatted_user:
        prompt_parts.append(formatted_user)
    if prompt_parts:
        return "\n\n".join(prompt_parts)
    return fallback_user_prompt


def message_graph_to_trajectory(question, message_graph, outcome, feedback, system_prompt=""):
    steps = []
    if message_graph.mas_message_graph is not None:
        import networkx as nx
        try:
            topo_order = list(nx.topological_sort(message_graph.mas_message_graph))
        except Exception:
            topo_order = list(message_graph.mas_message_graph.nodes())
        for idx, node_id in enumerate(topo_order, 1):
            node_data = message_graph.mas_message_graph.nodes[node_id]
            msg = node_data.get("message")
            steps.append({
                "step_index": idx, "agent": str(node_id),
                "input": _format_message_node_input(msg, compose_prompt_record(system_prompt, question)),
                "output": msg.response if msg else "",
                "tool_calls": "", "feedback": feedback if idx == len(topo_order) else "",
            })
    else:
        steps.append({
            "step_index": 1, "agent": "mas", "input": compose_prompt_record(system_prompt, question),
            "output": message_graph.action or "", "tool_calls": "", "feedback": feedback,
        })
    return {"task_description": question, "outcome": outcome, "steps": steps}


# ============================================================
# Main evaluation logic
# ============================================================
def get_benchmark_config(benchmark):
    """Return the system prompt and evaluator for a benchmark."""
    configs = {
        "kodcode": {
            "system_prompt": "You are an expert Python programmer. Write a correct Python function. Output ONLY Python code.",
            "task_domain": "kodcode",
        },
        "triviaqa": {
            "system_prompt": (
                "Answer the given question. You must conduct reasoning inside <think> and </think>. "
                "If you need more info, use <search> query </search>. "
                "Provide your final answer inside <answer> and </answer>."
            ),
            "task_domain": "triviaqa",
        },
        "popqa": {
            "system_prompt": (
                "Answer the given question. You must conduct reasoning inside <think> and </think>. "
                "If you need more info, use <search> query </search>. "
                "Provide your final answer inside <answer> and </answer>."
            ),
            "task_domain": "popqa",
        },
        "pddl": {
            "system_prompt": "You are a planning assistant. Given a planning problem, provide a step-by-step plan to achieve the goal.",
            "task_domain": "pddl",
        },
    }
    return configs[benchmark]


def get_task_question(benchmark, task):
    """Extract the question text from a task record."""
    if benchmark == "kodcode":
        return task["question"].strip()
    elif benchmark in ("triviaqa", "popqa"):
        return task["question"].strip()
    elif benchmark == "pddl":
        return task.get("task", "") or json.dumps(task)
    return str(task)


def get_task_answer(benchmark, task):
    """Extract the gold answer from a task record."""
    if benchmark == "kodcode":
        return {"test": task["test"].strip(), "test_info": task["test_info"]}
    elif benchmark == "triviaqa":
        return task["answer"]["normalized_aliases"]
    elif benchmark == "popqa":
        import ast
        return ast.literal_eval(task["possible_answers"])
    elif benchmark == "pddl":
        return task
    return None


def evaluate_response(benchmark, response, answer):
    """Evaluate a response and return (score, feedback)."""
    if benchmark == "kodcode":
        return eval_kodcode_task(response, answer["test"], answer["test_info"])
    elif benchmark in ("triviaqa", "popqa"):
        return eval_qa_task(response, answer)
    elif benchmark == "pddl":
        return eval_pddl_task(response, answer)
    return 0.0, "Unknown benchmark"


def run_single_benchmark(benchmark, args):
    """Run evaluation for a single benchmark."""
    from utils.config_loader import load_benchmark_config

    ConMemModule, ConMemConfig, ConMemCentralizedMemory = _import_conmem()
    is_mas_mode = args.mas_type in ("camel", "macnet", "autogen")
    memory_backend = getattr(args, "memory_backend", "conmem").lower()
    if memory_backend in ("reme", "simplemem") and not is_mas_mode:
        raise ValueError(f"{memory_backend} memory backend currently supports MAS modes only: camel/macnet/autogen")
    memory_label = {
        "conmem": "ConMem",
        "reme": "ReMe",
        "simplemem": "SimpleMem",
    }.get(memory_backend, memory_backend)
    bench_cfg = get_benchmark_config(benchmark)
    is_dynamic_benchmark = benchmark in {"triviaqa", "popqa", "pddl"}

    # ---- Load config file ----
    if hasattr(args, 'config') and args.config:
        config_path = args.config
    else:
        config_path = None
    
    bench_config = load_benchmark_config(benchmark, config_path)
    logger.info(f"[{benchmark}] Loaded config: max_new_tokens={bench_config.generation.max_new_tokens}, "
                f"temperature={bench_config.generation.temperature}")

    # Use the config model_name when the CLI does not specify one.
    model_name = args.model_name if args.model_name else bench_config.model.llm_name_or_path

    storage_dir = args.storage_dir
    memory_storage_dir = _resolve_memory_storage_dir(args.memory_storage_dir)
    _seed_memory_storage_dir(
        getattr(args, "seed_memory_from_storage_dir", None),
        memory_storage_dir,
        overwrite=getattr(args, "overwrite_seed_memory", False),
    )
    disable_memory_retrieval = bool(getattr(args, "disable_memory_retrieval", False))
    if disable_memory_retrieval and memory_backend != "conmem":
        raise ValueError("--disable_memory_retrieval currently supports only --memory_backend conmem")
    results_dir = os.path.join(args.storage_dir, benchmark)
    env_path = os.path.join(PROJECT_ROOT, ".env")

    # Initialize memory and MAS.
    config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
    dynamic_max_turns = args.max_turns if getattr(args, "max_turns", None) else bench_config.interaction.max_turns
    dynamic_timeout_seconds = (
        args.timeout_seconds
        if getattr(args, "timeout_seconds", None) is not None
        else getattr(bench_config.interaction, "timeout_seconds", None)
    )
    max_obs_length = getattr(bench_config.interaction, "max_obs_length", None)

    dynamic_env_kwargs = {}
    run_dynamic_single = None
    run_dynamic_mas = None
    dynamic_env_cls = None
    if benchmark == "triviaqa":
        from data.triviaqa.env import TriviaQAEnv
        from scripts.eval_triviaqa import (
            run_multiturn_mas as run_triviaqa_mas,
            run_multiturn_single as run_triviaqa_single,
        )

        dynamic_env_cls = TriviaQAEnv
        run_dynamic_single = run_triviaqa_single
        run_dynamic_mas = run_triviaqa_mas
        dynamic_env_kwargs = config.factual_qa_retriever_kwargs_for_domain("triviaqa")
    elif benchmark == "popqa":
        from data.popqa.env import PopQAEnv
        from scripts.eval_popqa import (
            run_multiturn_mas as run_popqa_mas,
            run_multiturn_single as run_popqa_single,
        )

        dynamic_env_cls = PopQAEnv
        run_dynamic_single = run_popqa_single
        run_dynamic_mas = run_popqa_mas
        dynamic_env_kwargs = config.factual_qa_retriever_kwargs_for_domain("popqa")
    elif benchmark == "pddl":
        from data.pddl.env.pddl_env import PDDLEnv
        from scripts.eval_pddl import (
            run_multiturn_mas as run_pddl_mas,
            run_multiturn_single as run_pddl_single,
        )

        dynamic_env_cls = PDDLEnv
        run_dynamic_single = run_pddl_single
        run_dynamic_mas = run_pddl_mas

    if benchmark in {"triviaqa", "popqa"} and getattr(args, "search_url", None):
        dynamic_env_kwargs["search_url"] = args.search_url

    baseline_mas = None
    reme_workspace_id = None
    simplemem_storage_dir = None

    if is_mas_mode:
        if memory_backend == "reme":
            ReMeCentralizedMemory = _import_reme_centralized_memory()
            workspace_id = args.reme_workspace or f"reme:{benchmark}:{args.mas_type}:{model_name}"
            reme_workspace_id = workspace_id
            centralized_memory = ReMeCentralizedMemory(
                base_url=args.reme_url,
                workspace_id=workspace_id,
                task_domain=benchmark,
                top_k=args.reme_top_k,
                timeout=args.reme_timeout,
                read_only=args.read_only_memory,
                trajectory_dir=args.reme_trajectory_dir or os.path.join(results_dir, "reme_trajectories"),
            )
            conmem = centralized_memory
            logger.info(f"[{benchmark}] Using ReMe workspace: {workspace_id} ({args.reme_url})")
        elif memory_backend == "simplemem":
            SimpleMemCentralizedMemory = _import_simplemem_centralized_memory()
            simplemem_storage_dir = args.simplemem_storage_dir or os.path.join(results_dir, "simplemem_memory")
            centralized_memory = SimpleMemCentralizedMemory(
                storage_dir=simplemem_storage_dir,
                task_domain=benchmark,
                top_k=args.simplemem_top_k,
                embedding_backend=args.simplemem_embedding_backend,
                embedding_model=args.simplemem_embedding_model,
                embedding_api_base=args.simplemem_embedding_api_base,
                embedding_api_key=args.simplemem_embedding_api_key,
                embedding_timeout=args.simplemem_embedding_timeout,
                read_only=args.read_only_memory,
                trajectory_dir=args.simplemem_trajectory_dir or os.path.join(results_dir, "simplemem_trajectories"),
            )
            conmem = centralized_memory
            logger.info(f"[{benchmark}] Using SimpleMem memory store: {simplemem_storage_dir}")
        else:
            centralized_memory = ConMemCentralizedMemory(config, memory_storage_dir, task_domain=benchmark)
            centralized_memory.disable_retrieval = disable_memory_retrieval
            conmem = centralized_memory.conmem
        MASClass = _import_mas_class(args.mas_type)
        mas = MASClass(
            llm_name_or_path=model_name,
            centralized_memory=centralized_memory,
            share_llm=True,
            task_domain=bench_cfg["task_domain"],
            api_base=args.api_base,
            model_name=model_name,
        )
        # Use generation settings from the config file.
        generation_config = {
            "max_new_tokens": bench_config.generation.max_new_tokens,
            "temperature": bench_config.generation.temperature,
            "top_p": 0.95,
            "do_sample": bench_config.generation.temperature > 0,
        }
        if args.run_baseline:
            baseline_mas = MASClass(
                llm_name_or_path=model_name,
                centralized_memory=None,
                share_llm=True,
                task_domain=bench_cfg["task_domain"],
                api_base=args.api_base,
                model_name=model_name,
            )
        logger.info(f"[{benchmark}] {args.mas_type.upper()} MAS initialized ({len(mas.agents_list)} agents)")
    else:
        conmem = ConMemModule(config, memory_storage_dir, task_domain=benchmark)
        conmem.set_runtime_context(model_name=model_name, mas_architecture="single")
        logger.info(f"[{benchmark}] Single agent mode")

    if memory_backend == "conmem":
        logger.info(f"[{benchmark}] ConMem shared storage: {memory_storage_dir}")
    elif memory_backend == "reme":
        logger.info(f"[{benchmark}] ReMe HTTP memory: {args.reme_url}")
    elif memory_backend == "simplemem":
        logger.info(f"[{benchmark}] SimpleMem memory store: {simplemem_storage_dir}")

    # Load data
    logger.info(f"[{benchmark}] Loading dataset...")
    tasks, ds_info = load_benchmark(
        benchmark,
        args.num_tasks,
        args.start_from,
        args.data_split,
        popqa_sampling=args.popqa_sampling,
        popqa_sample_seed=args.popqa_sample_seed,
    )
    task_positions = _load_task_positions_file(getattr(args, "task_indices_file", None))
    if task_positions is not None:
        tasks = _select_task_positions(tasks, task_positions)
        ds_info["selected_task_positions_file"] = os.path.abspath(args.task_indices_file)
        ds_info["selected_task_positions"] = task_positions
    num_tasks = len(tasks)
    logger.info(
        f"[{benchmark}] {num_tasks} tasks loaded from split={ds_info.get('split', 'n/a')} "
        f"(total available: {ds_info['total']})"
    )

    # Evaluation loop
    results = []
    baseline_results = []
    retrieved_card_counts = []
    memory_retrieval_usage_records = []
    memory_update_usage_records = []
    won_count = 0
    baseline_won_count = 0
    t_start = time.time()

    profiler = None
    if _PROFILE:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    for i in range(num_tasks):
        task = tasks[i]
        task_idx = args.start_from + i
        question = get_task_question(benchmark, task)
        answer = get_task_answer(benchmark, task)
        task_id = f"{benchmark}_{args.data_split}_{task_idx}"
        if hasattr(conmem, "_last_update_usage"):
            conmem._last_update_usage = {}

        logger.info(f"\n[{benchmark}] Task {task_idx} ({i+1}/{num_tasks}): {question[:80]}...")
        code_function_name = ""
        if benchmark == "kodcode":
            test_info = answer["test_info"]
            if test_info:
                code_function_name = str(test_info[0].get("function_name", "")).strip()

        resp_base = None
        if benchmark == "kodcode" and args.run_baseline and is_mas_mode:
            try:
                graphs_base = baseline_mas.generate(
                    task_domain_instructions=[bench_cfg["system_prompt"]],
                    user_inputs=[question],
                    generation_config=generation_config,
                    function_names=[code_function_name],
                )
                resp_base = graphs_base[0].action or ""
            except Exception as e:
                logger.error(f"  [Baseline MAS] Generation failed: {e}")
                resp_base = ""
        elif benchmark == "kodcode" and args.run_baseline:
            resp_base = generate_single(conmem.llm, bench_cfg["system_prompt"], question)

        if resp_base is not None:
            score_base, fb_base = evaluate_response(benchmark, resp_base, answer)
            baseline_results.append(score_base)
            logger.info(f"  [Baseline] score={score_base:.1f} {fb_base[:60]}")

        if benchmark == "kodcode":
            # ConMem + MAS/Single
            graphs = None
            memory_ctx = ""
            if is_mas_mode:
                try:
                    graphs = mas.generate(
                        task_domain_instructions=[bench_cfg["system_prompt"]],
                        user_inputs=[question],
                        generation_config=generation_config,
                        function_names=[code_function_name],
                    )
                    response = graphs[0].action or ""
                except Exception as e:
                    logger.error(f"  [MAS] Generation failed: {e}")
                    response = ""
                    graphs = None
            else:
                memory_ctx = "" if disable_memory_retrieval else conmem.on_task_start(
                    task_description=question, agent_role="executor", task_id=task_id)
                response = generate_single(
                    conmem.llm, bench_cfg["system_prompt"], question, memory_context=memory_ctx)

            score, feedback = evaluate_response(benchmark, response, answer)
            results.append(score)
            logger.info(f"  [{args.mas_type.upper()}+{memory_label}] score={score:.1f} {feedback[:60]}")

            outcome = "success" if score >= 1.0 else ("partial" if score > 0 else "failure")
            if getattr(args, "read_only_memory", False):
                logger.info("  [Read-Only Mode] Skipping memory storage")
            elif is_mas_mode and graphs is not None:
                traj = message_graph_to_trajectory(
                    question,
                    graphs[0],
                    outcome,
                    feedback,
                    system_prompt=bench_cfg["system_prompt"],
                )
                centralized_memory.add_memory(
                    trajectory=traj, task_id=task_id,
                    task_description=question, outcome=outcome)
            elif not is_mas_mode:
                prompt_record = compose_prompt_record(bench_cfg["system_prompt"], question, memory_ctx)
                conmem.on_task_complete(
                    task_id=task_id, task_description=question, outcome=outcome,
                    trajectory={
                        "task_description": question, "outcome": outcome,
                        "steps": [{"step_index": 1, "agent": "executor",
                                   "input": prompt_record, "output": response,
                                   "tool_calls": "", "feedback": feedback}],
                    })
            num_cards, retrieval_usage = _consume_retrieval_usage(is_mas_mode, centralized_memory if is_mas_mode else None, conmem)
            retrieved_card_counts.append(num_cards)
            memory_retrieval_usage_records.append({
                "task_id": task_id,
                "task_index": task_idx,
                "retrievals": retrieval_usage,
            })
            memory_update_usage_records.append(_consume_update_usage(conmem, task_id, task_idx))
        elif benchmark == "triviaqa":
            env_task_config = {"prompt": question, "answer": answer}

            if args.run_baseline and is_mas_mode:
                env_base = dynamic_env_cls(dynamic_env_kwargs)
                base_system_prompt, base_question = env_base.set_env(env_task_config)
                score_base, _, _ = run_dynamic_mas(
                    baseline_mas,
                    env_base,
                    base_system_prompt,
                    base_question,
                    answer,
                    generation_config,
                    dynamic_max_turns,
                    action_resolver=env_base.explorer.batch_search,
                    max_obs_length=max_obs_length,
                )
                baseline_results.append(score_base)
                logger.info(f"  [Baseline] score={score_base:.1f}")
            elif args.run_baseline:
                env_base = dynamic_env_cls(dynamic_env_kwargs)
                base_system_prompt, base_question = env_base.set_env(env_task_config)
                resp_base = conmem.llm.chat(base_system_prompt, base_question, temperature=0.0, max_tokens=2048)
                env_base.step(resp_base)
                score_base = env_base.feedback()
                baseline_results.append(score_base)
                logger.info(f"  [Baseline] score={score_base:.1f}")

            env = dynamic_env_cls(dynamic_env_kwargs)
            system_prompt, env_question = env.set_env(env_task_config)

            if is_mas_mode:
                score, response, steps = run_dynamic_mas(
                    mas,
                    env,
                    system_prompt,
                    env_question,
                    answer,
                    generation_config,
                    dynamic_max_turns,
                    action_resolver=env.explorer.batch_search,
                    max_obs_length=max_obs_length,
                )
            else:
                memory_ctx = "" if disable_memory_retrieval else conmem.on_task_start(task_description=question, agent_role="executor", task_id=task_id)
                score, response, steps = run_dynamic_single(
                    conmem.llm,
                    env,
                    system_prompt,
                    env_question,
                    memory_ctx,
                    dynamic_max_turns,
                    max_obs_length=max_obs_length,
                    conmem=conmem,
                    task_id=task_id,
                    agent_role="executor",
                    task_description=question,
                )

            results.append(score)
            feedback = f"score={score:.1f} (turns={len(steps)}, gt={answer[0] if answer else 'N/A'})"
            logger.info(f"  [{args.mas_type.upper()}+{memory_label}] {feedback}")

            outcome = "success" if score >= 1.0 else ("partial" if score > 0 else "failure")
            traj_data = {"task_description": question, "outcome": outcome, "steps": steps}
            if getattr(args, "read_only_memory", False):
                logger.info("  [Read-Only Mode] Skipping memory storage")
            elif is_mas_mode:
                centralized_memory.add_memory(
                    trajectory=traj_data,
                    task_id=task_id,
                    task_description=question,
                    outcome=outcome,
                )
            else:
                conmem.on_task_complete(
                    task_id=task_id,
                    task_description=question,
                    outcome=outcome,
                    trajectory=traj_data,
                )
            num_cards, retrieval_usage = _consume_retrieval_usage(is_mas_mode, centralized_memory if is_mas_mode else None, conmem)
            retrieved_card_counts.append(num_cards)
            memory_retrieval_usage_records.append({
                "task_id": task_id,
                "task_index": task_idx,
                "retrievals": retrieval_usage,
            })
            memory_update_usage_records.append(_consume_update_usage(conmem, task_id, task_idx))
        elif benchmark == "popqa":
            env_task_config = {
                "prompt": question,
                "answer": answer,
            }

            if args.run_baseline and is_mas_mode:
                env_base = dynamic_env_cls(dynamic_env_kwargs)
                base_system_prompt, base_question = env_base.set_env(env_task_config)
                score_base, _, _ = run_dynamic_mas(
                    baseline_mas,
                    env_base,
                    base_system_prompt,
                    base_question,
                    answer,
                    generation_config,
                    dynamic_max_turns,
                    action_resolver=env_base.explorer.batch_search,
                    max_obs_length=max_obs_length,
                )
                baseline_results.append(score_base)
                logger.info(f"  [Baseline] score={score_base:.1f}")
            elif args.run_baseline:
                env_base = dynamic_env_cls(dynamic_env_kwargs)
                base_system_prompt, base_question = env_base.set_env(env_task_config)
                resp_base = conmem.llm.chat(base_system_prompt, base_question, temperature=0.0, max_tokens=2048)
                env_base.step(resp_base)
                score_base = env_base.feedback()
                baseline_results.append(score_base)
                logger.info(f"  [Baseline] score={score_base:.1f}")

            env = dynamic_env_cls(dynamic_env_kwargs)
            system_prompt, env_question = env.set_env(env_task_config)

            if is_mas_mode:
                score, response, steps = run_dynamic_mas(
                    mas,
                    env,
                    system_prompt,
                    env_question,
                    answer,
                    generation_config,
                    dynamic_max_turns,
                    action_resolver=env.explorer.batch_search,
                    max_obs_length=max_obs_length,
                )
            else:
                memory_ctx = "" if disable_memory_retrieval else conmem.on_task_start(task_description=question, agent_role="executor", task_id=task_id)
                score, response, steps = run_dynamic_single(
                    conmem.llm,
                    env,
                    system_prompt,
                    env_question,
                    memory_ctx,
                    dynamic_max_turns,
                    max_obs_length=max_obs_length,
                    conmem=conmem,
                    task_id=task_id,
                    agent_role="executor",
                    task_description=question,
                )

            results.append(score)
            feedback = f"score={score:.1f} (turns={len(steps)}, gt={answer[0] if answer else 'N/A'})"
            logger.info(f"  [{args.mas_type.upper()}+{memory_label}] {feedback}")

            outcome = "success" if score >= 1.0 else ("partial" if score > 0 else "failure")
            traj_data = {"task_description": question, "outcome": outcome, "steps": steps}
            if getattr(args, "read_only_memory", False):
                logger.info("  [Read-Only Mode] Skipping memory storage")
            elif is_mas_mode:
                centralized_memory.add_memory(
                    trajectory=traj_data,
                    task_id=task_id,
                    task_description=question,
                    outcome=outcome,
                )
            else:
                conmem.on_task_complete(
                    task_id=task_id,
                    task_description=question,
                    outcome=outcome,
                    trajectory=traj_data,
                )
            num_cards, retrieval_usage = _consume_retrieval_usage(is_mas_mode, centralized_memory if is_mas_mode else None, conmem)
            retrieved_card_counts.append(num_cards)
            memory_retrieval_usage_records.append({
                "task_id": task_id,
                "task_index": task_idx,
                "retrievals": retrieval_usage,
            })
            memory_update_usage_records.append(_consume_update_usage(conmem, task_id, task_idx))
        elif benchmark == "pddl":
            game_name = task.get("additional_info", {}).get("subtask", "unknown")
            difficulty = task.get("difficulty", "unknown")
            env_task_config = {
                "game_name": game_name,
                "problem_index": task.get("additional_info", {}).get("problem_index", task.get("problem_index", i)),
                "goal": task.get("goal"),
                "subgoals": task.get("subgoals", []),
                "difficulty": difficulty,
                "id": task.get("id"),
            }
            logger.info(f"  [PDDL] game={game_name} difficulty={difficulty}")

            env = dynamic_env_cls({})
            try:
                system_prompt, init_user_prompt = env.set_env(env_task_config)
            except (ValueError, IndexError) as e:
                logger.error(f"  Task {task_idx} env setup failed, recording as score=0: {e}")
                results.append(0.0)
                retrieved_card_counts.append(0)
                memory_retrieval_usage_records.append({
                    "task_id": task_id,
                    "task_index": task_idx,
                    "retrievals": [],
                })
                memory_update_usage_records.append(_zero_update_usage(task_id, task_idx))
                continue
            task_description = f"{game_name}: {env._get_goal()}"

            if args.run_baseline:
                env_base = dynamic_env_cls({})
                try:
                    base_system_prompt, base_init_user_prompt = env_base.set_env(env_task_config)
                except (IndexError, Exception) as e:
                    logger.warning(f"  [Baseline] Skipping task {task_idx}: env setup failed ({e})")
                    base_system_prompt = None
                    base_init_user_prompt = None
                if base_system_prompt is not None:
                    if is_mas_mode:
                        score_base, won_base, _ = run_dynamic_mas(
                            baseline_mas,
                            env_base,
                            base_system_prompt,
                            base_init_user_prompt,
                            generation_config,
                            dynamic_max_turns,
                            timeout_seconds=dynamic_timeout_seconds,
                            memory_task_description=f"{game_name}: {env_base._get_goal()}",
                        )
                    else:
                        score_base, won_base, _ = run_dynamic_single(
                            conmem.llm,
                            env_base,
                            base_system_prompt,
                            base_init_user_prompt,
                            memory_ctx="",
                            max_turns=dynamic_max_turns,
                            timeout_seconds=dynamic_timeout_seconds,
                        )
                    baseline_results.append(score_base)
                    if won_base:
                        baseline_won_count += 1
                    logger.info(f"  [Baseline] score={score_base:.2f}, won={won_base}")

            if is_mas_mode:
                score, won, steps = run_dynamic_mas(
                    mas,
                    env,
                    system_prompt,
                    init_user_prompt,
                    generation_config,
                    dynamic_max_turns,
                    timeout_seconds=dynamic_timeout_seconds,
                    memory_task_description=task_description,
                )
            else:
                memory_ctx = "" if disable_memory_retrieval else conmem.on_task_start(
                    task_description=task_description,
                    agent_role="executor",
                    task_id=task_id,
                )
                score, won, steps = run_dynamic_single(
                    conmem.llm,
                    env,
                    system_prompt,
                    init_user_prompt,
                    memory_ctx,
                    dynamic_max_turns,
                    conmem=conmem,
                    task_id=task_id,
                    agent_role="executor",
                    task_description=task_description,
                    timeout_seconds=dynamic_timeout_seconds,
                )

            results.append(score)
            if won:
                won_count += 1
            feedback = f"score={score:.2f}, won={won}, turns={len(steps)}"
            logger.info(f"  [{args.mas_type.upper()}+{memory_label}] {feedback}")

            outcome = "success" if won else ("partial" if score > 0 else "failure")
            traj_data = {"task_description": task_description, "outcome": outcome, "steps": steps}
            if getattr(args, "read_only_memory", False):
                logger.info("  [Read-Only Mode] Skipping memory storage")
            elif is_mas_mode:
                centralized_memory.add_memory(
                    trajectory=traj_data,
                    task_id=task_id,
                    task_description=task_description,
                    outcome=outcome,
                )
            else:
                conmem.on_task_complete(
                    task_id=task_id,
                    task_description=task_description,
                    outcome=outcome,
                    trajectory=traj_data,
                )
            num_cards, retrieval_usage = _consume_retrieval_usage(is_mas_mode, centralized_memory if is_mas_mode else None, conmem)
            retrieved_card_counts.append(num_cards)
            memory_retrieval_usage_records.append({
                "task_id": task_id,
                "task_index": task_idx,
                "retrievals": retrieval_usage,
            })
            memory_update_usage_records.append(_consume_update_usage(conmem, task_id, task_idx))

        # Progress
        if (i + 1) % 10 == 0 or i == num_tasks - 1:
            rate = sum(results) / len(results)
            if benchmark == "pddl":
                logger.info(f"  [{benchmark}] Progress: {len(results)}/{num_tasks} avg_score={rate:.2%} won={won_count}/{len(results)}")
            else:
                logger.info(f"  [{benchmark}] Progress: {len(results)}/{num_tasks} pass_rate={rate:.2%}")

    if profiler is not None:
        profiler.disable()
        import io
        import pstats
        os.makedirs(results_dir, exist_ok=True)
        profile_path = os.path.join(results_dir, f"profile_{benchmark}_{args.mas_type}.prof")
        profiler.dump_stats(profile_path)
        logger.info(f"[{benchmark}] cProfile saved to {profile_path}")
        for sort_key in ("cumulative", "tottime"):
            buf = io.StringIO()
            pstats.Stats(profile_path, stream=buf).strip_dirs().sort_stats(sort_key).print_stats(30)
            logger.info(f"[{benchmark}] ===== cProfile top 30 ({sort_key}) =====\n{buf.getvalue()}")

    elapsed = time.time() - t_start
    pass_rate = sum(results) / len(results) if results else 0
    evaluated_tasks = len(results)

    # Statistics
    from utils.stats import stats
    reme_usage_stats = centralized_memory.get_usage_stats() if memory_backend == "reme" and is_mas_mode else None
    reme_trajectory_files = centralized_memory.get_saved_trajectory_files() if memory_backend == "reme" and is_mas_mode else None
    simplemem_usage_stats = centralized_memory.get_usage_stats() if memory_backend == "simplemem" and is_mas_mode else None
    simplemem_trajectory_files = centralized_memory.get_saved_trajectory_files() if memory_backend == "simplemem" and is_mas_mode else None
    tasks_with_cards = sum(1 for count in retrieved_card_counts if count > 0)
    total_cards_used = int(sum(retrieved_card_counts))
    api_stats = stats.to_dict()
    memory_cost_stats = _summarize_memory_cost_stats(
        memory_retrieval_usage_records,
        memory_update_usage_records,
        api_stats,
        evaluated_tasks,
    )
    summary = {
        "benchmark": benchmark,
        "mas_type": args.mas_type,
        "memory_backend": memory_backend,
        "reme_url": args.reme_url if memory_backend == "reme" else None,
        "reme_workspace": reme_workspace_id if memory_backend == "reme" else None,
        "simplemem_storage_dir": simplemem_storage_dir if memory_backend == "simplemem" else None,
        "simplemem_top_k": args.simplemem_top_k if memory_backend == "simplemem" else None,
        "simplemem_embedding_backend": args.simplemem_embedding_backend if memory_backend == "simplemem" else None,
        "simplemem_embedding_model": args.simplemem_embedding_model if memory_backend == "simplemem" else None,
        "simplemem_embedding_api_base": args.simplemem_embedding_api_base if memory_backend == "simplemem" else None,
        "model_name": model_name,
        "config_model": bench_config.model.llm_name_or_path,
        "max_new_tokens": bench_config.generation.max_new_tokens,
        "temperature": bench_config.generation.temperature,
        "data_split": args.data_split,
        "num_tasks": evaluated_tasks,
        "start_from": args.start_from,
        "disable_memory_retrieval": disable_memory_retrieval,
        "task_indices_file": os.path.abspath(args.task_indices_file) if args.task_indices_file else None,
        "pass_rate": pass_rate,
        "scores": results,
        "elapsed_seconds": round(elapsed, 1),
        "memory_cards": conmem.storage.count_active_cards(),
        "dataset_total": ds_info["total"],
        "api_stats": api_stats,
        "reme_usage_stats": reme_usage_stats,
        "reme_trajectory_files": reme_trajectory_files,
        "simplemem_usage_stats": simplemem_usage_stats,
        "simplemem_trajectory_files": simplemem_trajectory_files,
        "memory_card_stats": {
            "tasks_with_cards": tasks_with_cards,
            "tasks_without_cards": max(evaluated_tasks - tasks_with_cards, 0),
            "card_utilization": (tasks_with_cards / evaluated_tasks) if evaluated_tasks else 0.0,
            "total_cards_used": total_cards_used,
            "avg_cards_per_task": (total_cards_used / evaluated_tasks) if evaluated_tasks else 0.0,
            "retrieved_card_counts": retrieved_card_counts,
        },
        "memory_cost_stats": memory_cost_stats,
    }
    if "split_seed" in ds_info:
        summary["split_seed"] = ds_info["split_seed"]
    if "sampling" in ds_info:
        summary["sampling"] = ds_info["sampling"]
    if "sample_seed" in ds_info:
        summary["sample_seed"] = ds_info["sample_seed"]
    if "selected_indices" in ds_info:
        summary["selected_indices"] = ds_info["selected_indices"]
    if "selected_task_positions" in ds_info:
        summary["selected_task_positions"] = ds_info["selected_task_positions"]
    if baseline_results:
        summary["baseline_pass_rate"] = sum(baseline_results) / len(baseline_results)
        summary["baseline_scores"] = baseline_results
    if is_dynamic_benchmark:
        summary["max_turns"] = dynamic_max_turns
        if dynamic_timeout_seconds is not None:
            summary["timeout_seconds"] = dynamic_timeout_seconds
    if benchmark == "pddl":
        summary["avg_score"] = pass_rate
        summary["win_rate"] = won_count / evaluated_tasks if evaluated_tasks else 0
        summary["won"] = won_count
        if baseline_results:
            summary["baseline_avg_score"] = sum(baseline_results) / len(baseline_results)
            summary["baseline_win_rate"] = baseline_won_count / len(baseline_results)
            summary["baseline_won"] = baseline_won_count

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    if benchmark == "pddl":
        logger.info(
            f"[{benchmark}] RESULTS: avg_score={pass_rate:.2%} win_rate={summary['win_rate']:.2%} "
            f"({won_count}/{evaluated_tasks}) time={elapsed:.1f}s cards={conmem.storage.count_active_cards()}"
        )
    else:
        logger.info(
            f"[{benchmark}] RESULTS: pass_rate={pass_rate:.2%} ({int(sum(results))}/{evaluated_tasks}) "
            f"time={elapsed:.1f}s cards={conmem.storage.count_active_cards()}"
        )
    logger.info(
        f"[{benchmark}] COST: injected_tokens/task~{memory_cost_stats['avg_injected_tokens_per_task_est']:.1f} "
        f"retrieved/expanded/coordinated cards/task="
        f"{memory_cost_stats['avg_retrieved_cards_per_task']:.2f}/"
        f"{memory_cost_stats['avg_expanded_cards_per_task']:.2f}/"
        f"{memory_cost_stats['avg_coordinated_cards_per_task']:.2f} "
        f"controller_ms/retrieval={memory_cost_stats['avg_memory_controller_latency_ms_per_retrieval']:.1f} "
        f"update_llm_calls/task={memory_cost_stats['avg_memory_update_llm_calls_per_task']:.2f} "
        f"prompt_tokens/task={memory_cost_stats['avg_total_prompt_tokens_per_task']:.1f}"
    )
    logger.info(f"[{benchmark}] Saved to {results_file}")

    return summary


# ============================================================
# Main entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ConMem Unified Benchmark Runner")
    parser.add_argument("--config", type=str, default=None,
                        help="Config directory or file path (default: configs/conmem/*.yaml).")
    parser.add_argument("--benchmark", type=str, default="kodcode",
                        choices=["kodcode", "triviaqa", "popqa", "pddl", "all"],
                        help="Benchmark to run (default: kodcode)")
    parser.add_argument("--mas_type", type=str, default="single",
                        choices=["single", "camel", "macnet", "autogen"],
                        help="MAS type (default: single)")
    parser.add_argument("--api_base", type=str, default="http://localhost:8100/v1")
    parser.add_argument("--search_url", type=str, default=None,
                        help="Optional override for QA retrieval endpoint")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (default: value from config).")
    parser.add_argument("--num_tasks", type=int, default=20,
                        help="Number of tasks per benchmark (default: 20)")
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--data_split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Dataset split to use when supported (default: test)")
    parser.add_argument("--popqa_sampling", type=str, default="ordered",
                        choices=["ordered", "stratified_by_prop"],
                        help="PopQA task selection strategy (default: ordered)")
    parser.add_argument("--popqa_sample_seed", type=int, default=42,
                        help="Seed for PopQA stratified sampling")
    parser.add_argument("--storage_dir", type=str, default="./conmem_storage",
                        help="Benchmark result output directory")
    parser.add_argument("--memory_storage_dir", type=str, default=None,
                        help="Shared ConMem storage directory. Default: <project_root>/conmem_shared_storage")
    parser.add_argument("--task_indices_file", type=str, default=None,
                        help="JSON list of task positions to evaluate after loading the requested split/sample")
    parser.add_argument("--disable_memory_retrieval", action="store_true",
                        help="Disable memory retrieval/injection while keeping the host and task subset unchanged")
    parser.add_argument("--memory_backend", type=str, default="conmem",
                        choices=["conmem", "reme", "simplemem"],
                        help="Memory backend for MAS modes (default: conmem)")
    parser.add_argument("--reme_url", type=str, default="http://127.0.0.1:8003/",
                        help="ReMe HTTP service URL")
    parser.add_argument("--reme_workspace", type=str, default=None,
                        help="ReMe workspace id. Default: reme:<benchmark>:<mas_type>:<model>")
    parser.add_argument("--reme_top_k", type=int, default=5,
                        help="Number of ReMe task memories to retrieve")
    parser.add_argument("--reme_timeout", type=float, default=120.0,
                        help="ReMe HTTP request timeout in seconds")
    parser.add_argument("--reme_trajectory_dir", type=str, default=None,
                        help="Directory for raw trajectories sent to ReMe. Default: <storage_dir>/<benchmark>/reme_trajectories")
    parser.add_argument("--simplemem_storage_dir", type=str, default=None,
                        help="SimpleMem JSON memory store directory. Default: <storage_dir>/<benchmark>/simplemem_memory")
    parser.add_argument("--simplemem_top_k", type=int, default=5,
                        help="Number of SimpleMem memories to retrieve")
    parser.add_argument("--simplemem_embedding_backend", type=str, default="api",
                        choices=["auto", "api", "sentence_transformers", "lexical"],
                        help="SimpleMem semantic retrieval backend. Default: api")
    parser.add_argument("--simplemem_embedding_model", type=str, default=None,
                        help="SimpleMem embedding model. Default: EMBED_MODEL")
    parser.add_argument("--simplemem_embedding_api_base", type=str, default=None,
                        help="OpenAI-compatible embedding API base for SimpleMem. Default: EMBED_BASE_URL")
    parser.add_argument("--simplemem_embedding_api_key", type=str, default=None,
                        help="Embedding API key for SimpleMem. Default: EMBED_API_KEY/OPENAI_API_KEY/EMPTY")
    parser.add_argument("--simplemem_embedding_timeout", type=float, default=60.0,
                        help="SimpleMem embedding request timeout in seconds")
    parser.add_argument("--simplemem_trajectory_dir", type=str, default=None,
                        help="Directory for raw trajectories saved by SimpleMem. Default: <storage_dir>/<benchmark>/simplemem_trajectories")
    parser.add_argument("--seed_memory_from_storage_dir", type=str, default=None,
                        help="Copy this ConMem storage DB into --memory_storage_dir before the run")
    parser.add_argument("--overwrite_seed_memory", action="store_true",
                        help="Overwrite an existing --memory_storage_dir/conmem.db when seeding memory")
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--read_only_memory", action="store_true",
                        help="Read from existing memory storage without writing new cards")
    parser.add_argument("--max_turns", type=int, default=None,
                        help="Optional override for dynamic benchmark max turns")
    parser.add_argument("--timeout_seconds", type=float, default=None,
                        help="Optional per-task wall-clock timeout for dynamic benchmarks")

    args = parser.parse_args()

    benchmarks = ["kodcode", "triviaqa", "popqa", "pddl"] if args.benchmark == "all" else [args.benchmark]

    all_summaries = []
    for bench in benchmarks:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Benchmark: {bench.upper()} | MAS: {args.mas_type.upper()}")
        logger.info(f"{'#'*60}")

        # Reset stats for each benchmark.
        from utils.stats import stats
        stats.reset()

        try:
            summary = run_single_benchmark(bench, args)
            all_summaries.append(summary)
        except Exception as e:
            logger.error(f"[{bench}] Failed: {e}", exc_info=True)
            all_summaries.append({"benchmark": bench, "error": str(e)})

    # Summary
    if len(benchmarks) > 1:
        logger.info(f"\n{'#'*60}")
        logger.info("# OVERALL SUMMARY")
        logger.info(f"{'#'*60}\n")
        for s in all_summaries:
            if "error" in s:
                logger.info(f"  {s['benchmark']:12s}  ERROR: {s['error']}")
            else:
                logger.info(
                    f"  {s['benchmark']:12s}  pass_rate={s['pass_rate']:.2%}  "
                    f"tasks={s['num_tasks']}  time={s['elapsed_seconds']}s  "
                    f"cards={s['memory_cards']}")

        overall_file = os.path.join(args.storage_dir, "overall_results.json")
        with open(overall_file, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"\nOverall results saved to {overall_file}")


if __name__ == "__main__":
    main()
