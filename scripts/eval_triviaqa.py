#!/usr/bin/env python3
"""
ConMem + TriviaQA evaluation script for multi-turn interaction.

TriviaQA is a DynamicEnv: the agent can issue multiple <search> calls before
returning a final <answer>.
Interaction flow:
  Turn 1: LLM generation -> env.step() -> <search> returns results -> continue
  Turn 2: LLM sees search results -> env.step() -> <answer> ends the episode
  ...until <answer> or max_turns

Usage:
    python scripts/eval_triviaqa.py --mas_type single --num_tasks 20
    python scripts/eval_triviaqa.py --mas_type macnet --num_tasks 100 --max_turns 5
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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

def _import_conmem_module():
    from mas_core.memory.backbone.conmem.conmem_module import ConMemModule
    from mas_core.memory.backbone.conmem.config import ConMemConfig
    return ConMemModule, ConMemConfig

def _import_conmem_centralized_memory():
    from mas_core.memory.backbone.conmem.centralized_adapter import ConMemCentralizedMemory
    from mas_core.memory.backbone.conmem.config import ConMemConfig
    return ConMemCentralizedMemory, ConMemConfig


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
    raise ValueError(f"Unsupported MAS type: {mas_type}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_triviaqa")

TRIVIAQA_SYSTEM_PROMPT = """\
Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""


def _cap_observation(observation: str, max_obs_length: int | None) -> str:
    if not observation:
        return ""
    if not max_obs_length or max_obs_length <= 0 or len(observation) <= max_obs_length:
        return observation
    clipped = observation[:max_obs_length].rsplit(" ", 1)[0].strip()
    return (clipped or observation[:max_obs_length]).rstrip() + "..."


def _build_single_turn_prompt(memory_ctx: str, user_turns: list[str]) -> str:
    parts = []
    if memory_ctx:
        parts.append(memory_ctx)
    parts.extend(turn for turn in user_turns if turn)
    return "\n\n".join(parts)


# ============================================================
# Multi-turn interaction loop.
# ============================================================

def run_multiturn_single(
    llm,
    env,
    system_prompt,
    question,
    memory_ctx,
    max_turns,
    max_obs_length=None,
    conmem=None,
    task_id=None,
    agent_role="executor",
    task_description=None,
):
    """Run multi-turn single-agent interaction between an LLM and TriviaQAEnv."""
    user_turns = [question]
    all_steps = []
    response = ""
    effective_task_description = task_description or question

    for turn in range(max_turns):
        full_user = _build_single_turn_prompt(memory_ctx, user_turns)

        response = llm.chat(system_prompt, full_user, temperature=0.0, max_tokens=2048)

        # Execute the environment step.
        observation, reward, done = env.step(response)
        all_steps.append({
            "step_index": turn + 1, "agent": "executor",
            "input": full_user, "output": response,
            "tool_calls": "", "feedback": observation or "",
        })

        logger.info(f"    Turn {turn+1}: done={done}, reward={reward:.1f}")

        if done:
            break

        if conmem is not None:
            memory_ctx = conmem.on_agent_step(
                task_description=effective_task_description,
                agent_role=agent_role,
                task_id=task_id,
                agent_message=response,
                observation=observation or "",
            )

        # Feed the observation into the next user turn.
        if observation:
            prompt_observation = _cap_observation(observation, max_obs_length)
            user_turns.append(f"<information>{prompt_observation}</information>")

    final_reward = env.feedback()
    return final_reward, response, all_steps


def run_multiturn_mas(mas, env, system_prompt, question, ground_truth, generation_config, max_turns, action_resolver=None, max_obs_length=None):
    """Run one MAS generation call.

    The actor/user proxy performs iterative <search> -> <information> loops
    inside mas.generate(); the summarizer/user proxy emits the final <answer>.
    The outer loop is retained only for API compatibility.
    """
    del max_turns  # Kept for run_benchmarks.py API compatibility; no longer used.
    task_context = question
    all_steps = []
    agent_roles = [agent.role for agent in mas.agents_list] if hasattr(mas, 'agents_list') else []

    try:
        graphs = mas.generate([system_prompt], [task_context], generation_config, action_resolver=action_resolver)
        msg_graph = graphs[0]
        response = msg_graph.action or ""
    except Exception as e:
        logger.warning(f"    MAS generate failed: {e}")
        response = ""
        msg_graph = None

    observation, reward, done = env.step(response)

    if msg_graph and msg_graph.mas_message_graph:
        for agent_role in agent_roles:
            if agent_role in msg_graph.mas_message_graph:
                node_data = msg_graph.mas_message_graph.nodes[agent_role]
                msg_node = node_data.get("message")
                if msg_node:
                    system_part = getattr(msg_node, "formatted_system_prompt", None) or msg_node.system_prompt_template or ""
                    user_part = getattr(msg_node, "formatted_user_prompt", None) or ""
                    full_input = f"{system_part}\n\n{user_part}" if system_part else user_part
                    agent_response = msg_node.response if msg_node.response else ""

                    is_answer_agent = "actor" in agent_role or "summarizer" in agent_role or "user proxy" in agent_role
                    if is_answer_agent:
                        import re
                        matches = re.findall(r"<answer>(.*?)</answer>", agent_response, re.DOTALL)
                        extracted_answer = matches[-1].strip() if matches else ""

                        answer_correct = False
                        if extracted_answer and ground_truth:
                            answer_correct = env._check_answer(extracted_answer, ground_truth)

                        agent_reward = 0.0
                        if matches:
                            agent_reward = 1.0 if answer_correct else 0.0

                        feedback_obj = {
                            "observation": observation or "",
                            "reward": agent_reward,
                            "answer_correct": answer_correct,
                            "extracted_answer": extracted_answer,
                            "done": done if matches else False,
                        }
                    else:
                        feedback_obj = {
                            "observation": None,
                            "reward": None,
                            "answer_correct": None,
                        }

                    all_steps.append({
                        "step_index": len(all_steps) + 1,
                        "turn": 1,
                        "agent": agent_role,
                        "input": full_input,
                        "output": agent_response,
                        "tool_calls": "",
                        "feedback": feedback_obj,
                    })
    else:
        all_steps.append({
            "step_index": len(all_steps) + 1,
            "agent": "mas",
            "input": task_context,
            "output": response,
            "tool_calls": "",
            "feedback": {
                "observation": observation or "",
                "reward": reward,
                "done": done,
            },
        })

    logger.info(f"    done={done}, reward={reward:.1f}")
    final_reward = env.feedback()
    return final_reward, response, all_steps


# ============================================================
# Main evaluation.
# ============================================================

def run_evaluation(args):
    from datasets import load_dataset
    from data.triviaqa.env import TriviaQAEnv
    from utils.config_loader import load_benchmark_config

    # ---- Load config file ----
    if args.config:
        config_path = args.config
    else:
        config_path = None
    
    bench_config = load_benchmark_config('triviaqa', config_path)
    logger.info(f"Loaded config: max_new_tokens={bench_config.generation.max_new_tokens}, "
                f"temperature={bench_config.generation.temperature}, "
                f"max_turns={bench_config.interaction.max_turns}")

    # Use the config model_name when the CLI does not specify one.
    if args.model_name is None:
        args.model_name = bench_config.model.llm_name_or_path

    is_mas_mode = args.mas_type in ("camel", "macnet", "autogen")
    memory_backend = args.memory_backend.lower()
    if memory_backend in ("reme", "simplemem") and not is_mas_mode:
        raise ValueError(f"{memory_backend} memory backend currently supports MAS modes only: camel/macnet/autogen")
    memory_label = {
        "conmem": "ConMem",
        "reme": "ReMe",
        "simplemem": "SimpleMem",
    }.get(memory_backend, memory_backend)
    env_path = os.path.join(PROJECT_ROOT, ".env")
    storage_dir = args.storage_dir
    memory_storage_dir = _resolve_memory_storage_dir(args.memory_storage_dir)

    baseline_mas = None
    env_kwargs = {}
    reme_workspace_id = None
    simplemem_storage_dir = None

    if is_mas_mode:
        ConMemCentralizedMemory, ConMemConfig = _import_conmem_centralized_memory()
        config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
        if memory_backend == "reme":
            ReMeCentralizedMemory = _import_reme_centralized_memory()
            workspace_id = args.reme_workspace or f"reme:triviaqa:{args.mas_type}:{args.model_name}"
            reme_workspace_id = workspace_id
            centralized_memory = ReMeCentralizedMemory(
                base_url=args.reme_url,
                workspace_id=workspace_id,
                task_domain="triviaqa",
                top_k=args.reme_top_k,
                timeout=args.reme_timeout,
                read_only=args.read_only_memory,
                trajectory_dir=args.reme_trajectory_dir or os.path.join(storage_dir, "reme_trajectories"),
            )
            conmem = centralized_memory
            logger.info(f"Using ReMe workspace: {workspace_id} ({args.reme_url})")
        elif memory_backend == "simplemem":
            SimpleMemCentralizedMemory = _import_simplemem_centralized_memory()
            simplemem_storage_dir = args.simplemem_storage_dir or os.path.join(storage_dir, "simplemem_memory")
            centralized_memory = SimpleMemCentralizedMemory(
                storage_dir=simplemem_storage_dir,
                task_domain="triviaqa",
                top_k=args.simplemem_top_k,
                embedding_backend=args.simplemem_embedding_backend,
                embedding_model=args.simplemem_embedding_model,
                embedding_api_base=args.simplemem_embedding_api_base,
                embedding_api_key=args.simplemem_embedding_api_key,
                embedding_timeout=args.simplemem_embedding_timeout,
                read_only=args.read_only_memory,
                trajectory_dir=args.simplemem_trajectory_dir or os.path.join(storage_dir, "simplemem_trajectories"),
            )
            conmem = centralized_memory
            logger.info(f"Using SimpleMem memory store: {simplemem_storage_dir}")
        else:
            centralized_memory = ConMemCentralizedMemory(config, memory_storage_dir, task_domain="triviaqa")
            conmem = centralized_memory.conmem
        env_kwargs.update(config.factual_qa_retriever_kwargs_for_domain("triviaqa"))
        if args.search_url:
            env_kwargs["search_url"] = args.search_url
        MASClass = _import_mas_class(args.mas_type)
        mas = MASClass(llm_name_or_path=args.model_name, centralized_memory=centralized_memory,
                       share_llm=True, task_domain="triviaqa", api_base=args.api_base, model_name=args.model_name)
        generation_config = {
            "max_new_tokens": bench_config.generation.max_new_tokens,
            "temperature": bench_config.generation.temperature,
            "top_p": 0.95,
            "do_sample": bench_config.generation.temperature > 0,
        }
        if args.run_baseline:
            baseline_mas = MASClass(
                llm_name_or_path=args.model_name,
                centralized_memory=None,
                share_llm=True,
                task_domain="triviaqa",
                api_base=args.api_base,
                model_name=args.model_name,
            )
        logger.info(f"{args.mas_type.upper()} MAS initialized with {len(mas.agents_list)} agents")
    else:
        ConMemModule, ConMemConfig = _import_conmem_module()
        config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
        conmem = ConMemModule(config, memory_storage_dir, task_domain="triviaqa")
        conmem.set_runtime_context(model_name=args.model_name, mas_architecture="single")
        env_kwargs.update(config.factual_qa_retriever_kwargs_for_domain("triviaqa"))
        if args.search_url:
            env_kwargs["search_url"] = args.search_url

    if memory_backend == "conmem":
        logger.info(f"ConMem shared storage: {memory_storage_dir}")
    elif memory_backend == "simplemem":
        logger.info(f"SimpleMem memory store: {simplemem_storage_dir}")
    else:
        logger.info(f"ReMe HTTP memory: {args.reme_url}")
    logger.info("Loading TriviaQA dataset...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
    test_ds = ds["validation"].select(range(1000, len(ds["validation"])))
    end_idx = min(args.start_from + args.num_tasks, len(test_ds))
    tasks = test_ds.select(range(args.start_from, end_idx))
    logger.info(f"Evaluating tasks [{args.start_from}, {end_idx}), max_turns={args.max_turns}")

    results, baseline_results = [], []
    t_start = time.time()

    for i, task in enumerate(tasks):
        task_idx = args.start_from + i
        question = task["question"].strip()
        ground_truth = task["answer"]["normalized_aliases"]
        task_id = f"triviaqa_{task_idx}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Task {task_idx} ({i+1}/{len(tasks)}): {question[:80]}...")

        # Baseline without memory.
        if args.run_baseline and is_mas_mode:
            env_base = TriviaQAEnv(env_kwargs)
            env_base.set_env({"prompt": question, "answer": ground_truth})
            score_base, _, _ = run_multiturn_mas(
                baseline_mas, env_base, TRIVIAQA_SYSTEM_PROMPT, question, ground_truth,
                generation_config, args.max_turns, action_resolver=env_base.explorer.batch_search,
                max_obs_length=bench_config.interaction.max_obs_length,
            )
            baseline_results.append(score_base)
            logger.info(f"  [Baseline] score={score_base:.1f}")
        elif args.run_baseline:
            env_base = TriviaQAEnv(env_kwargs)
            env_base.set_env({"prompt": question, "answer": ground_truth})
            resp_base = conmem.llm.chat(TRIVIAQA_SYSTEM_PROMPT, question, temperature=0.0, max_tokens=2048)
            env_base.step(resp_base)
            score_base = env_base.feedback()
            baseline_results.append(score_base)
            logger.info(f"  [Baseline] score={score_base:.1f}")

        # Memory-augmented multi-turn interaction.
        env = TriviaQAEnv(env_kwargs)
        env.set_env({"prompt": question, "answer": ground_truth})

        if is_mas_mode:
            score, response, steps = run_multiturn_mas(
                mas, env, TRIVIAQA_SYSTEM_PROMPT, question, ground_truth, generation_config, args.max_turns,
                action_resolver=env.explorer.batch_search,
                max_obs_length=bench_config.interaction.max_obs_length,
            )
        else:
            memory_ctx = conmem.on_task_start(task_description=question, agent_role="executor", task_id=task_id)
            score, response, steps = run_multiturn_single(
                conmem.llm, env, TRIVIAQA_SYSTEM_PROMPT, question, memory_ctx, args.max_turns,
                max_obs_length=bench_config.interaction.max_obs_length,
                conmem=conmem,
                task_id=task_id,
                agent_role="executor",
                task_description=question,
            )

        results.append(score)
        feedback = f"score={score:.1f} (turns={len(steps)}, gt={ground_truth[0] if ground_truth else 'N/A'})"
        logger.info(f"  [{args.mas_type.upper()}+{memory_label}] {feedback}")

        # Store trajectory in memory unless read-only mode is enabled.
        if not getattr(args, 'read_only_memory', False):
            outcome = "success" if score >= 1.0 else ("partial" if score > 0 else "failure")
            traj_data = {"task_description": question, "outcome": outcome, "steps": steps}
            if is_mas_mode:
                centralized_memory.add_memory(trajectory=traj_data, task_id=task_id, task_description=question, outcome=outcome)
            else:
                conmem.on_task_complete(task_id=task_id, task_description=question, outcome=outcome, trajectory=traj_data)
        else:
            logger.info("  [Read-Only Mode] Skipping memory storage")

        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            logger.info(f"\n--- Progress: {i+1}/{len(tasks)} pass_rate={sum(results)/len(results):.2%} ---")

    elapsed = time.time() - t_start
    pass_rate = sum(results) / len(results) if results else 0
    from utils.stats import stats
    api = stats.to_dict()
    reme_usage_stats = centralized_memory.get_usage_stats() if memory_backend == "reme" and is_mas_mode else None
    reme_trajectory_files = centralized_memory.get_saved_trajectory_files() if memory_backend == "reme" and is_mas_mode else None
    simplemem_usage_stats = centralized_memory.get_usage_stats() if memory_backend == "simplemem" and is_mas_mode else None
    simplemem_trajectory_files = centralized_memory.get_saved_trajectory_files() if memory_backend == "simplemem" and is_mas_mode else None

    logger.info(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    logger.info(f"Mode: {args.mas_type.upper()} | Pass rate: {pass_rate:.2%} | Time: {elapsed:.1f}s | Cards: {conmem.storage.count_active_cards()}")
    if baseline_results:
        logger.info(f"Baseline: {sum(baseline_results)/len(baseline_results):.2%}")
    logger.info(f"\n{stats.summary()}")
    if reme_usage_stats:
        logger.info("ReMe service usage: %s", reme_usage_stats)
    if reme_trajectory_files:
        logger.info("Saved ReMe trajectories: %s", len(reme_trajectory_files))
    if simplemem_usage_stats:
        logger.info("SimpleMem usage: %s", simplemem_usage_stats)
    if simplemem_trajectory_files:
        logger.info("Saved SimpleMem trajectories: %s", len(simplemem_trajectory_files))

    os.makedirs(storage_dir, exist_ok=True)
    with open(os.path.join(storage_dir, "eval_results.json"), "w") as f:
        json.dump({
            "benchmark": "triviaqa", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"mas_type": args.mas_type, "model_name": args.model_name,
                "memory_backend": memory_backend,
                "reme_url": args.reme_url if memory_backend == "reme" else None,
                "reme_workspace": reme_workspace_id if memory_backend == "reme" else None,
                "simplemem_storage_dir": simplemem_storage_dir if memory_backend == "simplemem" else None,
                "simplemem_top_k": args.simplemem_top_k if memory_backend == "simplemem" else None,
                "simplemem_embedding_backend": args.simplemem_embedding_backend if memory_backend == "simplemem" else None,
                "simplemem_embedding_model": args.simplemem_embedding_model if memory_backend == "simplemem" else None,
                "simplemem_embedding_api_base": args.simplemem_embedding_api_base if memory_backend == "simplemem" else None,
                "config_model": bench_config.model.llm_name_or_path,
                "max_new_tokens": bench_config.generation.max_new_tokens,
                "temperature": bench_config.generation.temperature,
                "max_turns": bench_config.interaction.max_turns,
                "api_base": args.api_base,
                "num_tasks": len(results), "start_from": args.start_from},
            "results": {"pass_rate": pass_rate, "correct": int(sum(r >= 1.0 for r in results)), "total": len(results), "scores": results},
            "cost": {"elapsed_seconds": round(elapsed, 1), "total_tokens": api["total_tokens"],
                "prompt_tokens": api["total_prompt_tokens"], "completion_tokens": api["total_completion_tokens"],
                "total_api_calls": api["total_calls"], "total_failures": api["total_failures"],
                "by_source": api["by_source"], "reme_service": reme_usage_stats,
                "simplemem_service": simplemem_usage_stats},
            "reme_trajectory_files": reme_trajectory_files,
            "simplemem_trajectory_files": simplemem_trajectory_files,
            "memory": {"active_cards": conmem.storage.count_active_cards(), "current_round": conmem.storage.get_current_round()},
            **({"baseline": {"pass_rate": sum(baseline_results)/len(baseline_results), "scores": baseline_results}} if baseline_results else {}),
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {os.path.join(storage_dir, 'eval_results.json')}")


def main():
    parser = argparse.ArgumentParser(description="ConMem + TriviaQA Evaluation (Multi-turn)")
    parser.add_argument("--read_only_memory", action="store_true",
                        help="Read-only memory mode: use existing memories without storing new ones.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (default: configs/conmem/triviaqa.yaml).")
    parser.add_argument("--mas_type", type=str, default="single", choices=["single", "camel", "macnet", "autogen"])
    parser.add_argument("--api_base", type=str, default="http://localhost:8100/v1")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve",
                        help="Retrieval service URL; avoid conflicts with local LLM API ports")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (default: value from config).")
    parser.add_argument("--num_tasks", type=int, default=20)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--storage_dir", type=str, default="./conmem_storage",
                        help="Result output directory.")
    parser.add_argument("--memory_storage_dir", type=str, default=None,
                        help="Shared ConMem storage directory. Default: <project_root>/conmem_shared_storage.")
    parser.add_argument("--memory_backend", type=str, default="conmem",
                        choices=["conmem", "reme", "simplemem"],
                        help="Memory backend: conmem, reme, or simplemem (default: conmem).")
    parser.add_argument("--reme_url", type=str, default="http://127.0.0.1:8003/",
                        help="ReMe HTTP service URL (default: http://127.0.0.1:8003/).")
    parser.add_argument("--reme_workspace", type=str, default=None,
                        help="ReMe workspace_id. Default: reme:triviaqa:<mas_type>:<model_name>.")
    parser.add_argument("--reme_top_k", type=int, default=5,
                        help="ReMe retrieval top_k (default: 5).")
    parser.add_argument("--reme_timeout", type=float, default=120.0,
                        help="ReMe HTTP request timeout in seconds (default: 120).")
    parser.add_argument("--reme_trajectory_dir", type=str, default=None,
                        help="Directory for raw trajectories sent to ReMe. Default: <storage_dir>/reme_trajectories.")
    parser.add_argument("--simplemem_storage_dir", type=str, default=None,
                        help="SimpleMem JSON memory-store directory. Default: <storage_dir>/simplemem_memory.")
    parser.add_argument("--simplemem_top_k", type=int, default=5,
                        help="SimpleMem retrieval top_k (default: 5).")
    parser.add_argument("--simplemem_embedding_backend", type=str, default="api",
                        choices=["auto", "api", "sentence_transformers", "lexical"],
                        help="SimpleMem semantic retrieval backend (default: api).")
    parser.add_argument("--simplemem_embedding_model", type=str, default=None,
                        help="SimpleMem embedding model. Defaults to EMBED_MODEL.")
    parser.add_argument("--simplemem_embedding_api_base", type=str, default=None,
                        help="SimpleMem OpenAI-compatible embedding API base. Defaults to EMBED_BASE_URL.")
    parser.add_argument("--simplemem_embedding_api_key", type=str, default=None,
                        help="SimpleMem embedding API key. Defaults to EMBED_API_KEY/OPENAI_API_KEY/EMPTY.")
    parser.add_argument("--simplemem_embedding_timeout", type=float, default=60.0,
                        help="SimpleMem embedding request timeout in seconds (default: 60).")
    parser.add_argument("--simplemem_trajectory_dir", type=str, default=None,
                        help="Directory for raw SimpleMem trajectories. Default: <storage_dir>/simplemem_trajectories.")
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_turns", type=int, default=5, help="Max interaction turns (default: 5)")
    args = parser.parse_args()
    _setup_imports()
    run_evaluation(args)

if __name__ == "__main__":
    main()
