#!/usr/bin/env python3
"""
ConMem + PopQA evaluation script for multi-turn interaction.

PopQA is a DynamicEnv: the agent can issue multiple <search> calls before
returning a final <answer>. The interaction flow matches TriviaQA.

Usage:
    python scripts/eval_popqa.py --mas_type single --num_tasks 20
    python scripts/eval_popqa.py --mas_type macnet --num_tasks 100 --max_turns 5
    python scripts/eval_popqa.py --mas_type macnet --search_url http://127.0.0.1:8000/retrieve
"""
import argparse
import ast
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
logger = logging.getLogger("eval_popqa")

POPQA_SYSTEM_PROMPT = """\
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
    """Run multi-turn single-agent interaction between an LLM and PopQAEnv."""
    user_turns = [question]
    all_steps = []
    response = ""
    effective_task_description = task_description or question

    for turn in range(max_turns):
        full_user = _build_single_turn_prompt(memory_ctx, user_turns)
        response = llm.chat(system_prompt, full_user, temperature=0.0, max_tokens=2048)

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
        if observation:
            prompt_observation = _cap_observation(observation, max_obs_length)
            user_turns.append(f"<information>{prompt_observation}</information>")

    return env.feedback(), response, all_steps


def run_multiturn_mas(mas, env, system_prompt, question, ground_truth, generation_config, max_turns, action_resolver=None, max_obs_length=None):
    """Run one MAS generation call.

    The actor/user proxy performs iterative <search> -> <information> loops
    inside mas.generate(); the summarizer/user proxy emits the final <answer>.
    The trajectory schema matches TriviaQA: each MAS role emits one step with
    step_index, turn, agent, input, output, tool_calls, and feedback. The
    answer-producing role feedback includes observation, reward, answer_correct,
    extracted_answer, and done.
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
                            for gt in ground_truth:
                                if gt.lower() in extracted_answer.lower():
                                    answer_correct = True
                                    break

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


def run_evaluation(args):
    from datasets import load_dataset
    from data.popqa.env import PopQAEnv
    from data.popqa.sampling import select_popqa_tasks
    from utils.config_loader import load_benchmark_config

    # ---- Load config file ----
    if args.config:
        config_path = args.config
    else:
        config_path = None
    
    bench_config = load_benchmark_config('popqa', config_path)
    logger.info(f"Loaded config: max_new_tokens={bench_config.generation.max_new_tokens}, "
                f"temperature={bench_config.generation.temperature}, "
                f"max_turns={bench_config.interaction.max_turns}")

    # Use the config model_name when the CLI does not specify one.
    if args.model_name is None:
        args.model_name = bench_config.model.llm_name_or_path

    is_mas_mode = args.mas_type in ("camel", "macnet", "autogen")
    env_path = os.path.join(PROJECT_ROOT, ".env")
    storage_dir = args.storage_dir
    memory_storage_dir = _resolve_memory_storage_dir(args.memory_storage_dir)

    baseline_mas = None
    env_kwargs = {}

    if is_mas_mode:
        ConMemCentralizedMemory, ConMemConfig = _import_conmem_centralized_memory()
        config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
        centralized_memory = ConMemCentralizedMemory(config, memory_storage_dir, task_domain="popqa")
        conmem = centralized_memory.conmem
        env_kwargs.update(config.factual_qa_retriever_kwargs_for_domain("popqa"))
        if args.search_url:
            env_kwargs["search_url"] = args.search_url
        MASClass = _import_mas_class(args.mas_type)
        mas = MASClass(llm_name_or_path=args.model_name, centralized_memory=centralized_memory,
                       share_llm=True, task_domain="popqa", api_base=args.api_base, model_name=args.model_name)
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
                task_domain="popqa",
                api_base=args.api_base,
                model_name=args.model_name,
            )
        logger.info(f"{args.mas_type.upper()} MAS initialized with {len(mas.agents_list)} agents")
    else:
        ConMemModule, ConMemConfig = _import_conmem_module()
        config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
        conmem = ConMemModule(config, memory_storage_dir, task_domain="popqa")
        conmem.set_runtime_context(model_name=args.model_name, mas_architecture="single")
        env_kwargs.update(config.factual_qa_retriever_kwargs_for_domain("popqa"))
        if args.search_url:
            env_kwargs["search_url"] = args.search_url

    logger.info(f"ConMem shared storage: {memory_storage_dir}")
    logger.info("Loading PopQA dataset...")
    ds = load_dataset("akariasai/PopQA")
    test_ds = ds["test"].select(range(7000, len(ds["test"])))
    tasks, selected_indices, sampling_info = select_popqa_tasks(
        test_ds,
        num_tasks=args.num_tasks,
        start_from=args.start_from,
        sampling=args.popqa_sampling,
        seed=args.popqa_sample_seed,
    )
    logger.info(
        f"Evaluating {len(tasks)} tasks, sampling={args.popqa_sampling}, "
        f"start_from={args.start_from}, max_turns={args.max_turns}"
    )

    results, baseline_results = [], []
    t_start = time.time()

    for i, task in enumerate(tasks):
        task_idx = 7000 + selected_indices[i] if i < len(selected_indices) else args.start_from + i
        question = task["question"].strip()
        ground_truth = ast.literal_eval(task["possible_answers"])
        task_id = f"popqa_{task_idx}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Task {task_idx} ({i+1}/{len(tasks)}): {question[:80]}...")
        env_task_config = {
            "prompt": question,
            "answer": ground_truth,
        }

        if args.run_baseline and is_mas_mode:
            env_base = PopQAEnv(env_kwargs)
            env_base.set_env(env_task_config)
            score_base, _, _ = run_multiturn_mas(
                baseline_mas, env_base, POPQA_SYSTEM_PROMPT, question,
                generation_config, args.max_turns, action_resolver=env_base.explorer.batch_search,
                max_obs_length=bench_config.interaction.max_obs_length,
            )
            baseline_results.append(score_base)
            logger.info(f"  [Baseline] score={score_base:.1f}")
        elif args.run_baseline:
            env_base = PopQAEnv(env_kwargs)
            env_base.set_env(env_task_config)
            resp = conmem.llm.chat(POPQA_SYSTEM_PROMPT, question, temperature=0.0, max_tokens=2048)
            env_base.step(resp)
            baseline_results.append(env_base.feedback())
            logger.info(f"  [Baseline] score={env_base.feedback():.1f}")

        env = PopQAEnv(env_kwargs)
        env.set_env(env_task_config)

        if is_mas_mode:
            score, response, steps = run_multiturn_mas(
                mas, env, POPQA_SYSTEM_PROMPT, question, generation_config, args.max_turns,
                action_resolver=env.explorer.batch_search,
                max_obs_length=bench_config.interaction.max_obs_length,
            )
        else:
            memory_ctx = conmem.on_task_start(task_description=question, agent_role="executor", task_id=task_id)
            score, response, steps = run_multiturn_single(
                conmem.llm, env, POPQA_SYSTEM_PROMPT, question, memory_ctx, args.max_turns,
                max_obs_length=bench_config.interaction.max_obs_length,
                conmem=conmem,
                task_id=task_id,
                agent_role="executor",
                task_description=question,
            )

        results.append(score)
        feedback = f"score={score:.1f} (turns={len(steps)}, gt={ground_truth[0]})"
        logger.info(f"  [{args.mas_type.upper()}+ConMem] {feedback}")

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

    logger.info(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    logger.info(f"Mode: {args.mas_type.upper()} | Pass rate: {pass_rate:.2%} | Time: {elapsed:.1f}s | Cards: {conmem.storage.count_active_cards()}")
    logger.info(f"\n{stats.summary()}")

    os.makedirs(storage_dir, exist_ok=True)
    with open(os.path.join(storage_dir, "eval_results.json"), "w") as f:
        json.dump({
            "benchmark": "popqa", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"mas_type": args.mas_type, "model_name": args.model_name,
                "config_model": bench_config.model.llm_name_or_path,
                "max_new_tokens": bench_config.generation.max_new_tokens,
                "temperature": bench_config.generation.temperature,
                "max_turns": bench_config.interaction.max_turns,
                "api_base": args.api_base,
                "num_tasks": len(results), "start_from": args.start_from,
                "sampling": sampling_info["sampling"],
                "sample_seed": sampling_info["sample_seed"],
                "selected_indices": sampling_info["selected_indices"]},
            "results": {"pass_rate": pass_rate, "correct": int(sum(r >= 1.0 for r in results)), "total": len(results), "scores": results},
            "cost": {"elapsed_seconds": round(elapsed, 1), "total_tokens": api["total_tokens"],
                "prompt_tokens": api["total_prompt_tokens"], "completion_tokens": api["total_completion_tokens"],
                "total_api_calls": api["total_calls"], "total_failures": api["total_failures"], "by_source": api["by_source"]},
            "memory": {"active_cards": conmem.storage.count_active_cards(), "current_round": conmem.storage.get_current_round()},
        }, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="ConMem + PopQA Evaluation (Multi-turn)")
    parser.add_argument("--read_only_memory", action="store_true",
                        help="Read-only memory mode: use existing memories without storing new ones.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (default: configs/conmem/popqa.yaml).")
    parser.add_argument("--mas_type", type=str, default="single", choices=["single", "camel", "macnet", "autogen"])
    parser.add_argument("--api_base", type=str, default="http://localhost:8100/v1")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve",
                        help="Retrieval service URL; the official Search-R1 retriever defaults to port 8000")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (default: value from config).")
    parser.add_argument("--num_tasks", type=int, default=20)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--popqa_sampling", type=str, default="ordered",
                        choices=["ordered", "stratified_by_prop"],
                        help="PopQA task selection strategy (default: ordered)")
    parser.add_argument("--popqa_sample_seed", type=int, default=42,
                        help="Seed for PopQA stratified sampling")
    parser.add_argument("--storage_dir", type=str, default="./conmem_storage",
                        help="Result output directory.")
    parser.add_argument("--memory_storage_dir", type=str, default=None,
                        help="Shared ConMem storage directory. Default: <project_root>/conmem_shared_storage.")
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_turns", type=int, default=5)
    args = parser.parse_args()
    _setup_imports()
    run_evaluation(args)

if __name__ == "__main__":
    main()
