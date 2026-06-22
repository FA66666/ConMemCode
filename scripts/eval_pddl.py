#!/usr/bin/env python3
"""
ConMem + PDDL evaluation script for multi-turn interaction.

PDDL is a DynamicEnv: the agent sends one action, pddlgym executes it, and the
agent uses the new observation to select the next action.
Interaction flow:
  Turn 1: env.set_env() returns the initial observation and goal -> LLM action -> env.step()
  Turn 2: env returns the new observation -> LLM next action -> env.step()
  ...until the goal is reached, timeout_seconds expires, or max_turns is reached.

Usage:
    python scripts/eval_pddl.py --mas_type single --num_tasks 20
    python scripts/eval_pddl.py --mas_type macnet --num_tasks 60
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
logger = logging.getLogger("eval_pddl")

VALID_ACTIONS_PREFIX = "Valid actions are:"
META_ACTIONS = {"check valid actions", "look around"}


def extract_action(response):
    """Extract one action from LLM output, using an <action> tag or plain text."""
    match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else response.strip()


def extract_actions(response):
    """Extract the single action to execute for this turn.

    LatentMem-style PDDL interaction maps one model output to one env.step.
    If the model emits multiple <action> tags, only the first one is executed
    so planning turns stay aligned with environment steps.
    """
    action = extract_action(response)
    return [action] if action else []


def extract_valid_actions_observation(observation):
    """Return the valid action list from a `check valid actions` observation."""
    if not observation:
        return []
    pattern = rf"{re.escape(VALID_ACTIONS_PREFIX)}\s*(.*)"
    match = re.search(pattern, observation, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    raw_actions = match.group(1).strip()
    if not raw_actions:
        return []
    return [part.strip() for part in raw_actions.split(",") if part.strip()]


def format_valid_action_constraint(valid_actions):
    if not valid_actions:
        return ""
    action_lines = "\n".join(f"- {action}" for action in valid_actions)
    return (
        "<valid_action_constraint>\n"
        "The environment lists the currently valid actions for this exact state. "
        "For this turn, you must output exactly one action copied from the list below. "
        "Do not translate it to another action syntax, invent a new action, "
        "or change object names, rooms, hands, or order. Prefer a progress action "
        "over diagnostic actions such as `check valid actions` or `look around`.\n"
        f"{action_lines}\n"
        "</valid_action_constraint>"
    )


def normalize_action_text(action):
    text = extract_action(action or "").lower()
    text = re.sub(r"[`\"']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".")


def is_meta_action(action):
    return normalize_action_text(action) in META_ACTIONS


def filter_decision_actions(valid_actions):
    """Keep the prompt focused on executable progress actions when possible."""
    progress_actions = [action for action in valid_actions if not is_meta_action(action)]
    return progress_actions or valid_actions


def get_current_valid_actions(env):
    """Ask the environment for the current valid action space, if available."""
    action_space_getter = getattr(env, "_get_action_space", None)
    if not callable(action_space_getter):
        return []
    try:
        return filter_decision_actions(sorted(str(action) for action in action_space_getter()))
    except Exception as exc:
        logger.debug("Unable to get PDDL action space for prompt constraint: %s", exc)
        return []


_TOKEN_STOPWORDS = {
    "a", "an", "and", "arm", "at", "beverage", "from", "glass", "hand",
    "holding", "hub", "in", "ingredient", "ingredients", "level", "nut",
    "of", "on", "the", "to", "with",
}


def _action_tokens(action):
    text = normalize_action_text(action).replace("-", " ")
    return {
        token
        for token in re.findall(r"[a-z]+\d*|l\d+", text)
        if token not in _TOKEN_STOPWORDS
    }


def gripper_action_signature(action):
    """Parse canonical and rendered gripper actions into comparable signatures."""
    text = normalize_action_text(action)
    if text == "check valid actions":
        return ("check",)

    move = re.search(r"\bmove(?:\s+from)?\s+([a-z0-9_-]+)\s+(?:to\s+)?([a-z0-9_-]+)\b", text)
    if move:
        return ("move", move.group(1), move.group(2))

    pick = re.search(
        r"\bpick(?:\s+up)?\s+([a-z0-9_-]+)(?:\s+at)?\s+([a-z0-9_-]+)"
        r"(?:\s+with\s+arm)?\s+([a-z0-9_-]+)\b",
        text,
    )
    if pick:
        return ("pick", pick.group(1), pick.group(2), pick.group(3))

    drop = re.search(
        r"\bdrop\s+([a-z0-9_-]+)(?:\s+at)?\s+([a-z0-9_-]+)"
        r"(?:\s+with\s+arm)?\s+([a-z0-9_-]+)\b",
        text,
    )
    if drop:
        return ("drop", drop.group(1), drop.group(2), drop.group(3))
    return None


def _find_valid_action_match(action, valid_actions):
    normalized = normalize_action_text(action)
    for valid in valid_actions:
        if normalize_action_text(valid) == normalized:
            return valid

    signature = gripper_action_signature(action)
    if signature is not None:
        for valid in valid_actions:
            if gripper_action_signature(valid) == signature:
                return valid
        return None

    model_tokens = _action_tokens(action)
    if not model_tokens:
        return None
    best_valid = None
    best_score = (0, 0.0)
    for valid in valid_actions:
        if is_meta_action(valid):
            continue
        valid_tokens = _action_tokens(valid)
        common = model_tokens & valid_tokens
        if len(common) < 2:
            continue
        coverage = len(common) / max(len(model_tokens), 1)
        score = (len(common), coverage)
        if score > best_score:
            best_score = score
            best_valid = valid
    if best_valid is not None and best_score[1] >= 0.5:
        return best_valid
    return None


def format_goal_progress_context(env):
    """Expose current unsatisfied goal literals derivable from state and goal."""
    unsatisfied = []
    try:
        local_env = getattr(env, "local_env", None)
        if local_env is not None:
            for subgoal in getattr(local_env, "subgoals", []) or []:
                if not local_env._subgoal_satisfied(subgoal):
                    unsatisfied.append(str(subgoal).strip())
        else:
            last_obs = getattr(env, "last_obs", None)
            goal_literals = getattr(env, "goal_literals", None) or []
            if last_obs is not None:
                obs_literals = getattr(last_obs, "literals", set())
                literal_to_text = getattr(env, "_literal_to_text", None)
                for literal in goal_literals:
                    if literal not in obs_literals:
                        if callable(literal_to_text):
                            unsatisfied.append(literal_to_text(literal).strip())
                        else:
                            unsatisfied.append(str(literal).strip())
    except Exception as exc:
        logger.debug("Unable to format PDDL goal progress context: %s", exc)
        return ""

    if not unsatisfied:
        return ""
    goal_lines = "\n".join(f"- {goal}" for goal in unsatisfied[:20])
    return (
        "<goal_progress>\n"
        "Unsatisfied goal conditions in the current state:\n"
        f"{goal_lines}\n"
        "</goal_progress>"
    )


def _extract_gripper_state_location(observation, name):
    if not observation:
        return None
    match = re.search(
        rf"\b{re.escape(name.lower())}\s+is\s+at\s+([a-z0-9_-]+)\b",
        observation.lower(),
    )
    return match.group(1) if match else None


def _repair_gripper_action(action, valid_actions, state_observation):
    signature = gripper_action_signature(action)
    if not signature or signature[0] != "pick":
        return None

    _, obj, requested_room, _ = signature
    actual_room = _extract_gripper_state_location(state_observation, obj)
    robby_room = _extract_gripper_state_location(state_observation, "robby")
    target_room = actual_room or requested_room
    if not robby_room or not target_room or robby_room == target_room:
        return None

    desired_move = ("move", robby_room, target_room)
    for valid in valid_actions:
        if gripper_action_signature(valid) == desired_move:
            return valid
    return None


def guard_action_with_valid_actions(action, valid_actions, state_observation=""):
    """Enforce a pending valid-action list before stepping the environment."""
    if not valid_actions:
        return action, ""

    matched = _find_valid_action_match(action, valid_actions)
    if matched:
        if normalize_action_text(matched) == normalize_action_text(action):
            return matched, ""
        return matched, f"guarded_action: mapped model output to listed valid action `{matched}`"

    repaired = _repair_gripper_action(action, valid_actions, state_observation)
    if repaired:
        return repaired, f"guarded_action: replaced invalid model output with listed valid action `{repaired}`"

    for valid in valid_actions:
        if normalize_action_text(valid) == "check valid actions":
            return valid, f"guarded_action: rejected unlisted model output `{action}`"

    return valid_actions[0], f"guarded_action: rejected unlisted model output `{action}`"


def _deadline_from_timeout(timeout_seconds):
    if timeout_seconds is None:
        return None
    timeout = float(timeout_seconds)
    if timeout <= 0:
        return None
    return time.monotonic() + timeout


def _deadline_expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def _append_timeout_feedback(feedback, timeout_seconds):
    if timeout_seconds is None:
        return feedback
    return f"{feedback} timeout=True budget_seconds={float(timeout_seconds):.1f}"


# ============================================================
# Multi-turn interaction loop.
# ============================================================

def run_multiturn_single(
    llm,
    env,
    system_prompt,
    init_user_prompt,
    memory_ctx,
    max_turns,
    conmem=None,
    task_id=None,
    agent_role="executor",
    task_description=None,
    timeout_seconds=None,
):
    """Run multi-turn single-agent interaction with PDDLEnv.

    Only one action is executed per turn. The trajectory schema matches
    KodCode: each LLM call emits one step with step_index, agent, input,
    output, tool_calls, and a string environment feedback summary.
    """
    conversation = [f"<observation>{init_user_prompt}</observation>"]
    all_steps = []
    effective_task_description = task_description or init_user_prompt
    deadline = _deadline_from_timeout(timeout_seconds)
    pending_valid_actions = []
    latest_state_observation = init_user_prompt

    for turn in range(max_turns):
        if _deadline_expired(deadline):
            logger.info(
                f"    PDDL timeout before turn {turn+1} "
                f"(budget={float(timeout_seconds):.1f}s)"
            )
            break

        current_valid_actions = get_current_valid_actions(env) or pending_valid_actions
        prompt_parts = []
        if memory_ctx:
            prompt_parts.append(memory_ctx)
        prompt_parts.extend(conversation)
        goal_progress_context = format_goal_progress_context(env)
        if goal_progress_context:
            prompt_parts.append(goal_progress_context)
        valid_action_constraint = format_valid_action_constraint(current_valid_actions)
        if valid_action_constraint:
            prompt_parts.append(valid_action_constraint)
        full_user = "\n".join(prompt_parts)

        response = llm.chat(system_prompt, full_user, temperature=0.0, max_tokens=1024)
        if _deadline_expired(deadline):
            feedback_str = _append_timeout_feedback(
                _format_env_feedback([], "", 0.0, env.feedback(), False),
                timeout_seconds,
            )
            all_steps.append({
                "step_index": len(all_steps) + 1,
                "agent": agent_role,
                "input": full_user,
                "output": response,
                "tool_calls": "",
                "feedback": feedback_str,
            })
            break

        actions = extract_actions(response)

        # Execute the single action for this turn and summarize environment feedback in one step.
        done = False
        last_action = ""
        last_observation = ""
        last_reward = 0.0
        last_score = env.feedback()
        executed_actions = []
        guard_notes = []
        for action in actions:
            if _deadline_expired(deadline):
                break
            action_to_execute, guard_note = guard_action_with_valid_actions(
                action,
                current_valid_actions,
                latest_state_observation,
            )
            if guard_note:
                guard_notes.append(guard_note)
                logger.info("    %s", guard_note)
            observation, reward, done = env.step(action_to_execute)
            current_score = env.feedback()
            conversation.append(f"<action>{action_to_execute}</action>")
            conversation.append(f"<observation>{observation}</observation>")
            executed_actions.append(action_to_execute)
            logger.info(
                f"    Turn {turn+1} action='{action_to_execute[:60]}' "
                f"step_reward={reward} score={current_score:.3f} done={done}"
            )
            last_action = action_to_execute
            last_observation = observation
            last_reward = reward
            last_score = current_score
            if done:
                break

        feedback_str = _format_env_feedback(
            executed_actions, last_observation, last_reward, last_score, done
        )
        if guard_notes:
            feedback_str = f"{feedback_str} {'; '.join(guard_notes)}"
        timed_out = _deadline_expired(deadline)
        if timed_out and not done:
            feedback_str = _append_timeout_feedback(feedback_str, timeout_seconds)
        all_steps.append({
            "step_index": len(all_steps) + 1,
            "agent": agent_role,
            "input": full_user,
            "output": response,
            "tool_calls": "",
            "feedback": feedback_str,
        })

        if done:
            break
        if timed_out:
            logger.info(
                f"    PDDL timeout after turn {turn+1} "
                f"(budget={float(timeout_seconds):.1f}s)"
            )
            break

        pending_valid_actions = extract_valid_actions_observation(last_observation)
        if last_observation and not pending_valid_actions and "not valid" not in last_observation.lower():
            latest_state_observation = last_observation
        if conmem is not None:
            # PDDL cases are multi-turn environment interactions. Mid-case
            # refreshes must be retrieval-only; card extraction/admission is
            # deferred to the explicit on_task_complete call after the case.
            memory_ctx = conmem.on_task_start(
                task_description=effective_task_description,
                agent_role=agent_role,
                task_id=task_id,
                interaction_context="\n".join(
                    part for part in (last_action, last_observation) if part
                ).strip(),
            )

    final_reward = env.feedback()
    won = env.won
    return final_reward, won, all_steps


def _format_env_feedback(actions, observation, reward, score, done):
    if not actions:
        return "no action executed"
    joined = " | ".join(a[:80] for a in actions)
    return (
        f"actions=[{joined}] step_reward={reward} score={score:.3f} "
        f"done={done} observation={(observation or '')[:200]}"
    )


def run_multiturn_mas(
    mas,
    env,
    system_prompt,
    init_user_prompt,
    generation_config,
    max_turns,
    timeout_seconds=None,
    memory_task_description=None,
):
    """Run multi-turn MAS interaction with PDDLEnv.

    Each turn calls MAS generation, extracts the final action, and executes one
    env.step. The trajectory schema matches KodCode: each MAS agent node emits
    one step. Only the final agent receives the environment feedback summary;
    other agent feedback fields stay empty. If no MAS graph exists, a single
    agent="mas" fallback step is emitted.
    """
    import networkx as nx

    task_context = init_user_prompt
    all_steps = []
    executed_history = []
    deadline = _deadline_from_timeout(timeout_seconds)
    pending_valid_actions = []
    latest_state_observation = init_user_prompt
    memory_task_description = memory_task_description or init_user_prompt

    for turn in range(max_turns):
        if _deadline_expired(deadline):
            logger.info(
                f"    PDDL timeout before turn {turn+1} "
                f"(budget={float(timeout_seconds):.1f}s)"
            )
            break

        current_valid_actions = get_current_valid_actions(env) or pending_valid_actions
        current_task_context = task_context
        goal_progress_context = format_goal_progress_context(env)
        if goal_progress_context:
            current_task_context = f"{current_task_context}\n\n{goal_progress_context}"
        valid_action_constraint = format_valid_action_constraint(current_valid_actions)
        if valid_action_constraint:
            current_task_context = f"{current_task_context}\n\n{valid_action_constraint}"

        try:
            try:
                graphs = mas.generate(
                    [system_prompt],
                    [current_task_context],
                    generation_config,
                    memory_task_descriptions=[memory_task_description],
                )
            except TypeError as e:
                if "memory_task_descriptions" not in str(e):
                    raise
                graphs = mas.generate(
                    [system_prompt],
                    [current_task_context],
                    generation_config,
                )
            msg_graph = graphs[0]
            response = msg_graph.action or ""
        except Exception as e:
            logger.warning(f"    Turn {turn+1}: MAS generate failed: {e}")
            response = ""
            msg_graph = None

        if _deadline_expired(deadline):
            feedback_str = _append_timeout_feedback(
                _format_env_feedback([], "", 0.0, env.feedback(), False),
                timeout_seconds,
            )
            all_steps.append({
                "step_index": len(all_steps) + 1,
                "agent": "mas",
                "input": current_task_context,
                "output": response,
                "tool_calls": "",
                "feedback": feedback_str,
            })
            break

        actions = extract_actions(response)

        # Execute the single action first, then attach one environment feedback summary.
        done = False
        last_observation = ""
        last_reward = 0.0
        last_score = env.feedback()
        executed_actions = []
        guard_notes = []
        for action in actions:
            if _deadline_expired(deadline):
                break
            action_to_execute, guard_note = guard_action_with_valid_actions(
                action,
                current_valid_actions,
                latest_state_observation,
            )
            if guard_note:
                guard_notes.append(guard_note)
                logger.info("    %s", guard_note)
            observation, reward, done = env.step(action_to_execute)
            current_score = env.feedback()
            last_observation = observation
            last_reward = reward
            last_score = current_score
            executed_actions.append(action_to_execute)
            logger.info(
                f"    Turn {turn+1} action='{action_to_execute[:60]}' "
                f"step_reward={reward} score={current_score:.3f} done={done}"
            )
            if done:
                break
        executed_history.extend(executed_actions)

        feedback_str = _format_env_feedback(
            executed_actions, last_observation, last_reward, last_score, done
        )
        if guard_notes:
            feedback_str = f"{feedback_str} {'; '.join(guard_notes)}"
        timed_out = _deadline_expired(deadline)
        if timed_out and not done:
            feedback_str = _append_timeout_feedback(feedback_str, timeout_seconds)

        # Split one step per agent role; attach feedback only to the final agent.
        if msg_graph and msg_graph.mas_message_graph is not None:
            graph = msg_graph.mas_message_graph
            try:
                topo_order = list(nx.topological_sort(graph))
            except Exception:
                topo_order = list(graph.nodes())
            total = len(topo_order)
            for idx, agent_role in enumerate(topo_order):
                node_data = graph.nodes[agent_role]
                msg_node = node_data.get("message")
                agent_input = (
                    getattr(msg_node, "formatted_user_prompt", None)
                    or (msg_node.user_prompt_template if msg_node else "")
                    or current_task_context
                )
                agent_output = msg_node.response if (msg_node and msg_node.response) else ""
                is_last = idx == total - 1
                all_steps.append({
                    "step_index": len(all_steps) + 1,
                    "agent": str(agent_role),
                    "input": agent_input,
                    "output": agent_output,
                    "tool_calls": "",
                    "feedback": feedback_str if is_last else "",
                })
        else:
            # Fallback: synthesize one aggregate step when no MAS graph is available.
            all_steps.append({
                "step_index": len(all_steps) + 1,
                "agent": "mas",
                "input": current_task_context,
                "output": response,
                "tool_calls": "",
                "feedback": feedback_str,
            })

        if done:
            break
        if timed_out:
            logger.info(
                f"    PDDL timeout after turn {turn+1} "
                f"(budget={float(timeout_seconds):.1f}s)"
            )
            break

        # Next turn: update the task context with the latest observation.
        history_lines = [f"  Action: {act}" for act in executed_history]
        task_context = (
            f"{init_user_prompt}\n\nAction history:\n"
            + ("\n".join(history_lines) if history_lines else "  (none)")
            + f"\n\nCurrent observation: {last_observation}\n\nWhat is your next action?"
        )
        pending_valid_actions = extract_valid_actions_observation(last_observation)
        if last_observation and not pending_valid_actions and "not valid" not in last_observation.lower():
            latest_state_observation = last_observation

    final_reward = env.feedback()
    won = env.won
    return final_reward, won, all_steps


# ============================================================
# Main evaluation.
# ============================================================

def run_evaluation(args):
    from data.pddl.builder import PDDL_TASK_NAMES, get_all_environment_configs
    from data.pddl.env.pddl_env import PDDLEnv
    from utils.config_loader import load_benchmark_config

    # ---- Load config file ----
    if args.config:
        config_path = args.config
    else:
        config_path = None
    
    bench_config = load_benchmark_config('pddl', config_path)
    if args.max_turns is None:
        args.max_turns = bench_config.interaction.max_turns
    timeout_seconds = (
        args.timeout_seconds
        if getattr(args, "timeout_seconds", None) is not None
        else getattr(bench_config.interaction, "timeout_seconds", None)
    )
    logger.info(f"Loaded config: max_new_tokens={bench_config.generation.max_new_tokens}, "
                f"temperature={bench_config.generation.temperature}, "
                f"max_turns={args.max_turns}, timeout_seconds={timeout_seconds}")

    # Use the config model_name when the CLI does not specify one.
    if args.model_name is None:
        args.model_name = bench_config.model.llm_name_or_path

    is_mas_mode = args.mas_type in ("camel", "macnet", "autogen")
    env_path = os.path.join(PROJECT_ROOT, ".env")
    storage_dir = args.storage_dir
    memory_storage_dir = _resolve_memory_storage_dir(args.memory_storage_dir)
    data_path = os.path.join(PROJECT_ROOT, "data", "pddl", "test.jsonl")

    baseline_mas = None

    if is_mas_mode:
        ConMemCentralizedMemory, ConMemConfig = _import_conmem_centralized_memory()
        config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
        centralized_memory = ConMemCentralizedMemory(config, memory_storage_dir, task_domain="pddl")
        conmem = centralized_memory.conmem
        MASClass = _import_mas_class(args.mas_type)
        mas = MASClass(llm_name_or_path=args.model_name, centralized_memory=centralized_memory,
                       share_llm=True, task_domain="pddl", api_base=args.api_base, model_name=args.model_name)
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
                task_domain="pddl",
                api_base=args.api_base,
                model_name=args.model_name,
            )
        logger.info(f"{args.mas_type.upper()} MAS initialized with {len(mas.agents_list)} agents")
    else:
        ConMemModule, ConMemConfig = _import_conmem_module()
        config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
        conmem = ConMemModule(config, memory_storage_dir, task_domain="pddl")
        conmem.set_runtime_context(model_name=args.model_name, mas_architecture="single")

    logger.info(f"ConMem shared storage: {memory_storage_dir}")

    # LatentMem-style PDDL setup: enumerate fixed per-domain problem indices
    # and use the local jsonl only to attach difficulty / goal metadata.
    all_tasks = get_all_environment_configs(PDDL_TASK_NAMES, data_path)
    end = min(args.start_from + args.num_tasks, len(all_tasks))
    tasks = all_tasks[args.start_from:end]
    logger.info(
        f"Evaluating {len(tasks)} PDDL tasks, max_turns={args.max_turns}, "
        f"timeout_seconds={timeout_seconds}"
    )

    results, won_count = [], 0
    baseline_results, baseline_won_count = [], 0
    t_start = time.time()

    for i, task_data in enumerate(tasks):
        task_idx = args.start_from + i
        game_name = (
            task_data.get("game_name")
            or task_data.get("additional_info", {}).get("subtask", "unknown")
        )
        problem_index = task_data.get("problem_index", i)
        difficulty = task_data.get("difficulty", "unknown")
        task_id = f"pddl_{task_idx}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Task {task_idx} ({i+1}/{len(tasks)}): {game_name} [{difficulty}]")

        # Initialize PDDLEnv with the official system_prompt and user_prompt from env.set_env.
        env = PDDLEnv({})
        try:
            system_prompt, init_user_prompt = env.set_env({
                "game_name": game_name,
                "problem_index": problem_index,
                "goal": task_data.get("goal"),
                "subgoals": task_data.get("subgoals", []),
                "difficulty": difficulty,
                "id": task_data.get("id"),
            })
        except (ValueError, IndexError) as e:
            # Record as an explicit failure (score 0) instead of silently skipping,
            # so aggregate metrics correctly reflect unrunnable tasks.
            logger.error(f"  Task {task_idx} env setup failed, recording as score=0: {e}")
            results.append(0.0)
            continue
        task_description = f"{game_name}: {env._get_goal()}"

        if args.run_baseline:
            env_base = PDDLEnv({})
            try:
                base_system_prompt, base_init_user_prompt = env_base.set_env({
                    "game_name": game_name,
                    "problem_index": problem_index,
                    "goal": task_data.get("goal"),
                    "subgoals": task_data.get("subgoals", []),
                    "difficulty": difficulty,
                    "id": task_data.get("id"),
                })
            except (ValueError, IndexError) as e:
                logger.error(f"  [Baseline] Task {task_idx} env setup failed, recording as score=0: {e}")
                baseline_results.append(0.0)
                base_system_prompt = None
                base_init_user_prompt = None

            if base_system_prompt is not None:
                if is_mas_mode:
                    score_base, won_base, _ = run_multiturn_mas(
                        baseline_mas, env_base, base_system_prompt, base_init_user_prompt,
                        generation_config, args.max_turns, timeout_seconds=timeout_seconds,
                        memory_task_description=f"{game_name}: {env_base._get_goal()}",
                    )
                else:
                    score_base, won_base, _ = run_multiturn_single(
                        conmem.llm, env_base, base_system_prompt, base_init_user_prompt,
                        memory_ctx="", max_turns=args.max_turns, timeout_seconds=timeout_seconds,
                    )
                baseline_results.append(score_base)
                if won_base:
                    baseline_won_count += 1
                logger.info(f"  [Baseline] score={score_base:.2f}, won={won_base}")

        if is_mas_mode:
            score, won, steps = run_multiturn_mas(
                mas,
                env,
                system_prompt,
                init_user_prompt,
                generation_config,
                args.max_turns,
                timeout_seconds=timeout_seconds,
                memory_task_description=task_description,
            )
        else:
            memory_ctx = conmem.on_task_start(
                task_description=task_description, agent_role="executor", task_id=task_id)
            score, won, steps = run_multiturn_single(
                conmem.llm, env, system_prompt, init_user_prompt, memory_ctx, args.max_turns,
                conmem=conmem,
                task_id=task_id,
                agent_role="executor",
                task_description=task_description,
                timeout_seconds=timeout_seconds,
            )

        results.append(score)
        if won:
            won_count += 1
        feedback = f"score={score:.2f}, won={won}, turns={len(steps)}"
        logger.info(f"  [{args.mas_type.upper()}+ConMem] {feedback}")

        # Store trajectory in memory unless read-only mode is enabled.
        if not getattr(args, 'read_only_memory', False):
            outcome = "success" if won else ("partial" if score > 0 else "failure")
            traj_data = {"task_description": f"{game_name}: {env._get_goal()}", "outcome": outcome, "steps": steps}
            if is_mas_mode:
                centralized_memory.add_memory(trajectory=traj_data, task_id=task_id,
                    task_description=f"{game_name}: {env._get_goal()}", outcome=outcome)
            else:
                conmem.on_task_complete(task_id=task_id, task_description=f"{game_name}: {env._get_goal()}",
                    outcome=outcome, trajectory=traj_data)
        else:
            logger.info("  [Read-Only Mode] Skipping memory storage")

        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            avg = sum(results) / len(results)
            logger.info(f"\n--- Progress: {i+1}/{len(tasks)} avg_score={avg:.2%} won={won_count}/{len(results)} ---")

    elapsed = time.time() - t_start
    avg_score = sum(results) / len(results) if results else 0
    win_rate = won_count / len(results) if results else 0
    from utils.stats import stats
    api = stats.to_dict()

    logger.info(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    logger.info(f"Mode: {args.mas_type.upper()} | Avg score: {avg_score:.2%} | Win rate: {win_rate:.2%} ({won_count}/{len(results)})")
    if baseline_results:
        baseline_avg = sum(baseline_results) / len(baseline_results)
        baseline_win_rate = baseline_won_count / len(baseline_results)
        logger.info(f"Baseline: avg_score={baseline_avg:.2%} | win_rate={baseline_win_rate:.2%} ({baseline_won_count}/{len(baseline_results)})")
    logger.info(f"Time: {elapsed:.1f}s | Cards: {conmem.storage.count_active_cards()}")
    logger.info(f"\n{stats.summary()}")

    os.makedirs(storage_dir, exist_ok=True)
    with open(os.path.join(storage_dir, "eval_results.json"), "w") as f:
        json.dump({
            "benchmark": "pddl", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"mas_type": args.mas_type, "model_name": args.model_name,
                "config_model": bench_config.model.llm_name_or_path,
                "max_new_tokens": bench_config.generation.max_new_tokens,
                "temperature": bench_config.generation.temperature,
                "max_turns": args.max_turns,
                "timeout_seconds": timeout_seconds,
                "api_base": args.api_base,
                "num_tasks": len(results), "start_from": args.start_from},
            "results": {"avg_score": avg_score, "win_rate": win_rate, "won": won_count,
                "total": len(results), "scores": results},
            "cost": {"elapsed_seconds": round(elapsed, 1), "total_tokens": api["total_tokens"],
                "prompt_tokens": api["total_prompt_tokens"], "completion_tokens": api["total_completion_tokens"],
                "total_api_calls": api["total_calls"], "total_failures": api["total_failures"], "by_source": api["by_source"]},
            "memory": {"active_cards": conmem.storage.count_active_cards(), "current_round": conmem.storage.get_current_round()},
            **({"baseline": {
                "avg_score": sum(baseline_results) / len(baseline_results),
                "win_rate": baseline_won_count / len(baseline_results),
                "won": baseline_won_count,
                "total": len(baseline_results),
                "scores": baseline_results,
            }} if baseline_results else {}),
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {os.path.join(storage_dir, 'eval_results.json')}")


def main():
    parser = argparse.ArgumentParser(description="ConMem + PDDL Evaluation (Multi-turn)")
    parser.add_argument("--read_only_memory", action="store_true",
                        help="Read-only memory mode: use existing memories without storing new ones.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (default: configs/conmem/pddl.yaml).")
    parser.add_argument("--mas_type", type=str, default="single", choices=["single", "camel", "macnet", "autogen"])
    parser.add_argument("--api_base", type=str, default="http://localhost:8100/v1")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (default: value from config).")
    parser.add_argument("--num_tasks", type=int, default=20)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--storage_dir", type=str, default="./conmem_storage",
                        help="Result output directory.")
    parser.add_argument("--memory_storage_dir", type=str, default=None,
                        help="Shared ConMem storage directory. Default: <project_root>/conmem_shared_storage.")
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum generation token count (default: config value)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_turns", type=int, default=None,
                        help="Optional max interaction turns override (default: config)")
    parser.add_argument("--timeout_seconds", type=float, default=None,
                        help="Optional per-task wall-clock timeout override (default: config)")
    args = parser.parse_args()
    _setup_imports()
    run_evaluation(args)

if __name__ == "__main__":
    main()
