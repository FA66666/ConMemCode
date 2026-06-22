"""
ConMem + KodCode evaluation script with Multi-Agent System support.

Evaluation flow:
1. Load KodCode datasets.
2. For each programming task:
   a. Generate code with a single agent or MAS, with memory retrieval enabled.
   b. Run tests to score code correctness.
   c. Store the full trajectory through on_task_complete().
3. Supports single / camel / macnet / autogen modes.

Prerequisite: a local vLLM service is running (default: localhost:8100).

Usage:
    # Single-agent mode with an API-based LLM.
    python scripts/eval_kodcode.py --mas_type single

    # CAMEL multi-agent mode.
    python scripts/eval_kodcode.py --mas_type camel

    # MacNet multi-agent mode.
    python scripts/eval_kodcode.py --mas_type macnet

    # AutoGen multi-agent mode.
    python scripts/eval_kodcode.py --mas_type autogen

    # Set task count.
    python scripts/eval_kodcode.py --mas_type camel --num_tasks 100

    # Resume from a task offset while reusing existing memory.
    python scripts/eval_kodcode.py --mas_type camel --start_from 10
"""
import argparse
import json
import logging
import os
import sys
import re
import time
import types

# Ensure the project root is on sys.path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Import setup.
# ============================================================

# mas_core/__init__.py triggers chained imports through structures and memory.
# Pre-register intermediate packages as empty modules to avoid running package
# initializers while preserving __path__ for direct imports.
_STUB_PACKAGES = [
    "mas_core",
    "mas_core.structures",
    "mas_core.structures.camel",
    "mas_core.structures.camel.prompts",
    "mas_core.structures.macnet",
    "mas_core.structures.macnet.prompts",
    "mas_core.structures.autogen",
    "mas_core.structures.autogen.prompts",
    "mas_core.memory",
    "mas_core.memory.backbone",
    "mas_core.memory.backbone.conmem",
    "mas_core.memory.backbone.reme",
    "mas_core.memory.backbone.simplemem",
]


def _setup_imports():
    """Register stub packages to avoid chained __init__.py imports."""
    for _pkg_name in _STUB_PACKAGES:
        _mod = sys.modules.get(_pkg_name)
        if _mod is None:
            _mod = types.ModuleType(_pkg_name)
            sys.modules[_pkg_name] = _mod
        _mod.__path__ = [os.path.join(PROJECT_ROOT, _pkg_name.replace(".", "/"))]
        _parent_name, _, _attr = _pkg_name.rpartition(".")
        if _parent_name and _parent_name in sys.modules:
            setattr(sys.modules[_parent_name], _attr, _mod)


# ============================================================
# Lazy import helpers.
# ============================================================

def _import_conmem_module():
    """Import ConMem core modules."""
    from mas_core.memory.backbone.conmem.conmem_module import ConMemModule
    from mas_core.memory.backbone.conmem.config import ConMemConfig
    return ConMemModule, ConMemConfig


def _import_conmem_centralized_memory():
    """Import the ConMem centralized memory adapter for MAS mode."""
    from mas_core.memory.backbone.conmem.centralized_adapter import ConMemCentralizedMemory
    from mas_core.memory.backbone.conmem.config import ConMemConfig
    return ConMemCentralizedMemory, ConMemConfig


def _import_reme_centralized_memory():
    """Import the ReMe HTTP centralized memory adapter for MAS mode."""
    from mas_core.memory.backbone.reme.centralized_adapter import ReMeCentralizedMemory
    return ReMeCentralizedMemory


def _import_simplemem_centralized_memory():
    """Import the SimpleMem centralized memory adapter for MAS mode."""
    from mas_core.memory.backbone.simplemem.centralized_adapter import SimpleMemCentralizedMemory
    return SimpleMemCentralizedMemory


def _resolve_memory_storage_dir(shared_storage_dir=None):
    from mas_core.memory.backbone.conmem.storage import resolve_conmem_storage_dir

    return resolve_conmem_storage_dir(
        shared_storage_dir=shared_storage_dir,
        project_root=PROJECT_ROOT,
    )


def _import_mas_class(mas_type: str):
    """Import the MAS class for the requested mas_type."""
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


def _import_interaction_manager():
    """Import interaction manager classes."""
    from interactions import lazy_get_inter_cls, InteractionConfig, InteractionDataProto
    from transformers import GenerationConfig
    return lazy_get_inter_cls, InteractionConfig, InteractionDataProto, GenerationConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_kodcode")


# ============================================================
# Code execution with the full KodCodeEnv environment.
# ============================================================

def compute_kodcode_reward_detailed(code: str, test_code: str, test_info: list) -> tuple[float, str, dict]:
    """Execute code in KodCodeEnv and return score plus detailed feedback."""
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
        detail = {
            "summary": f"Error: {e}",
            "score": 0.0,
            "full_feedback": f"Error: {e}",
            "test_passed": False,
            "test_results": False,
        }
        return 0.0, detail["summary"], detail

    detail = env.feedback_detail or {
        "summary": "All tests failed." if score <= 0 else f"Partial pass: {score:.1%}",
        "score": score,
        "full_feedback": "",
        "test_passed": score >= 1.0,
        "test_results": score >= 1.0,
    }
    return score, detail["summary"], detail


def compute_kodcode_reward(code: str, test_code: str, test_info: list) -> tuple[float, str]:
    """Execute code and compute reward while preserving the legacy interface."""
    score, summary, _ = compute_kodcode_reward_detailed(code, test_code, test_info)
    return score, summary


def contains_python_function(code: str) -> bool:
    from common.utils.code_utils import contains_python_function_source

    return contains_python_function_source((code or "").strip())


def extract_code_from_message_graph(message_graph) -> str:
    from common.utils.code_utils import collect_python_program

    if message_graph is None:
        return ""

    candidates = []
    if getattr(message_graph, "action", None):
        candidates.append(message_graph.action)
    graph = getattr(message_graph, "mas_message_graph", None)
    if graph is not None:
        try:
            import networkx as nx
            node_ids = list(nx.topological_sort(graph))
        except Exception:
            node_ids = list(graph.nodes())
        for node_id in reversed(node_ids):
            msg = graph.nodes[node_id].get("message")
            response = getattr(msg, "response", None) if msg is not None else None
            if response:
                candidates.append(response)

    for candidate in candidates:
        program = collect_python_program((candidate or "").strip())
        if contains_python_function(program):
            return program
    return collect_python_program(candidates[0]) if candidates else ""


def prompt_response_stats(message_graph) -> dict:
    stats = {"prompt_chars": 0, "response_chars": 0, "agent_steps": 0}
    if message_graph is None or message_graph.mas_message_graph is None:
        return stats
    for _, node_data in message_graph.mas_message_graph.nodes(data=True):
        msg = node_data.get("message")
        if msg is None:
            continue
        stats["agent_steps"] += 1
        stats["prompt_chars"] += len(getattr(msg, "formatted_system_prompt", None) or "")
        stats["prompt_chars"] += len(getattr(msg, "formatted_user_prompt", None) or "")
        stats["response_chars"] += len(getattr(msg, "response", None) or "")
    return stats


# ============================================================
# Single-agent LLM code generation.
# ============================================================

def generate_code(llm, task_prompt: str, memory_context: str = "") -> str:
    """Generate Python code with an LLM in single-agent mode."""
    system_prompt = (
        "You are an expert Python programmer. "
        "Write a correct Python function that solves the given problem. "
        "Output ONLY the Python code, no explanation."
    )

    user_prompt = ""
    if memory_context:
        user_prompt += f"{memory_context}\n\n"
    user_prompt += f"## Problem\n{task_prompt}\n\nWrite the Python solution:"

    response = llm.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=4096)
    return response


# ============================================================
# Convert MAS trajectories to the ConMem trajectory schema.
# ============================================================

def message_graph_to_trajectory_data(
    question: str,
    message_graph,
    outcome: str,
    feedback_detail: dict,
) -> dict:
    """
    Convert a MAS MessageGraph into the trajectory dictionary stored by ConMem.

    Extract each agent's input and output from the mas_message_graph DAG and
    emit steps in topological order.
    
    Args:
        feedback_detail: Detailed feedback containing summary, score,
            test_passed, and full_feedback.
    """
    steps = []
    
    # Build feedback data, accepting either a dict or a string.
    step_feedback = {
        "summary": feedback_detail.get("summary", ""),
        "score": feedback_detail.get("score", 0.0),
        "test_passed": feedback_detail.get("test_passed", False),
        "full_feedback": feedback_detail.get("full_feedback", ""),
    } if isinstance(feedback_detail, dict) else feedback_detail

    if message_graph is not None and message_graph.mas_message_graph is not None:
        import networkx as nx
        try:
            topo_order = list(nx.topological_sort(message_graph.mas_message_graph))
        except Exception:
            topo_order = list(message_graph.mas_message_graph.nodes())

        for step_idx, node_id in enumerate(topo_order, start=1):
            node_data = message_graph.mas_message_graph.nodes[node_id]
            msg = node_data.get("message")
            agent_response = msg.response if msg and msg.response else ""
            full_input = question
            if msg is not None:
                input_parts = []
                formatted_system = getattr(msg, "formatted_system_prompt", None) or ""
                formatted_user = getattr(msg, "formatted_user_prompt", None) or ""
                if formatted_system:
                    input_parts.append(formatted_system)
                if formatted_user:
                    input_parts.append(formatted_user)
                if input_parts:
                    full_input = "\n\n".join(input_parts)

            steps.append({
                "step_index": step_idx,
                "agent": str(node_id),
                "input": full_input,
                "output": agent_response,
                "tool_calls": "",
                "feedback": step_feedback if step_idx == len(topo_order) else {},
            })
    else:
        # Fallback: use the aggregate action when no mas_message_graph exists.
        # If message_graph is None, emit an empty output.
        output = message_graph.action if message_graph is not None else ""
        steps.append({
            "step_index": 1,
            "agent": "mas",
            "input": question,
            "output": output or "",
            "tool_calls": "",
            "feedback": step_feedback,
        })

    return {
        "task_description": question,
        "outcome": outcome,
        "steps": steps,
    }


# ============================================================
# Main evaluation loop.
# ============================================================

def run_evaluation(args):
    """Run ConMem + KodCode evaluation."""
    from data.kodcode.builder import load_kodcode_splits
    from utils.config_loader import load_benchmark_config

    is_mas_mode = args.mas_type in ("camel", "macnet", "autogen")
    memory_backend = args.memory_backend.lower()
    if memory_backend in ("reme", "simplemem") and not is_mas_mode:
        raise ValueError(f"{memory_backend} memory backend currently supports MAS modes only: camel/macnet/autogen")
    memory_label = {
        "conmem": "ConMem",
        "reme": "ReMe",
        "simplemem": "SimpleMem",
    }.get(memory_backend, memory_backend)

    # ---- 1. Load config file ----
    if args.config:
        config_path = args.config
    else:
        config_path = None  # Use the default config path.
    
    bench_config = load_benchmark_config('kodcode', config_path)
    logger.info(f"Loaded config: max_new_tokens={bench_config.generation.max_new_tokens}, "
                f"temperature={bench_config.generation.temperature}, "
                f"max_turns={bench_config.interaction.max_turns}")

    # ---- 2. Initialize memory and MAS ----
    env_path = os.path.join(PROJECT_ROOT, ".env")
    storage_dir = args.storage_dir
    memory_storage_dir = _resolve_memory_storage_dir(args.memory_storage_dir)

    # Import interaction manager classes.
    lazy_get_inter_cls, InteractionConfig, InteractionDataProto, GenerationConfig = _import_interaction_manager()

    # Use the config model_name when the CLI does not specify one.
    if args.model_name is None:
        args.model_name = bench_config.model.llm_name_or_path
    
    baseline_mas = None
    reme_workspace_id = None
    simplemem_storage_dir = None

    if is_mas_mode:
        if memory_backend == "reme":
            ReMeCentralizedMemory = _import_reme_centralized_memory()
            workspace_id = args.reme_workspace or f"reme:kodcode:{args.mas_type}:{args.model_name}"
            reme_workspace_id = workspace_id
            centralized_memory = ReMeCentralizedMemory(
                base_url=args.reme_url,
                workspace_id=workspace_id,
                task_domain="kodcode",
                top_k=args.reme_top_k,
                timeout=args.reme_timeout,
                read_only=args.read_only_memory,
                trajectory_dir=args.reme_trajectory_dir or os.path.join(storage_dir, "reme_trajectories"),
            )
            conmem = centralized_memory  # Preserve the existing statistics interface.
            logger.info(f"Using ReMe workspace: {workspace_id} ({args.reme_url})")
        elif memory_backend == "simplemem":
            SimpleMemCentralizedMemory = _import_simplemem_centralized_memory()
            simplemem_storage_dir = args.simplemem_storage_dir or os.path.join(storage_dir, "simplemem_memory")
            centralized_memory = SimpleMemCentralizedMemory(
                storage_dir=simplemem_storage_dir,
                task_domain="kodcode",
                top_k=args.simplemem_top_k,
                embedding_backend=args.simplemem_embedding_backend,
                embedding_model=args.simplemem_embedding_model,
                embedding_api_base=args.simplemem_embedding_api_base,
                embedding_api_key=args.simplemem_embedding_api_key,
                embedding_timeout=args.simplemem_embedding_timeout,
                read_only=args.read_only_memory,
                trajectory_dir=args.simplemem_trajectory_dir or os.path.join(storage_dir, "simplemem_trajectories"),
            )
            conmem = centralized_memory  # Preserve the existing statistics interface.
            logger.info(f"Using SimpleMem memory store: {simplemem_storage_dir}")
        else:
            # MAS mode uses the ConMemCentralizedMemory adapter and HTTP API.
            ConMemCentralizedMemory, ConMemConfig = _import_conmem_centralized_memory()
            config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
            centralized_memory = ConMemCentralizedMemory(config, memory_storage_dir, task_domain="kodcode")
            conmem = centralized_memory.conmem  # Underlying ConMemModule for statistics.

        MASClass = _import_mas_class(args.mas_type)
        logger.info(f"Initializing {args.mas_type.upper()} MAS with API: {args.api_base}, model: {args.model_name}")
        # Create the MAS instance with the configured model name.
        mas = MASClass(
            llm_name_or_path=args.model_name,
            centralized_memory=centralized_memory,
            share_llm=True,
            task_domain="kodcode",
            api_base=args.api_base,
            model_name=args.model_name,
        )
        logger.info(f"{args.mas_type.upper()} MAS initialized with {len(mas.agents_list)} agents")

        # Create interaction manager settings. KodCode is a StaticEnv and uses one turn.
        # Config values are preferred, with CLI overrides handled earlier.
        gen_config = GenerationConfig(
            max_new_tokens=bench_config.generation.max_new_tokens,
            temperature=bench_config.generation.temperature,
            top_p=0.95,
            do_sample=bench_config.generation.temperature > 0,
        )
        inter_config = InteractionConfig(
            max_turns=bench_config.interaction.max_turns,
            max_obs_length=bench_config.interaction.max_obs_length
        )

        # Resolve the interaction manager dynamically; KodCodeEnv maps to SingleTurnInteractionManager.
        from data.kodcode.env import KodCodeEnv
        inter_cls = lazy_get_inter_cls(KodCodeEnv)
        logger.info(f"Using {inter_cls.__name__} for interaction")
        if args.run_baseline:
            baseline_mas = MASClass(
                llm_name_or_path=args.model_name,
                centralized_memory=None,
                share_llm=True,
                task_domain="kodcode",
                api_base=args.api_base,
                model_name=args.model_name,
            )
    else:
        # Single-agent mode uses the API-based ConMemModule directly.
        ConMemModule, ConMemConfig = _import_conmem_module()
        config = ConMemConfig.from_env(env_path if os.path.exists(env_path) else None)
        conmem = ConMemModule(config, memory_storage_dir, task_domain="kodcode")
        conmem.set_runtime_context(model_name=args.model_name, mas_architecture="single")

    logger.info(f"{memory_label} initialized. Storage: {storage_dir}")
    if memory_backend == "conmem":
        logger.info(f"ConMem shared storage: {memory_storage_dir}")
    elif memory_backend == "simplemem":
        logger.info(f"SimpleMem memory store: {simplemem_storage_dir}")
    logger.info(f"Mode: {args.mas_type.upper()}")

    # ---- 2. Load KodCode datasets ----
    logger.info("Loading KodCode dataset...")
    split_seed = 42
    splits = load_kodcode_splits(split_seed=split_seed)
    if args.data_split not in splits:
        raise ValueError(f"KodCode only supports data_split in {list(splits)}")
    dataset = splits[args.data_split]
    logger.info(f"Dataset loaded: {len(dataset)} samples from split={args.data_split} (seed={split_seed})")

    # Select task subset.
    end_idx = min(args.start_from + args.num_tasks, len(dataset))
    tasks = dataset.select(range(args.start_from, end_idx))
    logger.info(f"Evaluating tasks [{args.start_from}, {end_idx})")

    # ---- 3. Evaluation loop ----
    results_with_memory = []     # Memory-augmented generation results.
    results_without_memory = []  # No-memory baseline results.
    
    # Memory-card usage statistics.
    stats_with_cards = []      # Results for tasks that used at least one memory card.
    stats_without_cards = []   # Results for tasks that used no memory cards.
    card_counts = []           # Number of cards used by each task.
    per_task_records = []
    infrastructure_failures = 0
    mas_retries = 0
    
    t_start = time.time()

    for i, task in enumerate(tasks):
        task_idx = args.start_from + i
        question = task["question"].strip()
        test_code = task["test"].strip()
        test_info = task["test_info"]
        task_id = f"kodcode_{task_idx}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Task {task_idx} ({i+1}/{len(tasks)})")
        logger.info(f"Question: {question[:100]}...")

        system_prompt = (
            "You are an expert Python programmer. "
            "Write a correct Python function that solves the given problem. "
            "Output ONLY the Python code, no explanation."
        )

        # ------ No-memory baseline ------
        if args.run_baseline:
            if is_mas_mode:
                code_no_mem = ""
                try:
                    from data.kodcode.env import KodCodeEnv
                    env_base = KodCodeEnv({})
                    env_base.set_env({"prompt": question, "test": test_code, "test_info": test_info})

                    baseline_inter_manager = inter_cls(baseline_mas, inter_config, gen_config)
                    gen_batch = InteractionDataProto(no_tensor_batch={
                        "domain_instructions": [system_prompt],
                        "task_descriptions": [question],
                        "envs": [env_base],
                        "function_names": [test_info[0]["function_name"] if test_info else ""],
                    })
                    result_base = baseline_inter_manager.run_inter_loop(gen_batch)
                    baseline_trajectory = result_base.no_tensor_batch["trajectories"][0]
                    if baseline_trajectory.trajectory:
                        code_no_mem = extract_code_from_message_graph(baseline_trajectory.trajectory[-1])
                except Exception as e:
                    logger.warning(f"  [Baseline] MAS interaction failed: {e}")
                    try:
                        function_name = test_info[0]["function_name"] if test_info else ""
                        base_graphs = baseline_mas.generate(
                            task_domain_instructions=[system_prompt],
                            user_inputs=[question],
                            generation_config=gen_config.to_dict(),
                            function_names=[function_name],
                        )
                        code_no_mem = extract_code_from_message_graph(base_graphs[0])
                    except Exception as e2:
                        logger.warning(f"  [Baseline] MAS fallback failed: {e2}")
                        code_no_mem = ""
                score_no_mem, feedback_no_mem = compute_kodcode_reward(
                    code_no_mem, test_code, test_info
                )
                results_without_memory.append(score_no_mem)
                logger.info(f"  [Baseline]  score={score_no_mem:.1f}  {feedback_no_mem[:80]}")
            else:
                code_no_mem = generate_code(conmem.llm, question, memory_context="")
                score_no_mem, feedback_no_mem = compute_kodcode_reward(
                    code_no_mem, test_code, test_info
                )
                results_without_memory.append(score_no_mem)
                logger.info(f"  [Baseline]  score={score_no_mem:.1f}  {feedback_no_mem[:80]}")

        # ------ Main evaluation: memory + MAS or memory + single agent ------
        trajectory = None
        message_graph = None
        memory_context = ""
        code_with_mem = ""
        pipeline_status = "ok"
        generation_error = ""
        mas_retry_used = False
        if is_mas_mode:
            try:
                # Create the KodCode environment.
                from data.kodcode.env import KodCodeEnv
                env = KodCodeEnv({})
                env.set_env({"prompt": question, "test": test_code, "test_info": test_info})

                # Create the interaction manager and run the single-turn interaction.
                inter_manager = inter_cls(mas, inter_config, gen_config)

                gen_batch = InteractionDataProto(no_tensor_batch={
                    "domain_instructions": [system_prompt],
                    "task_descriptions": [question],
                    "envs": [env],
                    "function_names": [test_info[0]["function_name"] if test_info else ""],
                })

                result = inter_manager.run_inter_loop(gen_batch)
                trajectory = result.no_tensor_batch["trajectories"][0]

                # Extract final code from the trajectory.
                if trajectory.trajectory:
                    final_msg_graph = trajectory.trajectory[-1]
                    message_graph = final_msg_graph
                    code_with_mem = extract_code_from_message_graph(final_msg_graph)
                else:
                    code_with_mem = ""

                logger.info(f"  [MAS] {args.mas_type.upper()} generated code ({len(code_with_mem)} chars)")
            except Exception as e:
                generation_error = repr(e)
                pipeline_status = "mas_interaction_exception"
                logger.error(f"  [MAS] Interaction failed: {e}")
                # Retry through the same MAS generate() path; do not replace it with a single agent.
                try:
                    mas_retry_used = True
                    mas_retries += 1
                    function_name = test_info[0]["function_name"] if test_info else ""
                    message_graphs = mas.generate(
                        task_domain_instructions=[system_prompt],
                        user_inputs=[question],
                        generation_config=gen_config.to_dict(),
                        function_names=[function_name],
                    )
                    message_graph = message_graphs[0]
                    code_with_mem = extract_code_from_message_graph(message_graph)
                    pipeline_status = "mas_generate_retry"
                    trajectory = None
                except Exception as e2:
                    generation_error += f" | mas_generate_retry={e2!r}"
                    pipeline_status = "mas_generation_exception"
                    logger.error(f"  [MAS] Same-MAS generation retry also failed: {e2}")
                    code_with_mem = ""
                    trajectory = None
        else:
            # Single-agent mode: retrieve memory manually and call the LLM once.
            memory_context = conmem.on_task_start(
                task_description=question,
                agent_role="executor",
                task_id=task_id,
            )
            if memory_context:
                logger.info(f"  [Memory] Retrieved {len(memory_context)} chars of context")
            else:
                logger.info(f"  [Memory] No relevant memory found")
            code_with_mem = generate_code(conmem.llm, question, memory_context=memory_context)

        # Keep failures inside the selected MAS. If the MAS graph has responses
        # but action extraction is empty, recover from those responses; otherwise
        # mark the task for diagnostics and continue to the benchmark evaluator.
        if is_mas_mode and not contains_python_function(code_with_mem):
            recovered_code = extract_code_from_message_graph(message_graph)
            if contains_python_function(recovered_code):
                code_with_mem = recovered_code
                pipeline_status = "recovered_from_message_graph"
            elif pipeline_status == "ok":
                pipeline_status = "mas_empty_or_noncode_output"
            elif pipeline_status == "mas_generate_retry":
                pipeline_status = "mas_generate_retry_no_code"

        # Evaluate generated code.
        if (
            is_mas_mode
            and trajectory is not None
            and not mas_retry_used
            and pipeline_status == "ok"
            and 'env' in locals()
            and getattr(env, 'feedback_detail', None) is not None
        ):
            # MAS mode already executed env.step() inside run_inter_loop().
            score_with_mem = env.reward
            detail = env.feedback_detail
            summary = detail["summary"]
        elif is_mas_mode and message_graph is not None:
            # Same-MAS retry/recovery path: run tests outside run_inter_loop().
            score_with_mem, summary, detail = compute_kodcode_reward_detailed(
                code_with_mem, test_code, test_info
            )
        else:
            # Single-agent mode still needs to run tests.
            score_with_mem, summary, detail = compute_kodcode_reward_detailed(
                code_with_mem, test_code, test_info
            )
        feedback_with_mem = detail["full_feedback"]
        results_with_memory.append(score_with_mem)
        no_function_generated = not contains_python_function(code_with_mem)
        infrastructure_failure = (score_with_mem == 0.0 and no_function_generated)
        if infrastructure_failure:
            infrastructure_failures += 1
        
        # Track memory-card usage.
        used_cards = False
        num_cards = 0
        if is_mas_mode:
            # MAS mode reads the actual retrieval count from centralized_memory.
            num_cards = centralized_memory.get_and_reset_retrieval_count()
            used_cards = num_cards > 0
        else:
            # Single-agent mode reads the latest retrieval count from conmem.
            num_cards = conmem._last_card_count
            used_cards = num_cards > 0
        
        card_counts.append(num_cards)
        
        if used_cards:
            stats_with_cards.append(score_with_mem)
        else:
            stats_without_cards.append(score_with_mem)

        graph_stats = prompt_response_stats(message_graph)
        if trajectory is not None and getattr(trajectory, "trajectory", None):
            graph_stats = prompt_response_stats(trajectory.trajectory[-1])
        per_task_records.append({
            "task_idx": task_idx,
            "task_id": task_id,
            "score": score_with_mem,
            "used_cards": used_cards,
            "retrieved_card_count": num_cards,
            "pipeline_status": pipeline_status,
            "generation_error": generation_error,
            "mas_retry_used": mas_retry_used,
            "no_function_generated": no_function_generated,
            "infrastructure_failure": infrastructure_failure,
            "output_chars": len(code_with_mem or ""),
            "mas_prompt_chars": graph_stats["prompt_chars"],
            "mas_response_chars": graph_stats["response_chars"],
            "mas_agent_steps": graph_stats["agent_steps"],
            "summary": summary,
        })
        
        logger.info(f"  [{args.mas_type.upper()}+{memory_label}]  score={score_with_mem:.1f}  {summary}  [Cards: {num_cards}]")

        # Store trajectory in memory unless read-only mode is enabled.
        if not getattr(args, 'read_only_memory', False):
            outcome = "success" if score_with_mem >= 1.0 else (
                "partial" if score_with_mem > 0 else "failure"
            )
            if is_mas_mode:
                # MAS mode builds a full trajectory from the interaction trace or MessageGraph.
                # Only reasoning inputs/outputs are recorded; writing the trajectory is not timed.
                if trajectory is not None:
                    # Build trajectory data from the interaction trace.
                    # Walk each MessageGraph and extract every agent input/output pair.
                    steps = []
                    step_counter = 0
                    
                    if hasattr(trajectory, 'trajectory') and trajectory.trajectory:
                        for msg_graph_idx, msg_graph in enumerate(trajectory.trajectory, 1):
                            # Extract detailed input/output pairs from mas_message_graph.
                            if msg_graph.mas_message_graph is not None:
                                import networkx as nx
                                try:
                                    topo_order = list(nx.topological_sort(msg_graph.mas_message_graph))
                                except Exception:
                                    topo_order = list(msg_graph.mas_message_graph.nodes())
                                
                                for agent_id in topo_order:
                                    node_data = msg_graph.mas_message_graph.nodes[agent_id]
                                    msg_node = node_data.get("message")
                                    if msg_node:
                                        step_counter += 1
                                        # Prefer formatted prompts stored by LLMAgent.
                                        full_input = ""
                                        if getattr(msg_node, 'formatted_system_prompt', None):
                                            full_input += msg_node.formatted_system_prompt + "\n\n"
                                        if getattr(msg_node, 'formatted_user_prompt', None):
                                            full_input += msg_node.formatted_user_prompt

                                        if not full_input:
                                            # Fallback: use the raw prompt templates.
                                            full_input = f"[System: {msg_node.system_prompt_template or ''}]\n\n[User: {msg_node.user_prompt_template or ''}]"
                                        
                                        # Extract the agent response.
                                        agent_output = msg_node.response or ""
                                        
                                        # Build step feedback; only the final step carries full feedback.
                                        is_last_step = (msg_graph_idx == len(trajectory.trajectory) and 
                                                       agent_id == topo_order[-1])
                                        step_feedback = {
                                            "summary": detail["summary"] if is_last_step else "",
                                            "score": detail["score"] if is_last_step else 0.0,
                                            "test_passed": detail["test_passed"] if is_last_step else False,
                                            "full_feedback": detail["full_feedback"] if is_last_step else "",
                                        }
                                        
                                        steps.append({
                                            "step_index": step_counter,
                                            "agent": str(agent_id),
                                            "input": full_input,
                                            "output": agent_output,
                                            "tool_calls": "",
                                            "feedback": step_feedback,
                                        })
                            
                            # If no mas_message_graph exists, record the aggregate MAS action.
                            if not steps or msg_graph.mas_message_graph is None:
                                step_counter += 1
                                is_last_step = (msg_graph_idx == len(trajectory.trajectory))
                                step_feedback = {
                                    "summary": detail["summary"] if is_last_step else "",
                                    "score": detail["score"] if is_last_step else 0.0,
                                    "test_passed": detail["test_passed"] if is_last_step else False,
                                    "full_feedback": detail["full_feedback"] if is_last_step else "",
                                }
                                steps.append({
                                    "step_index": step_counter,
                                    "agent": "mas",
                                    "input": question,
                                    "output": msg_graph.action if hasattr(msg_graph, 'action') else str(msg_graph),
                                    "tool_calls": "",
                                    "feedback": step_feedback,
                                })
                    
                    if not steps:
                        steps = [{
                            "step_index": 1,
                            "agent": "mas",
                            "input": question,
                            "output": code_with_mem,
                            "tool_calls": "",
                            "feedback": {
                                "summary": detail["summary"],
                                "score": detail["score"],
                                "test_passed": detail["test_passed"],
                                "full_feedback": detail["full_feedback"],
                            },
                        }]
                    
                    trajectory_data = {
                        "task_description": question,
                        "outcome": outcome,
                        "test_code": test_code,
                        "steps": steps,
                    }
                else:
                    # Fallback: build trajectory data from MessageGraph directly.
                    trajectory_data = message_graph_to_trajectory_data(
                        question, message_graph, outcome, detail,
                    )

                if isinstance(trajectory_data, dict):
                    trajectory_data["infrastructure_failure"] = infrastructure_failure

                centralized_memory.add_memory(
                    trajectory=trajectory_data,
                    task_id=task_id,
                    task_description=question,
                    outcome=outcome,
                )
            else:
                # Single-agent mode.
                trajectory_data = {
                    "task_description": question,
                    "outcome": outcome,
                    "test_code": test_code,
                    "steps": [
                        {
                            "step_index": 1,
                            "agent": "executor",
                            "input": question,
                            "output": code_with_mem,
                            "tool_calls": "",
                            "feedback": {
                                "summary": detail["summary"],
                                "score": detail["score"],
                                "test_passed": detail["test_passed"],
                                "full_feedback": detail["full_feedback"],
                            },
                        },
                    ],
                }
                conmem.on_task_complete(
                    task_id=task_id,
                    task_description=question,
                    trajectory=trajectory_data,
                    outcome=outcome,
                )
        else:
            logger.info("  [Read-Only Mode] Skipping memory storage")

        # Periodically print progress.
        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            mem_rate = sum(results_with_memory) / len(results_with_memory)
            
            # Memory-card statistics.
            with_cards_rate = sum(stats_with_cards) / len(stats_with_cards) if stats_with_cards else 0.0
            without_cards_rate = sum(stats_without_cards) / len(stats_without_cards) if stats_without_cards else 0.0
            
            logger.info(f"\n--- Progress: {i+1}/{len(tasks)} ---")
            logger.info(f"  {args.mas_type.upper()}+{memory_label} pass rate:  {mem_rate:.2%} ({int(sum(results_with_memory))}/{len(results_with_memory)})")
            if stats_with_cards or stats_without_cards:
                avg_cards = sum(card_counts) / len(card_counts) if card_counts else 0.0
                logger.info(f"  With cards:    {with_cards_rate:.2%} ({int(sum(stats_with_cards))}/{len(stats_with_cards)})")
                logger.info(f"  Without cards: {without_cards_rate:.2%} ({int(sum(stats_without_cards))}/{len(stats_without_cards)})")
                logger.info(f"  Avg cards/task: {avg_cards:.2f} (total: {sum(card_counts)})")
            if args.run_baseline and results_without_memory:
                base_rate = sum(results_without_memory) / len(results_without_memory)
                logger.info(f"  Baseline pass rate: {base_rate:.2%} ({int(sum(results_without_memory))}/{len(results_without_memory)})")
            logger.info(f"  Memory cards:  {conmem.storage.count_active_cards()}")
            logger.info(f"  Current round: {conmem.storage.get_current_round()}")

    # ---- 4. Final results ----
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")

    mem_rate = sum(results_with_memory) / len(results_with_memory)
    logger.info(f"Mode:               {args.mas_type.upper()}")
    if is_mas_mode:
        logger.info(f"Model:              {args.model_name}")
    logger.info(f"{memory_label} pass rate:   {mem_rate:.2%} ({int(sum(results_with_memory))}/{len(results_with_memory)})")

    if args.run_baseline and results_without_memory:
        base_rate = sum(results_without_memory) / len(results_without_memory)
        logger.info(f"Baseline pass rate: {base_rate:.2%} ({int(sum(results_without_memory))}/{len(results_without_memory)})")
        logger.info(f"Improvement:        {mem_rate - base_rate:+.2%}")

    logger.info(f"Total memory cards: {conmem.storage.count_active_cards()}")
    logger.info(f"Total rounds:       {conmem.storage.get_current_round()}")

    # API usage stats
    from utils.stats import stats
    logger.info(f"\n{'='*60}")
    logger.info("API USAGE STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"\n{stats.summary()}")

    # Save results with a discriminative filename.
    from utils.stats import stats
    api = stats.to_dict()
    reme_usage_stats = centralized_memory.get_usage_stats() if memory_backend == "reme" and is_mas_mode else None
    reme_trajectory_files = centralized_memory.get_saved_trajectory_files() if memory_backend == "reme" and is_mas_mode else None
    simplemem_usage_stats = centralized_memory.get_usage_stats() if memory_backend == "simplemem" and is_mas_mode else None
    simplemem_trajectory_files = centralized_memory.get_saved_trajectory_files() if memory_backend == "simplemem" and is_mas_mode else None
    if reme_usage_stats:
        logger.info("ReMe service usage: %s", reme_usage_stats)
    if reme_trajectory_files:
        logger.info("Saved ReMe trajectories: %s", len(reme_trajectory_files))
    if simplemem_usage_stats:
        logger.info("SimpleMem usage: %s", simplemem_usage_stats)
    if simplemem_trajectory_files:
        logger.info("Saved SimpleMem trajectories: %s", len(simplemem_trajectory_files))
    
    # Filename format: eval_results_{mas_type}_{model}_{num_tasks}_{timestamp}.json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split('/')[-1] if args.model_name else "default"
    safe_model_name = model_short.replace('-', '_').replace('.', '_')
    results_filename = f"eval_results_{args.mas_type}_{safe_model_name}_n{len(tasks)}_{timestamp}.json"
    results_file = os.path.join(storage_dir, results_filename)
    results_data = {
        "benchmark": "kodcode",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "mas_type": args.mas_type,
            "memory_backend": memory_backend,
            "reme_url": args.reme_url if memory_backend == "reme" else None,
            "reme_workspace": reme_workspace_id if memory_backend == "reme" else None,
            "simplemem_storage_dir": simplemem_storage_dir if memory_backend == "simplemem" else None,
            "simplemem_top_k": args.simplemem_top_k if memory_backend == "simplemem" else None,
            "simplemem_embedding_backend": args.simplemem_embedding_backend if memory_backend == "simplemem" else None,
            "simplemem_embedding_model": args.simplemem_embedding_model if memory_backend == "simplemem" else None,
            "simplemem_embedding_api_base": args.simplemem_embedding_api_base if memory_backend == "simplemem" else None,
            "model_name": args.model_name,
            "config_model": bench_config.model.llm_name_or_path,
            "api_base": args.api_base,
            "max_new_tokens": bench_config.generation.max_new_tokens,
            "temperature": bench_config.generation.temperature,
            "max_turns": bench_config.interaction.max_turns,
            "num_tasks": len(tasks),
            "start_from": args.start_from,
            "data_split": args.data_split,
        },
        "results": {
            "pass_rate": mem_rate,
            "correct": int(sum(results_with_memory)),
            "total": len(results_with_memory),
            "scores": results_with_memory,
        },
        "cost": {
            "elapsed_seconds": round(elapsed, 1),
            "total_tokens": api["total_tokens"],
            "prompt_tokens": api["total_prompt_tokens"],
            "completion_tokens": api["total_completion_tokens"],
            "total_api_calls": api["total_calls"],
            "total_failures": api["total_failures"],
            "by_source": api["by_source"],
            "reme_service": reme_usage_stats,
            "simplemem_service": simplemem_usage_stats,
        },
        "reme_trajectory_files": reme_trajectory_files,
        "simplemem_trajectory_files": simplemem_trajectory_files,
        "memory_card_stats": {
            "with_cards": {
                "correct": int(sum(stats_with_cards)),
                "total": len(stats_with_cards),
                "pass_rate": sum(stats_with_cards) / len(stats_with_cards) if stats_with_cards else 0.0,
            },
            "without_cards": {
                "correct": int(sum(stats_without_cards)),
                "total": len(stats_without_cards),
                "pass_rate": sum(stats_without_cards) / len(stats_without_cards) if stats_without_cards else 0.0,
            },
            "card_counts": {
                "total_cards_used": int(sum(card_counts)),
                "avg_cards_per_task": sum(card_counts) / len(card_counts) if card_counts else 0.0,
                "tasks_with_cards": len(stats_with_cards),
                "tasks_without_cards": len(stats_without_cards),
            },
        },
        "pipeline": {
            "infrastructure_failures": infrastructure_failures,
            "mas_retries": mas_retries,
            "tasks": per_task_records,
        },
    }
    if args.run_baseline and results_without_memory:
        results_data["baseline"] = {
            "pass_rate": base_rate,
            "correct": int(sum(results_without_memory)),
            "total": len(results_without_memory),
            "scores": results_without_memory,
        }

    os.makedirs(storage_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="ConMem + KodCode Evaluation (Multi-Agent Support)")
    parser.add_argument("--read_only_memory", action="store_true",
                        help="Read-only memory mode: use existing memories without storing new ones.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (default: configs/conmem/kodcode.yaml).")
    parser.add_argument("--mas_type", type=str, default="single",
                        choices=["single", "camel", "macnet", "autogen"],
                        help="MAS host type: single, camel, macnet, or autogen (default: single).")
    parser.add_argument("--api_base", type=str, default="http://localhost:8100/v1",
                        help="vLLM OpenAI-compatible API base URL (default: http://localhost:8100/v1)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (default: value from config).")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Number of tasks to evaluate (default: 10).")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Task offset to start from (default: 0).")
    parser.add_argument("--data_split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="KodCode data split (default: test).")
    parser.add_argument("--storage_dir", type=str, default="./conmem_storage",
                        help="Result output directory (default: ./conmem_storage).")
    parser.add_argument("--memory_storage_dir", type=str, default=None,
                        help="Shared ConMem storage directory. Default: <project_root>/conmem_shared_storage.")
    parser.add_argument("--memory_backend", type=str, default="conmem",
                        choices=["conmem", "reme", "simplemem"],
                        help="Memory backend: conmem, reme, or simplemem (default: conmem).")
    parser.add_argument("--reme_url", type=str, default="http://127.0.0.1:8003/",
                        help="ReMe HTTP service URL (default: http://127.0.0.1:8003/).")
    parser.add_argument("--reme_workspace", type=str, default=None,
                        help="ReMe workspace_id. Default: reme:kodcode:<mas_type>:<model_name>.")
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
    parser.add_argument("--run_baseline", action="store_true",
                        help="Also run a no-memory baseline for comparison.")

    args = parser.parse_args()

    # Set up imports for all modes and avoid chained __init__.py imports.
    _setup_imports()

    run_evaluation(args)


if __name__ == "__main__":
    main()
