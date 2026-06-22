import re

from transformers import GenerationConfig

from mas_core.base_memory_mas import BaseMemoryMAS
from utils.message import Trajectory
from common.utils.code_utils import collect_python_program, rename_function
from common.interactions.base_interaction import (
    InteractionConfig,
    InteractionManager,
    InteractionDataProto
)


def _extract_kodcode_action(message_graph, function_name: str) -> str:
    candidates = []
    if message_graph.action:
        candidates.append(message_graph.action)
    graph = getattr(message_graph, "mas_message_graph", None)
    if graph is not None:
        for _, data in graph.nodes(data=True):
            msg = data.get("message")
            response = getattr(msg, "response", None)
            if response:
                candidates.append(response)

    for text in reversed(candidates):
        program = collect_python_program(text or "")
        if re.search(r"^\s*def\s+\w+\s*\(", program, re.MULTILINE):
            return rename_function(program, function_name) if function_name else program
    if candidates:
        fallback_program = collect_python_program(candidates[-1] or "")
        if fallback_program:
            return fallback_program
    return message_graph.action or ""

class SingleTurnInteractionManager(InteractionManager):
    def __init__(
        self,
        memory_mas: BaseMemoryMAS,
        interaction_config: InteractionConfig,
        generation_config: GenerationConfig
    ):
        super().__init__(memory_mas, interaction_config, generation_config)

    def run_inter_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:

        # preprocess: clip the prompt length
        domain_instructions = gen_batch.no_tensor_batch["domain_instructions"]
        task_descriptions = gen_batch.no_tensor_batch["task_descriptions"]
        envs = gen_batch.no_tensor_batch["envs"]
        function_names = gen_batch.no_tensor_batch.get("function_names")
        if function_names is None:
            function_names = []
            for env in envs:
                test_info = getattr(env, "test_info", None)
                if test_info:
                    function_names.append(test_info[0].get("function_name", ""))
                else:
                    function_names.append("")

        # call mas to generate
        message_graphs = self.memory_mas.generate(
            domain_instructions,
            task_descriptions,
            self.generation_config,
            function_names=function_names,
        )

        trajectories: list[Trajectory] = []

        for task_description, message_graph, env, function_name in zip(task_descriptions, message_graphs, envs, function_names):

            trajectory = Trajectory()
            trajectory.task_init_description = task_description
            action = _extract_kodcode_action(message_graph, function_name)
            message_graph.action = action
            trajectory.trajectory = [message_graph]
            _, trajectory.reward, trajectory.label = env.step(action)

            trajectories.append(trajectory)

        no_tensor_batch = dict(trajectories=trajectories)

        return InteractionDataProto(no_tensor_batch=no_tensor_batch)

