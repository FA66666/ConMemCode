import copy
import torch
from transformers import GenerationConfig
from typing import Optional, Dict, Any, List

from common.interactions.base_interaction import (
    InteractionDataProto,
    InteractionConfig, 
    InteractionManager
)
from mas_core.base_memory_mas import BaseMemoryMAS
from utils.message import MessageGraph, Trajectory

# ConMem imports
try:
    from mas_core.memory.backbone.conmem.conmem_module import ConMemModule
except ImportError:
    ConMemModule = None
    

class MultiTurnInteractionManager(InteractionManager):
    """Multi-turn interaction manager with ConMem integration."""
    
    def __init__(
        self,
        memory_mas: BaseMemoryMAS,
        interaction_config: InteractionConfig,
        generation_config: GenerationConfig,
        conmem: Optional[Any] = None,
        agent_role: str = "executor"
    ):
        """
        Initialize the multi-turn interaction manager.
        
        Args:
            memory_mas: MAS instance.
            interaction_config: Interaction settings.
            generation_config: Generation settings.
            conmem: Optional ConMemModule instance for memory retrieval and storage.
            agent_role: Agent role used for ConMem retrieval.
        """
        super().__init__(memory_mas, interaction_config, generation_config)
        self.conmem = conmem
        self.agent_role = agent_role
        self._memory_contexts: Dict[str, str] = {}  # task_id -> memory_context      

    def run_inter_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """Run main LLM generation loop (conversation format) - with ConMem integration."""
        batch_size = len(gen_batch.no_tensor_batch["domain_instructions"])
        task_descriptions = gen_batch.no_tensor_batch.get("task_descriptions", [])
        
        # Generate task IDs if not provided
        task_ids = gen_batch.no_tensor_batch.get("task_ids", None)
        if task_ids is None:
            import uuid
            task_ids = [f"batch_{uuid.uuid4().hex[:8]}_{i}" for i in range(batch_size)]
            gen_batch.no_tensor_batch["task_ids"] = task_ids
        
        rollings = gen_batch   
        rollings.no_tensor_batch["inter_histories"] = [[] for _ in range(batch_size)]
        rollings.no_tensor_batch["trajectories"] = [
            Trajectory(task_init_description=task_descriptions[i] if i < len(task_descriptions) else "") 
            for i in range(batch_size)
        ]
        
        # Step 1: Initialize ConMem memory contexts for each task
        self._memory_contexts = {}
        if self.conmem:
            for i, task_desc in enumerate(task_descriptions):
                if i < len(task_ids):
                    memory_context = self.conmem.on_task_start(
                        task_description=task_desc,
                        agent_role=self.agent_role,
                        task_id=task_ids[i]
                    )
                    self._memory_contexts[task_ids[i]] = memory_context or ""
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        
        # Track steps for ConMem
        all_steps = {task_id: [] for task_id in task_ids}
        
        for turn in range(self.interaction_config.max_turns):
            if not active_mask.sum():   
                break            
            mask_list = active_mask.tolist()  
            rollings_active = {
                k: [item for item, keep in zip(v, mask_list) if keep]
                for k, v in rollings.no_tensor_batch.items()
            }

            # Step 2: Build task contexts with memory injection
            task_contexts = self._build_task_contexts(rollings_active)

            message_graphs = self.memory_mas.generate(
                rollings_active["domain_instructions"], 
                task_contexts, 
                self.generation_config
            )

            responses = [msg_graph.action for msg_graph in message_graphs]
            responses = self._postprocess_responses(responses, rollings_active["envs"])
            all_responses, all_message_graphs = self._example_level_pad(responses, message_graphs, active_mask)
            
            next_obs, dones = self._execute_predictions(rollings, all_responses, active_mask)
            processed_obs = self._postprocess_observations(next_obs) 
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            
            # Step 3: Record steps for ConMem
            active_task_ids = [tid for tid, keep in zip(task_ids, mask_list) if keep]
            for i, (task_id, response, obs, reward, done) in enumerate(
                zip(active_task_ids, responses, processed_obs, 
                    [r for r, k in zip([0.0]*len(dones), mask_list) if k], dones)
            ):
                all_steps[task_id].append({
                    "step_index": turn + 1,
                    "agent": self.agent_role,
                    "input": task_contexts[i] if i < len(task_contexts) else "",
                    "output": response,
                    "tool_calls": "",
                    "feedback": f"obs={obs[:100]}, reward={reward}, done={done}"
                })
            
            interaction_histories, trajectories = self._update_interaction_history(
                rollings, 
                all_responses, 
                all_message_graphs, 
                processed_obs
            )
            rollings.no_tensor_batch["inter_histories"] = interaction_histories
            rollings.no_tensor_batch["trajectories"] = trajectories
            
            # Step 4: Optional - refresh memory context per turn
            if self.conmem and turn < self.interaction_config.max_turns - 1:
                for i, task_id in enumerate(task_ids):
                    if mask_list[i]:  # Only for active tasks
                        refreshed = self.conmem.on_agent_step(
                            task_description=task_descriptions[i] if i < len(task_descriptions) else "",
                            agent_role=self.agent_role,
                            task_id=task_id,
                            agent_message=all_responses[i] if i < len(all_responses) else "",
                            observation=processed_obs[i] if i < len(processed_obs) else ""
                        )
                        if refreshed:
                            self._memory_contexts[task_id] = refreshed
        
        # Step 5: Store trajectories to ConMem
        if self.conmem:
            for trajectory, task_id in zip(rollings.no_tensor_batch["trajectories"], task_ids):
                reward = trajectory.reward if hasattr(trajectory, 'reward') else 0.0
                outcome = "success" if reward >= 1.0 else ("partial" if reward > 0 else "failure")
                trajectory_data = {
                    "task_description": trajectory.task_init_description or "",
                    "outcome": outcome,
                    "steps": all_steps.get(task_id, [])
                }
                self.conmem.on_task_complete(
                    task_id=task_id,
                    task_description=trajectory.task_init_description or "",
                    trajectory=trajectory_data,
                    outcome=outcome
                )
        
        return rollings
    
    def _build_task_contexts(self, rollings: dict) -> list[str]:
        """Build task contexts with ConMem memory injection."""
        task_descriptions = rollings.get("task_descriptions")
        if task_descriptions is None:
            raise ValueError("task_descriptions is required")
        
        inter_histories = rollings.get("inter_histories")
        if inter_histories is None:
            raise ValueError("inter_histories is required")
        
        task_ids = rollings.get("task_ids", [])
        
        conversations: list[list[dict]] = []
        for i, (task_description, inter_history) in enumerate(zip(task_descriptions, inter_histories)):
            # Inject ConMem memory context if available
            memory_context = ""
            if task_ids and i < len(task_ids):
                memory_context = self._memory_contexts.get(task_ids[i], "")
            
            if memory_context:
                # Prepend memory context to the initial prompt
                init_prompt = [{"role": "system", "content": memory_context}]
                init_prompt.append({"role": "user", "content": task_description})
            else:
                init_prompt = [{"role": "user", "content": task_description}]
            
            conversations.append(init_prompt + inter_history)
        
        task_contexts = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=False
        )
        return task_contexts
    
    def _update_interaction_history(
        self, 
        rollings: InteractionDataProto, 
        responses: list[str], 
        message_graphs: list[MessageGraph], 
        observations: list[str]
    ) -> tuple[list[list[dict]], list[Trajectory]]:
        # Update conversations and wrap observations in <information> tags.
        inter_histories = copy.deepcopy(rollings.no_tensor_batch.get("inter_histories"))
        assert len(inter_histories) == len(responses) == len(observations)
        for inter_history, response, observation in zip(inter_histories, responses, observations):
            assistant_info = {"role": "assistant", "content": response}
            # Format observations with <information> tags to match LatentMem-style prompts.
            formatted_obs = f"<information>{observation}</information>" if observation else observation
            user_info = {"role": "user", "content": formatted_obs}
            
            inter_history.append(assistant_info)
            inter_history.append(user_info)
        
        # update trajectories
        trajectories = copy.deepcopy(rollings.no_tensor_batch.get("trajectories"))
        assert len(trajectories) == len(responses) == len(observations)
        for trajectory, message_graph, observation in zip(trajectories, message_graphs, observations):
            if message_graph is not None:  
                message_graph.observation = observation  
                trajectory.add_step(message_graph)
        
        return inter_histories, trajectories
    
    def _postprocess_responses(self, responses: list[str], envs: list) -> list[str]:
        processed_responses_str = []
        for r, env in zip(responses, envs):
            processed_r = env.preprocess_action(r)
            processed_responses_str.append(processed_r)

        return processed_responses_str


    def _example_level_pad(
        self, responses: list[str], message_graphs: list[MessageGraph], active_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]: 
        assert active_mask.sum() == len(responses)
        assert len(responses) == len(message_graphs)
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        
        # Create masked response strings
        padded_responses = [""] * batch_size
        padded_message_graphs = [None] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses[i] = responses[s]
                padded_message_graphs[i] = message_graphs[s]
                s += 1
                
        return padded_responses, padded_message_graphs

    def _execute_predictions(self, rollings: InteractionDataProto, responses: list[str], active_mask: torch.Tensor) -> tuple[list[str], list[str]]:
        observations = []
        dones = []
        for response, env, is_active in zip(responses, rollings.no_tensor_batch["envs"], active_mask):
            if is_active:
                observation, _, done = env.step(response)
            else:   
                observation = ""
                done = True
            observations.append(observation)
            dones.append(done)

        return observations, dones

    
    def _postprocess_observations(self, observations: list[str]) -> list[str]:
        next_obs_ids = self.tokenizer(
            observations,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            padding_side="right" 
        )['input_ids']

        max_len = self.interaction_config.max_obs_length
        if next_obs_ids.shape[1] > max_len:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {max_len}")
            extra_text = "..."
            extra_ids = self.tokenizer.encode(extra_text, add_special_tokens=False)
            extra_len = len(extra_ids)

            new_obs_ids = []
            for row in next_obs_ids:
                valid_len = (row != self.tokenizer.pad_token_id).sum().item()

                if valid_len > max_len:
                    truncated = row[: max_len - extra_len]
                    new_row = torch.cat(
                        [truncated, torch.tensor(extra_ids, device=row.device)],
                        dim=0
                    )
                else:
                    new_row = row[:max_len]

                new_obs_ids.append(new_row.unsqueeze(0))

            next_obs_ids = torch.cat(new_obs_ids, dim=0)
            observations = self.tokenizer.batch_decode(next_obs_ids, skip_special_tokens=True)

        return observations
