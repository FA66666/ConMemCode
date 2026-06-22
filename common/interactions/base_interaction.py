from transformers import GenerationConfig

from typing import Dict, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from mas_core.base_memory_mas import BaseMemoryMAS

@dataclass
class InteractionConfig:
    batch_size: Optional[int] = None
    max_turns: Optional[int] = None
    max_obs_length: Optional[int] = None


@dataclass
class InteractionDataProto:
    batch: Dict = field(default_factory=dict)
    no_tensor_batch: Dict = field(default_factory=dict)

class InteractionManager(ABC):

    def __init__(
        self,
        memory_mas: BaseMemoryMAS,
        interaction_config: InteractionConfig,
        generation_config: GenerationConfig,
    ):
        self.memory_mas = memory_mas
        # shared_tokenizer is optional; some MAS implementations do not expose it.
        self.tokenizer = getattr(self.memory_mas, 'shared_tokenizer', None)
        self.interaction_config = interaction_config
        self.generation_config = generation_config

    @abstractmethod
    def run_inter_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        ...

    def _clip_batch(self, batch: list[str]) -> list[str]:
        """Clip a text batch when a tokenizer is available."""
        shared_tokenizer = getattr(self.memory_mas, 'shared_tokenizer', None)
        if shared_tokenizer is None:
            # Without a tokenizer, keep the original batch unchanged.
            return batch

        max_prompt_length = getattr(self.interaction_config, 'max_prompt_length', 4096)

        encodings = shared_tokenizer(
            batch,
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = encodings["input_ids"]
        clipped_batch = shared_tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return clipped_batch
