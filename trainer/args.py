import os
import transformers
from typing import Optional
from dataclasses import field, dataclass
from transformers.utils import requires_backends


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seed: int = field(
        default=1,
        metadata={
            "help": "Random seed for this run; independent of released member IDs."
        },
    )
    dataset: str = field(default="cmip6", metadata={"help": "Training dataset."})
    lr_scheduler_type_custom: str = field(default=None)
    min_learning_rate: float = field(
        default=0.0, metadata={"help": "The minimum of learning rate."}
    )
    inputs_key_for_metrics: Optional[str] = field(
        default=None,
        metadata={
            "help": "The key in dictionary of inputs that will be passed to the `compute_metrics` function."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.include_inputs_for_metrics and self.inputs_key_for_metrics is None:
            raise ValueError(
                "--include_inputs_for_metrics requires inputs_key_for_metrics not to be None"
            )

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        requires_backends(self, ["torch"])
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"])
        elif "SLURM_PROCID" in os.environ:
            return int(os.environ["SLURM_NTASKS"])
        else:
            return 1
