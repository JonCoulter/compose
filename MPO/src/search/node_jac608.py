import itertools
import random
from typing import Optional, List
from ..tasks import BaseTask
import os


class Node:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        instruction: str,
        mm_prompt_path: str = None,
        mm_condition_prompt: str = None,
        task: BaseTask = None,
        parents: Optional[List["Node"]] = None,
        train_metric: float = -1,
        test_metric: float = -1,
        action_type: str = None,
    ):

        self.id = next(Node.id_iter)
        self.instruction = instruction
        self.parents = parents
        self.train_metric = train_metric
        self.test_metric = test_metric
        self.mm_prompt_path = mm_prompt_path
        self.mm_condition_prompt = mm_condition_prompt
        self.action_type = action_type
        
        if parents is None:
            self.depth = 0
            assert task is not None
            self.task = task
        else:
            self.depth = max(parent.depth for parent in parents) + 1
            self.task = parents[0].task

    def update_model_wrong_example(self, examples):
        self.model_wrong_examples = []
        self.model_wrong_examples.extend(examples)

    def update_model_correct_example(self, examples):
        self.model_correct_examples = []
        self.model_correct_examples.extend(examples)

    def get_wrong_examples(self, model_responses_num: int):
        num_wrong_examples = len(self.model_wrong_examples)

        if num_wrong_examples < model_responses_num:
            sampled_examples = self.model_wrong_examples
        else:
            sampled_examples = random.sample(self.model_wrong_examples, model_responses_num)

        return sampled_examples

    def to_dict(self):
        if self.mm_prompt_path is not None and isinstance(self.mm_prompt_path, str):
            log_mm_prompt_path = os.path.abspath(self.mm_prompt_path)
        else:
            log_mm_prompt_path = self.mm_prompt_path
            
        return {
            "id": self.id,
            "task": self.task.task_name,
            "instruction": self.instruction,
            "mm_prompt_path": log_mm_prompt_path,
            "parent_id": [parent.id for parent in self.parents] if self.parents else None,
            "depth": self.depth,
            "train_metric": self.train_metric,
            "test_metric": self.test_metric,
            "mm_condition_prompt": self.mm_condition_prompt,  
            "action_type": self.action_type,
        }
    
