from ..base_model import BaseModel
from ..optim_model import OptimizationModel
from ..tasks import BaseTask
from .base_search import BaseSearch
from .node import Node
import random
import numpy as np

OPERATOR_CHOICES = ["generation", "edit", "mix"]

class MPO(BaseSearch):
    def __init__(
        self,
        task: BaseTask,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        evaluator,
        log_dir: str,
        logger,
        method: str,
        beam_width: int,
        iteration: int,
        model_responses_num: int,
        **kwargs,
    ) -> None:
        super().__init__(
            base_model=base_model,
            optim_model=optim_model,
            log_dir=log_dir,
            logger=logger,
            method=method,
            iteration=iteration,
            model_responses_num=model_responses_num,
            **kwargs,
        )
        self.task = task
        self.evaluator = evaluator
        self.beam_width = beam_width

    def train(self):
        if self.method == "mpo":
            return self.optimize_mpo(self.task)
        else:
            raise ValueError(f"MPO class method {self.method} is not supported")

    def optimize_mpo(self, task: BaseTask):
        # Initialize first node
        node = Node(task.initial_prompt, task=task)
        self.evaluate_node(node=node, split="train")
        nodes_tracker = self.initialize_nodes_tracker(node)

        # First iteration of multimodal optimization
        inputs, action_types = self.get_action_types_and_inputs(it=-1, candidates=nodes_tracker["candidates"])

        self.logger.info(f"Action Types: {action_types}")
        batch_candidates = self._generate_nodes_parallel_pairs(
            inputs=inputs, action_types=action_types, node_generation_func=self.action
        )

        # Evaluate nodes using the evaluator
        self.evaluator(batch_candidates)

        self.update_candidates(nodes_tracker, batch_candidates)

        # Iterative multimodal optimization
        for it in range(self.iteration - 1):
            inputs, action_types = self.get_action_types_and_inputs(it=it, candidates=nodes_tracker["candidates"])
            self.logger.info(f"Action Types: {action_types}")

            batch_candidates = self._generate_nodes_parallel_pairs(
                inputs=inputs,
                action_types=action_types,
                node_generation_func=self.action,
            )

            # Evaluate nodes using the evaluator
            self.evaluator(batch_candidates)

            self.update_candidates(nodes_tracker, batch_candidates)

        # Log results
        self.log_node_tracker(nodes_tracker, filename=f"{task.task_name}")

        return nodes_tracker["updated"][-1][0].test_metric
        
    def get_action_types_and_inputs(self, it, candidates):
        action_types, inputs = [], []
        num_actions = self.beam_width**2

        if it == -1:
            action_types = ["generation"] * num_actions
            inputs = [[candidates[0]]] * num_actions
        else:
            train_metrics = np.array([node.train_metric for node in candidates])
            prob = train_metrics / train_metrics.sum()

            for i in range(num_actions):
                action_type = OPERATOR_CHOICES[i % len(OPERATOR_CHOICES)]
                action_types.append(action_type)
                if action_type in ["generation", "edit"]:
                    inputs.append([np.random.choice(candidates)])
                else:  # mix
                    parents = np.random.choice(candidates, size=2, p=prob, replace=False).tolist()
                    inputs.append(parents)

        return inputs, action_types

    def action(self, inputs: list[Node], action_type: str):
        if action_type == "generation":
            text_prompt, generated_mm_data = self.generation_action(inputs)
        elif action_type == "edit":
            text_prompt, generated_mm_data = self.edit_action(inputs)
        elif action_type == "mix":
            text_prompt, generated_mm_data = self.mix_action(inputs)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        new_node = Node(
            text_prompt,
            mm_prompt_path=generated_mm_data["mm_prompt_path"],
            mm_condition_prompt=generated_mm_data["mm_condition_prompt"],
            task=self.task,
            parents=inputs,
            action_type=action_type,
        )

        return new_node

    def generation_action(self, nodes):
        assert len(nodes) == 1
        node = nodes[0]

        text_prompt, generated_mm_data = self.optim_model.mpo_optim_generation(
            node=node,
            model_responses_num=self.model_responses_num,
        )

        return text_prompt, generated_mm_data

    def edit_action(self, nodes):
        assert len(nodes) == 1
        node = nodes[0]

        text_prompt, generated_mm_data = self.optim_model.mpo_optim_edit(
            node=node,
            model_responses_num=self.model_responses_num,
        )

        return text_prompt, generated_mm_data

    def mix_action(self, parents):
        assert len(parents) == 2
        text_prompt, generated_mm_data = self.optim_model.mpo_optim_mix(
            parents=parents,
            model_responses_num=self.model_responses_num,
        )

        return text_prompt, generated_mm_data
