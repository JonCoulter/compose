import json
from ..base_model import BaseModel
from ..optim_model import OptimizationModel
from .node import Node
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from typing import Callable


class BaseSearch(ABC):
    def __init__(
        self,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        log_dir: str,
        logger,
        method: str,
        iteration: int,
        model_responses_num: int,
        max_workers: int = 10,
        test_metric_evaluation_mode: str = "best",
        **kwargs,
    ) -> None:

        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model
        self.model_responses_num = model_responses_num
        self.method = method
        self.iteration = iteration
        self.log_dir = log_dir
        self.max_workers = max_workers
        self.test_metric_evaluation_mode = test_metric_evaluation_mode

    @abstractmethod
    def train(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_node(self, node: Node, split, num_examples: int = -1):
        if split not in ["train", "test"]:
            raise ValueError("Invalid split specified. Use 'train' or 'test'.")

        if split == "train":
            data = node.task.train_data[:num_examples] if num_examples != -1 else node.task.train_data
        else:
            data = node.task.test_data

        # Get model response and evaluation
        wrong_examples, correct_examples, metric = self.base_model.forward(data, node.instruction, node.mm_prompt_path)

        if split == "train":
            node.train_metric = metric
            node.update_model_correct_example(correct_examples)
            node.update_model_wrong_example(wrong_examples)
        if split == "test":
            node.test_metric = node.task.cal_all_metrics(wrong_examples + correct_examples)
            node.test_wrong_examples = wrong_examples
            node.test_correct_examples = correct_examples

        return None

    def _generate_nodes_parallel(
        self, nodes: list[Node], num_expand_per_node: int, node_generation_func: Callable[[Node], Node], **kwargs
    ):
        batch_candidates = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for node in nodes:
                for _ in range(num_expand_per_node):
                    future = executor.submit(node_generation_func, node, **kwargs)
                    futures.append(future)

            for future in as_completed(futures):
                try:
                    new_node = future.result()
                    if new_node is not None:
                        batch_candidates.append(new_node)
                except Exception as e:
                    self.logger.error(f"Error generating node: {e}")

        return batch_candidates

    def _generate_nodes_parallel_pairs(
        self, inputs: list[Node], action_types: list[str], node_generation_func: Callable[[Node, str], Node], **kwargs
    ):
        batch_candidates = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for input, action_type in zip(inputs, action_types):
                future = executor.submit(node_generation_func, input, action_type, **kwargs)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    new_node = future.result()
                    if new_node is not None:
                        batch_candidates.append(new_node)
                except Exception as e:
                    self.logger.error(f"Error generating node: {e}")

        return batch_candidates

    def update_candidates(self, nodes_tracker, new_nodes):
        """Helper function to update candidate pool and tracking nodes"""
        nodes_tracker["candidates"].extend(new_nodes)
        nodes_tracker["candidates"].sort(key=lambda x: x.train_metric, reverse=True)
        nodes_tracker["candidates"] = nodes_tracker["candidates"][: self.beam_width]
        nodes_tracker["updated"].append(nodes_tracker["candidates"][:])
        nodes_tracker["total"].append(new_nodes[:])

    def initialize_nodes_tracker(self, node):
        return {
            "candidates": [node],  # Current best candidates for next iteration
            "updated": [[node]],  # All nodes in order of updates
            "total": [[node]],  # All nodes ever created
        }

    def evaluate_test_nodes(self, nodes_tracker):
        self.logger.info(f"======== Evaluate test metric ========")

        if self.test_metric_evaluation_mode == "total":
            for nodes in nodes_tracker["total"]:
                for node in nodes:
                    if node.test_metric == -1:
                        self.evaluate_node(node=node, split="test")
        elif self.test_metric_evaluation_mode == "updated":
            for nodes in nodes_tracker["updated"]:
                for node in nodes:
                    if node.test_metric == -1:
                        self.evaluate_node(node=node, split="test")
        elif self.test_metric_evaluation_mode == "best":
            self.evaluate_node(nodes_tracker["updated"][-1][0], split="test")
        else:
            raise ValueError(f"Invalid test metric evaluation mode: {self.test_metric_evaluation_mode}")

    def log_node_tracker(self, nodes_tracker, filename):
        """
        Logs the nodes tracked in nodes_tracker.
        """
        self.evaluate_test_nodes(nodes_tracker)

        # save nodes_tracker in pickle file for better analysis
        with open(f"{self.log_dir}/nodes_tracker.pkl", "wb") as f:
            pickle.dump(nodes_tracker, f)

        # Prepare results data
        updated_nodes_data = [[node.to_dict() for node in nodes] for nodes in nodes_tracker["updated"]]
        total_nodes_data = [[node.to_dict() for node in nodes] for nodes in nodes_tracker["total"]]

        # Find best test node
        train_best_node = nodes_tracker["updated"][-1][0]

        test_best_node = None
        for nodes in nodes_tracker["total"]:
            for node in nodes:
                if isinstance(node.test_metric, dict):
                    if test_best_node is None or node.test_metric["target"] > test_best_node.test_metric["target"]:
                        test_best_node = node

        # Save results
        data = {
            "train_best_node": updated_nodes_data[-1][0],  # Get the best train node from nodes_data
            "test_best_node": test_best_node.to_dict(),
            "nodes_data": updated_nodes_data,
            "total_nodes_data": total_nodes_data,
            "train_best_wrong_examples": train_best_node.test_wrong_examples,
            "train_best_correct_examples": train_best_node.test_correct_examples,
        }

        self.save_data(data, filename=f"{filename}")

    def save_data(self, data, filename: str = "nodes"):
        log_dir_parts = self.log_dir.split(os.sep)
        if len(log_dir_parts) > 1:
            log_dir_parts.pop()
        new_log_dir = os.sep.join(log_dir_parts)

        version = 0
        base_filename = f"{filename}_{version}.json"
        full_path = os.path.join(new_log_dir, base_filename)

        while os.path.exists(full_path):
            version += 1
            base_filename = f"{filename}_{version}.json"
            full_path = os.path.join(new_log_dir, base_filename)

        with open(full_path, "w") as file:
            json.dump(data, file, indent=4)

        self.logger.info(f"Save log: {full_path}")
