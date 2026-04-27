from .model import get_language_model
from .tasks import BaseTask
import os
import random
from .utils import check_mm_type


class BaseModel:
    def __init__(self, base_model_setting: dict, task: BaseTask, logger):
        self.model = get_language_model(base_model_setting["model_name"])(**base_model_setting)
        self.task = task
        self.logger = logger
        self.debug_output = base_model_setting["debug_output"]

    def forward(self, examples, user_prompt, mm_prompt_path=None):
        batch_prompts = self._build_forward_prompts_completion(examples, user_prompt, mm_prompt_path)
        responses = self.model.batch_forward_func(batch_prompts)
        examples = [dict(example, response=response) for example, response in zip(examples, responses)]
        examples = self.task.clean_responses(examples)
        examples = self.task.cal_corrects(examples)
        metric = self.task.cal_metric(examples)
        self._log_forward_output(user_prompt, examples, metric, mm_prompt_path)
        if self.debug_output:
            self._log_examples(user_prompt, examples, mm_prompt_path)

        wrong_examples, correct_examples = self._split_examples_by_correctness(examples=examples)

        return wrong_examples, correct_examples, metric

    def forward_prompts(self, examples, user_prompts: list[str], mm_prompt_paths: list[str] = None):
        if mm_prompt_paths is None:
            mm_prompt_paths = [None] * len(user_prompts)

        all_prompts, prompt_slices = self._build_all_prompts(examples, user_prompts, mm_prompt_paths)
        responses = self.model.batch_forward_func(all_prompts)

        results = {
            "wrong_examples": [],
            "correct_examples": [],
            "metrics": [],
            "example_corrects": [],
        }

        for start_idx, end_idx, user_prompt, mm_path in prompt_slices:
            prompt_responses = responses[start_idx:end_idx]
            examples = [dict(example, response=response) for example, response in zip(examples, prompt_responses)]
            examples = self.task.clean_responses(examples)
            examples = self.task.cal_corrects(examples)
            metric = self.task.cal_metric(examples)
            self._log_forward_output(user_prompt, examples, metric, mm_path)
            if self.debug_output:
                self._log_examples(user_prompt, examples, mm_path)
            wrong_examples, correct_examples = self._split_examples_by_correctness(examples)
            results["wrong_examples"].append(wrong_examples)
            results["correct_examples"].append(correct_examples)
            results["metrics"].append(metric)
            results["example_corrects"].append([ex["correct"] for ex in examples])

        return results

    def forward_nodes(self, examples, nodes: list):
        if not nodes:
            return {"wrong_examples": [], "correct_examples": [], "metrics": [], "example_corrects": []}

        user_prompts = [node.instruction for node in nodes]
        mm_prompt_paths = [node.mm_prompt_path for node in nodes]
        fp_outputs = self.forward_prompts(examples, user_prompts, mm_prompt_paths if any(mm_prompt_paths) else None)

        for node, wrong_examples, correct_examples in zip(
            nodes, fp_outputs["wrong_examples"], fp_outputs["correct_examples"]
        ):
            node.update_model_wrong_example(wrong_examples)
            node.update_model_correct_example(correct_examples)

        return fp_outputs

    def _log_examples(self, user_prompt, examples, mm_prompt_path=None, debug_num=1):
        self.logger.info("---------------   Examples   -----------------")
        sampled_examples = random.sample(examples, debug_num)
        for example in sampled_examples:
            mm_path = self.task.get_mm_path(example)
            if isinstance(mm_path, dict):
                mm_info = mm_path["smiles"][0]
            else:
                mm_info = os.path.abspath(mm_path)
            self.logger.info(
                f"Input: {self.task.get_query(example)}\n{mm_info}\n\nResponse: {example['response']}\n\nModel Answer: {example['model_answer']}\nAnswer: {self.task.get_answer(example)}\nCorrect: {example['correct']}\n-----\n"
            )

    def _log_forward_output(self, user_prompt, examples, metric, mm_prompt_path=None):
        log_mm_prompt_path = (
            "None"
            if mm_prompt_path is None
            else (mm_prompt_path["smiles"][0] if isinstance(mm_prompt_path, dict) else os.path.abspath(mm_prompt_path))
        )
        forward_log_output = forward_log_template.format(
            task_name=self.task.task_name,
            user_prompt=user_prompt,
            num_examples=len(examples),
            metric=metric,
            mm_prompt_path=log_mm_prompt_path,
        )
        self.logger.info(forward_log_output)

    def _split_examples_by_correctness(self, examples):
        wrong_examples = [example for example in examples if example["correct"] == 0]
        correct_examples = [example for example in examples if example["correct"] == 1]
        return wrong_examples, correct_examples

    def _build_forward_prompts_completion(self, examples, user_prompt, mm_prompt_path=None):
        prompts = []
        for example in examples:
            content = [{"type": "text", "text": user_prompt}]
            if mm_prompt_path:
                mm_type = check_mm_type(mm_prompt_path)
                if mm_type == "image":
                    content.append({"type": "text", "text": "\n\nReference Image: "})
                elif mm_type == "molecule":
                    content.append({"type": "text", "text": "\n\nReference Molecule: "})
                content.append({"type": mm_type, mm_type: mm_prompt_path})

            content.append({"type": "text", "text": self.task.get_query(example)})
            mm_query_path = self.task.get_mm_path(example)
            mm_query_type = check_mm_type(mm_query_path)
            content.append({"type": mm_query_type, mm_query_type: mm_query_path})
            prompts.append([{"role": "user", "content": content}])
        return prompts

    def _build_all_prompts(self, examples, user_prompts, mm_prompt_paths):
        all_prompts = []
        prompt_slices = []
        for user_prompt, mm_path in zip(user_prompts, mm_prompt_paths):
            start_idx = len(all_prompts)
            prompts_for_this_prompt = self._build_forward_prompts_completion(examples, user_prompt, mm_path)
            all_prompts.extend(prompts_for_this_prompt)
            end_idx = len(all_prompts)
            prompt_slices.append((start_idx, end_idx, user_prompt, mm_path))
        return all_prompts, prompt_slices


forward_log_template = """---------------\tModel Output\t----------------
task_name: {task_name}
user_prompt:\n{user_prompt}

mm_prompt_path: {mm_prompt_path}

num_examples: {num_examples}
metric:     {metric}
"""
