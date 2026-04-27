from .base_task import BaseTask
import re
import os
import json
import random

TEST_SHUFFLE_SEED = 42

TASK_CONFIGS = {
    "vanebench": {},
    "vane_ai": {},
    "vane_real": {},
}

VANE_TASKS = ["vane_ai", "vane_real"]

class VideoVQA(BaseTask):
    def __init__(
        self,
        task_name: str,
        train_size: int,
        test_size: int,
        data_dir="",
        seed=None,
        benchmark="vqa",
        **kwargs,
    ):
        self.task_config = TASK_CONFIGS[task_name]

        super().__init__(
            task_name=task_name,
            train_size=train_size,
            test_size=test_size,
            data_dir=data_dir,
            seed=seed,
            benchmark=benchmark,
            **kwargs,
        )

        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}")

        self.options = {}
        self.benchmark = benchmark
        self.task_name = task_name
        self.initial_prompt = self.get_initial_prompt()
        self.mm_file_key = self.task_config.get("mm_file_key", "video_path")
        self.answer_key = self.task_config.get("answer_key", "answer")
        self.question_key = self.task_config.get("question_key", "question")
        self.train_size = train_size
        self.test_size = test_size

    def get_file_name(self):
        if self.task_name in VANE_TASKS:
            train_file = f"{self.data_dir}/video/vanebench/{self.task_name}_train.json"
            test_file = f"{self.data_dir}/video/vanebench/{self.task_name}_test.json"
        else:
            train_file = f"{self.data_dir}/video/{self.task_name}/train.json"
            test_file = f"{self.data_dir}/video/{self.task_name}/test.json"

        return train_file, test_file

    def load_task_dataset(self):
        train_file, test_file = self.get_file_name()

        if not os.path.exists(test_file) or not os.path.exists(train_file):
            raise ValueError(f"json files {test_file} or {train_file} do not exist.")

        with open(test_file, "r") as file:
            test_data = json.load(file)
        with open(train_file, "r") as file:
            train_data = json.load(file)

        # Shuffle test data with fixed seed
        random.seed(TEST_SHUFFLE_SEED)
        random.shuffle(test_data)

        # Shuffle train data based on seed if provided
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(train_data)

        # Add examples to train, eval and test sets using list slicing
        train_data = train_data[: self.train_size]
        if self.test_size:
            test_data = test_data[: self.test_size]

        # Create final data dictionary
        split_data = {"train": train_data, "test": test_data}

        return split_data

    def get_initial_prompt(self):
        return (
            'Given the video, answer the following question.\n'
        )

    def get_answer(self, example):
        return example[self.answer_key]

    def get_query(self, example):
        return example[self.question_key] + "\nYour response should end with \"The answer is [answer]\". Target video:"

    def get_mm_path(self, example):
        if self.task_name in VANE_TASKS:
            video_path = f"{self.data_dir}/video/vanebench/{example[self.mm_file_key]}"
        elif self.task_name == 'vanebench':
            video_path = f"{self.data_dir}/video/{self.task_name}/{example[self.mm_file_key]}"
        else:
            video_path = f"{self.data_dir}/video/{self.task_name}/videos/{example[self.mm_file_key]}"

        return video_path

    def _clean_response(self, example):
        response = example["response"].strip()

        pattern = r"answer is ([a-zA-Z]+)"
        match = re.search(pattern, response)

        if not match:
            pattern = r"answer is \(([A-Za-z]+)\)"
            match = re.search(pattern, response)

        if match:
            model_answer = match.group(1).replace('"', "")
            example["model_answer"] = model_answer
        else:
            example["model_answer"] = "Format error"

        return example

    def _cal_correct(self, example):
        gt_answer = self.get_answer(example).strip().lower()
        gt_answer = re.sub(r"[.,!?]", "", gt_answer)
        model_answer = example["model_answer"].strip().lower()
        example["correct"] = gt_answer == model_answer

        return example
