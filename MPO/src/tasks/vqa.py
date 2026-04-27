from .base_task import BaseTask
import re
import os
import json
import random

TEST_SHUFFLE_SEED = 42

TASK_CONFIGS = {
    "rsvqa": {
        "mm_file_key": "img_id",
        "answer_key": "answer",
        "mm_ext": ".jpg"
    },
    
    'drivingvqa': {
        "mm_file_key": "img_filename",
        "answer_key": "answer",
        "question_key": "question",
    },

    # Slake tasks
    'MRI': {
        "mm_file_key": "img_name",
    },
    'CT': {
        "mm_file_key": "img_name",
    },
    'X-Ray': {
        "mm_file_key": "img_name",
    },
}

SUBTASKS_SLAKE = ['MRI', 'CT', 'X-Ray'] 
OPEN_ENDED_TASKS = SUBTASKS_SLAKE + ['rsvqa'] # Open-ended VQA.
CLOSED_ENDED_TASKS = ['drivingvqa'] # Closed-ended VQA.

class VQA(BaseTask):
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
        self.mm_file_key = self.task_config.get("mm_file_key", "image_path")
        self.answer_key = self.task_config.get("answer_key", "answer")
        self.mm_ext = self.task_config.get("mm_ext", "")
        self.question_key = self.task_config.get("question_key", "question")
        self.train_size = train_size
        self.test_size = test_size

    def get_file_name(self):
        if self.task_name in SUBTASKS_SLAKE:
            train_file = f"{self.data_dir}/{self.benchmark}/slake/{self.task_name}_train.json"
            test_file = f"{self.data_dir}/{self.benchmark}/slake/{self.task_name}_test.json"
        else:
            train_file = f"{self.data_dir}/{self.benchmark}/{self.task_name}/train.json"
            test_file = f"{self.data_dir}/{self.benchmark}/{self.task_name}/test.json"

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
        if self.task_name in CLOSED_ENDED_TASKS:
            return 'Given the image, answer the following question.'
        else:
            return 'Given the image, answer the following question.'

    def get_answer(self, example):
        return example[self.answer_key]

    def get_query(self, example):
        if self.task_name in CLOSED_ENDED_TASKS:
            return f"{example[self.question_key]} Your response should end with \"The answer is [answer]\". Target image:"
        else:
            return f"{example[self.question_key]} Your response should be concise without an explanation. Target image:"

    def get_mm_path(self, example):
        if self.task_name in SUBTASKS_SLAKE:
            image_path = f"{self.data_dir}/{self.benchmark}/slake/images/{example[self.mm_file_key]}"
        else:
            image_path = f"{self.data_dir}/{self.benchmark}/{self.task_name}/images/{example[self.mm_file_key]}{self.mm_ext}"

        return image_path

    def _clean_response(self, example):
        response = example["response"].strip()

        if self.task_name in CLOSED_ENDED_TASKS: # Multi choice
            pattern = r"answer is ([a-zA-Z0-9]+)"
            match = re.search(pattern, response)

            if not match:
                pattern = r"answer is \(([A-Za-z]+)\)"
                match = re.search(pattern, response)

            if match:
                model_answer = match.group(1).replace('"','')
                example["model_answer"] = model_answer
            else:
                example['model_answer'] = "Format error"

        else: # Exact match
            response = re.sub(r'[.,!?]', '', response)
            example["model_answer"] = response

        return example

    def _cal_correct(self, example):
        gt_answer = self.get_answer(example).strip().lower()
        gt_answer = re.sub(r'[.,!?]', '', gt_answer)
        model_answer = example["model_answer"].strip().lower()
        example["correct"] = gt_answer == model_answer

        return example



