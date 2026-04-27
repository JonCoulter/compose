from .base_task import BaseTask
import re
import os
import json
import random
from sklearn.metrics import f1_score
import numpy as np

TEST_SHUFFLE_SEED = 42

TASK_CONFIGS = {}

class Classification(BaseTask):
    TASK_CONFIGS = TASK_CONFIGS

    def __init__(
        self,
        task_name: str,
        train_size: int,
        test_size: int,
        data_dir="",
        seed=None,
        benchmark="classification",
        **kwargs,
    ):
        self.task_config = self.TASK_CONFIGS[task_name]
        self.labels = self.task_config["labels"]
        self.mm_file_key = self.task_config.get("mm_file_key", "image_path")
        self.answer_key = self.task_config.get("answer_key", "answer")
        self.image_ext = self.task_config.get("image_ext", "")

        super().__init__(
            task_name=task_name,
            train_size=train_size,
            test_size=test_size,
            data_dir=data_dir,
            seed=seed,
            benchmark=benchmark,
            **kwargs,
        )

        self.options = {}
        self.benchmark = benchmark
        self.task_name = task_name
        self.initial_prompt = self.get_initial_prompt()
        self.train_size = train_size
        self.test_size = test_size

    def load_task_dataset(self):
        test_file = f"{self.data_dir}/{self.benchmark}/{self.task_name}/test.json"
        train_file = f"{self.data_dir}/{self.benchmark}/{self.task_name}/train.json"

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

        # Calculate label distribution
        self.print_label_distribution(train_data, "Train")
        self.print_label_distribution(test_data, "Test")

        # Create final data dictionary
        split_data = {"train": train_data, "test": test_data}

        return split_data

    def get_initial_prompt(self):
        initial_prompt = f"Given the image, answer the following question."
        return initial_prompt

    def get_answer(self, example):
        return example[self.answer_key]

    def get_query(self, example):
        return f"Classify the content of the target image. Choices: {str(self.labels)}\nTarget Image:"

    def get_mm_path(self, example):
        image_path = f"{self.data_dir}/{self.benchmark}/{self.task_name}/images/{example[self.mm_file_key]}"
        image_path += self.image_ext
        return image_path

    def _clean_response(self, example):
        response = example["response"].lower()
        last_match = None
        last_pos = -1  

        for label in self.labels:
            pos = response.rfind(label.lower()) 
            if pos != -1 and pos > last_pos:
                last_match = label
                last_pos = pos

        if last_match:
            example["model_answer"] = last_match # Select the last match answer.
        else:
            example["model_answer"] = "None"
        return example

    def _cal_correct(self, example):
        example["correct"] = self.get_answer(example).lower() == example["model_answer"].lower()
        return example

    def print_label_distribution(self, data, data_type):
        label_counts = {}
        for example in data:
            label = example[self.answer_key].lower()
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        print(f"\n{data_type} Data Label Distribution:")
        for label, count in label_counts.items():
            print(f"{label}: {count} ({count/len(data)*100:.1f}%)")

    def cal_metric_acc(self, examples):
        correct = [example['correct'] for example in examples]
        return round(np.mean(correct), 4)

    def cal_metric_f1(self, examples):
        labels = [self.get_answer(example) for example in examples]
        preds = [example['model_answer'] for example in examples]
        preds = [pred if pred != "None" else "None" for pred in preds]

        # Calculating F1 Score
        f1 = f1_score(labels, preds, average="macro", labels=np.unique(labels))
        return round(f1, 4)
    
    def cal_all_metrics(self, examples):
        metrics = {
            "acc": self.cal_metric_acc(examples),
            "f1": self.cal_metric_f1(examples),
            "target": self.cal_metric_acc(examples),
        }
        return metrics