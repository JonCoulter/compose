from .base_task import BaseTask
from .classification import Classification
import os
import json
import random

TEST_SHUFFLE_SEED = 42

TASK_CONFIGS = {
    "hummingbird": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "albatross": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "bunting": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "jay": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "cuckoo": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "cormorant": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "swallow": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "blackbird": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "auklet": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "grosbeak": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "oriole": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
    "grebe": {"labels": None, "mm_file_key": "filename", "answer_key": "label"},
}


class CUB(Classification):
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
        super().__init__(
            task_name=task_name,
            train_size=train_size,
            test_size=test_size,
            data_dir=data_dir,
            seed=seed,
            benchmark=benchmark,
            **kwargs,
        )
        self.labels = self.get_labels()

    def load_task_dataset(self):
        test_file = f"{self.data_dir}/{self.benchmark}/cub/{self.task_name}_test.json"
        train_file = f"{self.data_dir}/{self.benchmark}/cub/{self.task_name}_train.json"

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

    def get_labels(self):
        # Count the frequency of each label in train_data
        label_count = {}
        for item in self.test_data:
            label = item[self.answer_key]
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        # Sort labels by frequency in descending order
        sorted_labels = sorted(label_count.items(), key=lambda x: (-x[1], x[0]))

        labels = [label for label, _ in sorted_labels]

        print(f"Class Num : {len(labels)} labels: {labels}")
        return labels

    def get_mm_path(self, example):
        image_path = f"{self.data_dir}/{self.benchmark}/cub/images/{example[self.mm_file_key]}"
        return image_path
