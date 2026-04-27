from .base_task import BaseTask
from .classification import Classification
import os
import json
import random

TEST_SHUFFLE_SEED = 42

# fmt: off
TASK_CONFIGS = {
    "driveact" :{'labels':None, 'mm_file_key': 'video_file_name', 'answer_key': 'activity'},
    "ucfcrime" :{'labels':None, 'mm_file_key': 'video_file_name', 'answer_key': 'label'},
}
# fmt: on

class Video(Classification):
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
        test_file = f"{self.data_dir}/video/{self.task_name}/test.json"
        train_file = f"{self.data_dir}/video/{self.task_name}/train.json"

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
        initial_prompt = f"Given the video, answer the following question."
        return initial_prompt

    def get_query(self, example):
        return f"Classify the content of the target video.\nChoices: {str(self.labels)}\nTarget Video:"

    def get_labels(self):
        label_count = {}
        for item in self.test_data:
            answer = item[self.answer_key]
            labels = answer if isinstance(answer, list) else [answer]
            for label in labels:
                label_count[label] = label_count.get(label, 0) + 1

        # Sort labels by frequency in descending order
        sorted_labels = sorted(label_count.items(), key=lambda x: (-x[1], x[0]))

        labels = [label for label, _ in sorted_labels]

        print(f"Class Num : {len(labels)} labels: {labels}")
        return labels

    def get_mm_path(self, example):
        video_path = f"{self.data_dir}/video/{self.task_name}/videos/{example[self.mm_file_key]}"
        return video_path
