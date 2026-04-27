from .base_task import BaseTask
from .classification import Classification
import os
import json
import random

TEST_SHUFFLE_SEED = 42

# fmt: off
TASK_CONFIGS = {
    "Apple": {'labels': ["Cedar_apple_rust", "Black_rot", "Apple_scab", "healthy"]},
    "Cherry": {'labels': ["healthy", "Powdery_mildew"]},
    "Corn": {'labels': ["Cercospora_leaf_spot_and_Gray_leaf_spot", "Northern_Leaf_Blight", "Common_rust", "healthy"]},
    "Grape": {'labels': ["healthy", "Leaf_blight", "Black_rot", "Esca"]},
    "Peach": {'labels': ["Bacterial_spot", "healthy"]},
    "Pepper_bell": {'labels': ["healthy", "Bacterial_spot"]},
    "Potato": {'labels': ["healthy", "Early_blight", "Late_blight"]},
    "Strawberry": {'labels': ["healthy", "Leaf_scorch"]},
    "Tomato": {'labels': ["Tomato_Yellow_Leaf_Curl_Virus", "Bacterial_spot", "Leaf_Mold", "healthy", "Target_Spot", "Early_blight", "Spider_mites Two-spotted_spider_mite", 
                          "Septoria_leaf_spot", "Tomato_mosaic_virus", "Late_blight"]}
}
# fmt: on

class PlantVillage(Classification):
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

    def load_task_dataset(self):
        test_file = f"{self.data_dir}/{self.benchmark}/plantvillage/{self.task_name}_test.json"
        train_file = f"{self.data_dir}/{self.benchmark}/plantvillage/{self.task_name}_train.json"

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

    def get_mm_path(self, example):
        image_path = f"{self.data_dir}/{self.benchmark}/plantvillage/images/{example[self.mm_file_key]}"
        return image_path