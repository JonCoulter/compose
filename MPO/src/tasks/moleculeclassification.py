# define task prompts for various datasets
from .base_task import BaseTask
import os
import json
import random
import re
import numpy as np
from sklearn.metrics import f1_score

class MoleculeClassification(BaseTask):
    def __init__(
        self,
        train_size,
        test_size,
        task_name,
        data_dir="./dataset",
        seed=None,
        benchmark=None,
        logger=None,
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
        
        self.logger = logger
        self.mm_type = "molecule"
        self.initial_prompt, self.labels = self.get_metadata()

        # Calculate label distribution
        self.print_label_distribution(self.train_data, "Train")
        self.print_label_distribution(self.test_data, "Test")
        
        self.cal_metric = self.cal_metric_f1
        self.print_if_skewed(self.train_data, "Train")
        self.print_if_skewed(self.test_data, "Test")

    def load_task_dataset(self):
        file_path = f"{self.data_dir}/moleculeclassification/{self.task_name}.json"

        raw_data = json.load(open(file_path, "r"))
        split = raw_data['split']

        train_data = [raw_data['data_list'][i] for i in split['train']]
        for data in train_data:
            data['question'] = "\n" + data['question'].replace("\nMolecule: <mol>", '')
        test_data = [raw_data['data_list'][i] for i in split['test']]
        for data in test_data:
            data['question'] = "\n" + data['question'].replace("\nMolecule: <mol>", '')

        # Shuffle test data with fixed seed
        random.seed(42)
        random.shuffle(test_data)

        # Shuffle train data based on seed if provided
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(train_data)

        train_data = train_data[: self.train_size]
        if self.test_size:
            test_data = test_data[: self.test_size]

        answer_format = raw_data['answer_format']
        for example in train_data + test_data:
            example['question'] += "\n" + answer_format + "\n\nTarget Molecule: "

        split_data = {"train": train_data, "test": test_data}

        return split_data

    def get_metadata(self):
        file_path = f"{self.data_dir}/moleculeclassification/{self.task_name}.json"
        raw_data = json.load(open(file_path, "r"))
        
        initial_prompt = raw_data['prompt']
        labels = raw_data['labels']

        return initial_prompt, labels
        
    def get_query(self, example):
        return example['question']

    def get_answer(self, example):
        return example['answer']
    
    def get_mm(self, example):
        atoms = example['atoms']
        coordinates = example['coordinates']
        smiles = example['smiles']

        return {
            "atoms": atoms,
            "coordinates": coordinates,
            "smiles": smiles
        }
    
    def get_initial_prompt(self):
        pass

    def get_mm_path(self, example):
        return self.get_mm(example)
    
    def _clean_response(self, example):
        response = example['response'].lower()

        answer_pattern = r"\**([Ff]inal )*[Aa]nswer\**:"
        match = re.search(answer_pattern, response)
        model_answer = None
        if match:
            extracted_answer = response[match.end():].strip()
            for label in self.labels:
                if label.lower() in extracted_answer:
                    model_answer = label
                    break 

        example['model_answer'] = model_answer
        return example        
    
    def clean_responses(self, examples):
        return [self._clean_response(example) for example in examples]

    def _cal_correct(self, example):
        if example['model_answer'] is None:
            example['correct'] = False
        else:
            example['correct'] = self.get_answer(example).lower() == example['model_answer'].lower()
        return example

    def cal_corrects(self, examples):
        return [self._cal_correct(example) for example in examples]

    def cal_metric_acc(self, examples):
        correct = [example['correct'] for example in examples]
        return round(np.mean(correct), 4)

    def cal_metric_f1(self, examples):
        labels = [self.get_answer(example) for example in examples]
        preds = [example['model_answer'] for example in examples]
        preds = [pred if pred is not None else "unknown" for pred in preds]

        # Calculating F1 Score
        f1 = f1_score(labels, preds, average="macro", labels=np.unique(labels))
        return round(f1, 4)

    def cal_all_metrics(self, examples):
        metrics = {
            "acc": self.cal_metric_acc(examples),
            "f1": self.cal_metric_f1(examples),
            "target": self.cal_metric(examples)
        }
        return metrics

    def print_label_distribution(self, data, data_type):
        label_counts = {}
        for example in data:
            label = example["answer"].lower()
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        assert set([label.lower() for label in self.labels]) ==  set(label_counts.keys()), \
            f"Labels mismatch: {set([label.lower() for label in self.labels])} != {set(label_counts.keys())}"

        if self.logger is not None:
            self.logger.info(f"\n{data_type} Data Label Distribution:")
            for label, count in label_counts.items():
                self.logger.info(f"{label}: {count} ({count/len(data)*100:.2f}%)")
    
    def print_if_skewed(self, data, data_type):
        # Compute metrics if the prediction is skewed to one label
        label_counts = {}
        for example in data:
            label = example["answer"].lower()
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        total_count = len(data)


        skewed_label = max(label_counts, key=label_counts.get)
        labels = [self.get_answer(example).lower() for example in data]
        preds = [skewed_label] * total_count
        f1 = f1_score(labels, preds, average="macro")
        if self.logger is not None:
            self.logger.info(f"\n{data_type} Data is skewed to label '{skewed_label}' with F1 score: {f1:.4f}")
        