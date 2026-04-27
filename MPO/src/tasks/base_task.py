# define task prompts for various datasets
import numpy as np
from abc import ABC, abstractmethod



class BaseTask(ABC):
    def __init__(
        self,
        train_size,
        test_size,
        task_name,
        data_dir="./dataset",
        seed=None,
        benchmark=None,
        **kwargs,
    ):
        self.task_name = task_name
        self.data_dir = data_dir
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.benchmark = benchmark
        self.initial_prompt = None
        self.dataset = self.load_task_dataset()
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]

        print(f"benchmark : {self.benchmark}")
        print(f"task : {self.task_name}")
        print(f"train_data : {len(self.train_data)}")
        print(f"test_data : {len(self.test_data)}")

    @abstractmethod
    def load_task_dataset(self):
        '''
        Load the dataset for the task.
        Returns a dictionary containing two keys: 'train' for training data and 'test' for testing data.
        '''
        pass

    @abstractmethod
    def get_initial_prompt(self):
        '''
        Get the initial prompt for the task.
        '''
        pass

    @abstractmethod
    def get_query(self, example):
        '''
        Get the query for the task.
        '''
        pass

    @abstractmethod
    def get_answer(self, example):
        '''
        Get the answer for the task.
        '''
        pass
    
    @abstractmethod
    def get_mm_path(self, example):
        '''
        Retrieve the file path for the image associated with the given example.
        '''
        pass
    
    @abstractmethod
    def _clean_response(self, example):
        '''
        Extract the answer from the model's response.
        If the answer is not formatted correctly, return 'Answer Format Error'.
        '''
        pass
    
    def clean_responses(self, examples):
        return [self._clean_response(example) for example in examples]

    @abstractmethod
    def _cal_correct(self, examples):
        pass

    def cal_corrects(self, examples):
        return [self._cal_correct(example) for example in examples]


    def cal_metric(self, examples):
        correct = [example['correct'] for example in examples]
        return round(np.mean(correct), 3)
    
    def cal_all_metrics(self, examples):
        metrics = {
            "acc": self.cal_metric(examples),
            "target": self.cal_metric(examples)
        }
        return metrics