import importlib
from .base_task import BaseTask

TASK_CATEGORIES = {
    "MathVision": [
        "algebra", "combinatorics", "combinatorial_geometry", "metric_geometry_area",
        "metric_geometry_angle", "metric_geometry_length", "transformation_geometry",
        "solid_geometry"
    ],
    "VLICL": ["textocr"],
    "ChartQA": ["chartqa"],
    "Unsafe": [
        "deception", "harassment", "hate", "illegal_activity", "political",
        "public_and_personal_health", "self-harm", "sexual", "shocking", "spam", "violence"
    ],
    "VizWiz": ["vizwiz"],
    "Clevr": ["clevr"],
    "Classification": [
        "ham10000", "fives", "eurosat", "beans", "rice", "ripeness", "plantvillage",
        "cotton", "ucmersed", "stanford_pet", "alzheimer", "cat_dog"
    ],
    "PlantVillage": [
        "Apple", "Cherry", "Corn", "Grape", "Peach", "Pepper_bell", "Potato",
        "Strawberry", "Tomato"
    ],
    "CUB": [
        "hummingbird", "albatross", "bunting", "jay", "cuckoo", "cormorant", # 3 class
        "swallow", "blackbird", "auklet", "grosbeak", "oriole", "grebe", # 4 class
    ],

    "Classnum": ["butterfly", "wikiart"],

    "VQA": [
        "rsvqa", "vqarad", "drivingvqa", 
        "MRI", "CT", "X-Ray", # SLAKE
        "Count", "Relation", "Depth", "Distance" # CVBENCH
    ],
    "SCAM": ["scam"],
    "AudioClassification": ["dogs"],
    "Video": ["driveact", "ucfcrime"],
    "VideoVQA": ["vanebench", "vane_ai", "vane_real"],
    "AnimalKingdom": [
        "Movement", "Sensing", "Feeding", "Maintenance", "Aggressive", "Defensive", "Transport"
    ],
    "MoleculeClassification": ["pampa", "hia", "pgp", "bioavailability", "bbb",
        "cyp2c19", "cyp3a4", "cyp2d6", "cyp1a2", "cyp2c9", "dili", "herg",
        "carcinogen", "ames", "sarscov2vitro", "sarscov23clpro",
        "cyp2c9substrate", "cyp2d6substrate", "cyp3a4substrate"
    ]
}


def get_task(task_name):
    for class_name, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            try:
                module = importlib.import_module(f".{class_name.lower()}", package=__package__)
                return getattr(module, class_name)
            except ModuleNotFoundError:
                raise ValueError(f"Module for task '{task_name}' could not be found.")
    raise ValueError(f"{task_name} is not a recognized task")
