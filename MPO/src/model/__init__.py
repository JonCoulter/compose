from .openai_model import OpenAIModel
from .mmgenerator import (
    DiffusersImageGenerator,
    DummyImageGenerator,
    LocalPatternImageGenerator,
    OpenAIImageGenerator,
)
from .vllm import VLLMModel

# Define mappings for different model types
MODEL_MAPPING = {
    "gpt": OpenAIModel,
    "qwen": VLLMModel,
    "intern": VLLMModel,
    "gemma": VLLMModel,
}

MM_MODEL_MAPPING = {
    "gpt-image": OpenAIImageGenerator,
    "gpt-image-medium": OpenAIImageGenerator,
    "diffusers-sd-turbo": DiffusersImageGenerator,
    "diffusers-flux-schnell": DiffusersImageGenerator,
    "local-pattern": LocalPatternImageGenerator,
    "dummy": DummyImageGenerator,
}


# Function to get the appropriate language model
def get_language_model(model_name):
    model_query = model_name.lower()

    for key, model in MODEL_MAPPING.items():
        if key in model_query:
            return model

    raise ValueError(f"Language model {model_name} is not supported.")

# Function to get the appropriate multimodal model
def get_mm_model(model_name):
    model_query = model_name.lower()

    for key, model in MM_MODEL_MAPPING.items():
        if key == model_query:
            return model

    raise ValueError(f"Multimodal generator model {model_name} is not supported.")
