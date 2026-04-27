import base64
import asyncio
import copy
import io
import os
from collections import OrderedDict
from PIL import Image
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

MODEL_DICT = {
    "Qwen2.5-VL-7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "InternVL3-8B": "OpenGVLab/InternVL3-8B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "gemma-3-12b": "google/gemma-3-12b-it",
}

_cache = OrderedDict()
_MAX_CACHE_SIZE = 500


class VLLMModel:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        vllm_api_key: str = None,
        port: int = None,
        max_tokens: int = 1024,
        send_base64: bool = True,
        max_size=(512, 512),
        quality=85,
        **kwargs,
    ):
        self.model_name = MODEL_DICT.get(model_name)

        if "VL" in self.model_name or "gemma-3-12b" in self.model_name:
            self.process_content = self.process_image_content
        elif self.model_name == "Qwen/Qwen3-8B":
            self.process_content = self.process_molecule_content
        else:
            raise ValueError(f"Invalid model type: {self.model_name}")

        if port is None:
            raise ValueError("For VLLMModel, port is required")
        self.port = port
        vllm_api_base = f"http://localhost:{self.port}/v1"
        try:
            self.model = AsyncOpenAI(api_key=vllm_api_key, base_url=vllm_api_base)
        except Exception as e:
            raise RuntimeError("Failed to initialize VLLM client") from e

        self.temperature = temperature
        self.batch_forward_func = self.async_generate_responses
        self.send_base64 = send_base64
        self.max_size = max_size
        self.quality = quality
        self.max_tokens = max_tokens
        self.total_cost = 0

    async def _generate_single_prompt(self, prompt):
        backoff_time = 1
        while backoff_time < 300:
            try:
                if self.model_name == "Qwen/Qwen3-8B":
                    response = await self.model.chat.completions.create(
                        model=self.model_name,
                        messages=prompt,
                        temperature=self.temperature,
                        seed=42,
                        max_tokens=self.max_tokens,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                else:
                    response = await self.model.chat.completions.create(
                        model=self.model_name,
                        messages=prompt,
                        temperature=self.temperature,
                        seed=42,
                        max_tokens=self.max_tokens,
                    )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time *= 2
        raise Exception("Failed to generate response")

    async def async_generation(self, batch_prompts, chunk_size=300):
        responses = []
        for i in range(0, len(batch_prompts), chunk_size):
            batch = batch_prompts[i : i + chunk_size]
            tasks = [self._generate_single_prompt(prompt) for prompt in batch]
            if len(tasks) > 1:
                batch_responses = await tqdm_asyncio.gather(*tasks)
            else:
                batch_responses = await asyncio.gather(*tasks)
            responses.extend(batch_responses)
        return responses

    def generate(self, prompt):
        return self.async_generate_responses([prompt])[0]

    def async_generate_responses(self, batch_prompts):
        batch_prompts = copy.deepcopy(batch_prompts)
        batch_prompts = [self._preprocess_prompt(prompt) for prompt in batch_prompts]
        responses = asyncio.run(self.async_generation(batch_prompts))

        return responses

    def _preprocess_prompt(self, prompt):
        for item in prompt:
            if isinstance(item, dict) and "content" in item:
                self.process_content(item["content"])
        return prompt

    def encode_image(self, image_path):
        cache_key = (image_path, self.max_size, self.quality)
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            return _cache[cache_key]

        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(self.max_size)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=self.quality)
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

            _cache[cache_key] = encoded
            _cache.move_to_end(cache_key)

            if len(_cache) > _MAX_CACHE_SIZE:
                _cache.popitem(last=False)

            return encoded

    def encode_video(self, video_path):
        cache_key = (video_path, self.max_size, self.quality)
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            return _cache[cache_key]

        with open(video_path, "rb") as video_file:
            encoded = base64.b64encode(video_file.read()).decode("utf-8")
            _cache[cache_key] = encoded
            _cache.move_to_end(cache_key)

            if len(_cache) > _MAX_CACHE_SIZE:
                _cache.popitem(last=False)

            return encoded

    def process_image_content(self, content):
        if isinstance(content, list):
            for item in content:
                self.process_image_content(item)  # Recursive call for nested lists
        elif isinstance(content, dict) and content.get("type") == "image":
            content["type"] = "image_url"
            if "image" in content:
                if self.send_base64:
                    image_base64 = self.encode_image(content["image"])
                    content["image_url"] = {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    }
                else:
                    content["image_url"] = {
                        "url": f"file://{os.path.abspath(content['image'])}",
                    }
                content.pop("image")
        elif isinstance(content, dict) and content.get("type") == "video":
            content["type"] = "video_url"
            if "video" in content:
                if self.send_base64:
                    video_base64 = self.encode_video(content["video"])
                    content["video_url"] = {
                        "url": f"data:video/mp4;base64,{video_base64}",
                    }
                else:
                    content["video_url"] = {
                        "url": f"file://{os.path.abspath(content['video'])}",
                    }
                content.pop("video")

    def encode_mol(self, mol):
        assert len(mol["smiles"]) == 1, "Only one SMILES string is expected."
        return f"{mol['smiles'][0]}\n"

    def process_molecule_content(self, content):
        if isinstance(content, list):
            for item in content:
                self.process_molecule_content(item)  # Recursive call for nested lists
        elif isinstance(content, dict) and content.get("type") == "molecule":
            content["type"] = "text"
            content["text"] = self.encode_mol(content["molecule"])
            content.pop("molecule")
