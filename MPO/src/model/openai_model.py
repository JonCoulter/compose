from openai import OpenAI, AsyncOpenAI
import time
import asyncio
from tqdm.asyncio import tqdm_asyncio
import base64
import copy
import io
import os
from collections import OrderedDict
from PIL import Image
import cv2
from rich import print

OPENAI_MODEL_CONFIG = {
    "gpt-4o": {
        "full_name": "gpt-4o-2024-08-06",
        "type": "image",
        "price": {"prompt": 2.5 / 1000000, "completion": 10 / 1000000},
    },
    "gpt-4o-mini": {
        "full_name": "gpt-4o-mini-2024-07-18",
        "type": "image",
        "price": {"prompt": 0.15 / 1000000, "completion": 0.0006 / 1000000},
    },
    "gpt-4.1-nano": {
        "full_name": "gpt-4.1-nano-2025-04-14",
        "type": "image",
        "price": {"prompt": 0.1 / 1000000, "completion": 0.4 / 1000000},
    },
    "gpt-4.1-mini": {
        "full_name": "gpt-4.1-mini-2025-04-14",
        "type": "image",
        "price": {"prompt": 0.4 / 1000000, "completion": 1.6 / 1000000},
    },
    "gpt-4.1": {
        "full_name": "gpt-4.1",
        "type": "image",
        "price": {"prompt": 2 / 1000000, "completion": 8 / 1000000},
    },
}

_cache = OrderedDict()
_MAX_CACHE_SIZE = 500

class OpenAIModel:
    def __init__(
        self,
        model_name: str,
        openai_api_key: str,
        temperature: float,
        async_mode: bool = True,
        max_size: tuple = (512, 512),
        max_size_video: tuple = (512, 512),
        quality: int = 85,
        **kwargs,
    ):
        try:
            self.async_model = AsyncOpenAI(api_key=openai_api_key)
            self.model = OpenAI(api_key=openai_api_key)
        except Exception as e:
            print(f"Init openai client error: \n{e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e

        self.model_config = OPENAI_MODEL_CONFIG[model_name]
        self.model_name = self.model_config["full_name"]
        self.model_type = self.model_config["type"]
        self.price = self.model_config["price"]

        self.temperature = temperature
        self.async_mode = async_mode
        self.max_size = max_size
        self.max_size_video = max_size_video
        self.quality = quality

        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.gpt_chat_completion

        self.total_cost = 0

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
        cache_key = (video_path, self.max_size_video, self.quality)
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            return _cache[cache_key]

        video = cv2.VideoCapture(video_path)

        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            frame = cv2.resize(frame, self.max_size_video)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        _cache[cache_key] = base64Frames
        _cache.move_to_end(cache_key)

        if len(_cache) > _MAX_CACHE_SIZE:
            _cache.popitem(last=False)

        return base64Frames

    def process_content(self, content):
        if isinstance(content, list):
            i = 0
            while i < len(content):
                item = content[i]
                
                if not isinstance(item, dict):
                    i += 1
                    continue
                
                if item.get("type") == "video" and "video" in item:
                    video_frames = self.encode_video(item["video"])
                    # use only first, middle, last FRAMES
                    video_frames = [video_frames[0], video_frames[len(video_frames)//2], video_frames[-1]]
                    image_items = [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                    } for frame in video_frames]
                    
                    content.pop(i)  
                    for j, image_item in enumerate(image_items):
                        content.insert(i + j, image_item)
                    i += len(image_items)  
                
                elif item.get("type") == "image" and "image" in item:
                    image_base64 = self.encode_image(item["image"])
                    item["type"] = "image_url"
                    item["image_url"] = {"url": f"data:image/jpeg;base64,{image_base64}"}
                    item.pop("image")
                    i += 1

                elif item.get('type') == 'molecule' and 'molecule' in item:
                    item["type"] = "text"
                    item["text"] = item['molecule']['smiles'][0] + '\n'
                    item.pop('molecule')
                    i += 1

                elif "content" in item:
                    self.process_content(item["content"])
                    i += 1
                
                else:
                    i += 1
        
        elif isinstance(content, dict) and content.get("type") == "image" and "image" in content:
            image_base64 = self.encode_image(content["image"])
            content["type"] = "image_url"
            content["image_url"] = {"url": f"data:image/jpeg;base64,{image_base64}"}
            content.pop("image")

        elif isinstance(content, dict) and content.get("type") == "video" and "video" in content:
            video_frames = self.encode_video(content["video"])
            content["type"] = "image_url"
            content["image_url"] = [{"url": f"data:image/jpeg;base64,{frame}"} for frame in video_frames]

        elif isinstance(content, dict) and content.get("type") == "molecule" and "molecule" in content:
            content["type"] = "text"
            content["text"] = content["molecule"]["smiles"][0] + "\n"
            content.pop("molecule")


    def batch_forward_chatcompletion(self, batch_prompts):
        batch_prompts = copy.deepcopy(batch_prompts)
        batch_prompts = [self._preprocess_prompt(prompt) for prompt in batch_prompts]

        if self.async_mode:
            responses = self.async_generate_responses(batch_prompts=batch_prompts)
        else:
            responses = [self.gpt_chat_completion(prompt=prompt) for prompt in batch_prompts]
        return responses

    def gpt_chat_completion(self, prompt):
        prompt = copy.deepcopy(prompt)
        prompt = self._preprocess_prompt(prompt)

        backoff_time = 1
        start_time = time.time()
        while backoff_time < 30:
            try:
                create_kwargs = {
                    "messages": prompt,
                    "model": self.model_name,
                    "temperature": self.temperature,
                }
                response = self.model.chat.completions.create(**create_kwargs)
                inference_time = time.time() - start_time
                print(f"OPENAI Inference time: {inference_time:.4f} seconds")
                # Add cost calculation
                cost = self.calculate_cost(response)
                self.total_cost += cost
                print(f"OPENAI INFERENCE COST: {cost:.4f} USD, TOTAL COST: {self.total_cost:.4f} USD")
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2

        raise Exception("Failed to generate response from OpenAI")

    def calculate_cost(self, response):
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        cost = (prompt_tokens * self.price["prompt"]) + (completion_tokens * self.price["completion"])
        return cost


    async def _generate_single_prompt(self, prompt):
        backoff_time = 1
        while backoff_time < 30:
            try:
                create_kwargs = {
                    "messages": prompt,
                    "model": self.model_name,
                    "temperature": self.temperature,
                }
                response = await self.async_model.chat.completions.create(**create_kwargs)
                # Add cost calculation
                cost = self.calculate_cost(response)
                self.total_cost += cost
                print(f"OPENAI INFERENCE COST: {cost:.4f} USD, TOTAL COST: {self.total_cost:.4f} USD")
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(e, f" Sleeping {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time *= 2

    async def async_generation(self, batch_prompts, chunk_size=300):
        responses = []
        for i in range(0, len(batch_prompts), chunk_size):
            batch = batch_prompts[i : i + chunk_size]
            tasks = [self._generate_single_prompt(prompt) for prompt in batch]
            batch_responses = await tqdm_asyncio.gather(*tasks)
            responses.extend(batch_responses)
        return responses

    def async_generate_responses(self, batch_prompts):
        return asyncio.run(self.async_generation(batch_prompts))
