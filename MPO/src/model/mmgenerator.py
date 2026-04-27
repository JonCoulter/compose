import openai
from openai import OpenAI
from abc import ABC, abstractmethod
import base64
import hashlib
import os
import threading
from rich import print

import fcntl

_DIFFUSERS_LOCK_PATH = "/tmp/diffusers_generator.lock"

def _pil_resample_lanczos():
    from PIL import Image

    return getattr(Image, "Resampling", Image).LANCZOS

MM_GENERATION_MODEL_CONFIG = {
    "gpt-image": {
        "class": "OpenAIImageGenerator",
        "full_name": "gpt-image-1",
        "target_modality": "image",
        "quality": "low",  # "medium",
        "response_format": None,
        "price": {"input": 5 / 1000000, "output": 40 / 1000000},
    },
    "gpt-image-medium": {
        "class": "OpenAIImageGenerator",
        "full_name": "gpt-image-1",
        "target_modality": "image",
        "quality": "medium",
        "response_format": None,
        "price": {"input": 5 / 1000000, "output": 40 / 1000000},
    },
}

DIFFUSERS_MM_CONFIG = {
    "diffusers-sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "flux": False,
        "pipeline": "StableDiffusionPipeline",  # ← not XL
    },
    "diffusers-sdxl": {
        "model_id": "/ix/cs2770_2026s/jac608/project/models/sdxl",
        "flux": False,
        "pipeline": "StableDiffusionXLPipeline",
    },
    "diffusers-flux-schnell": {
        "model_id": "/ix/cs2770_2026s/jac608/project/models/flux-schnell",
        "flux": True,
        "pipeline": "FluxPipeline",
    },
}

class MMGenerator(ABC):
    def __init__(self, logger, **kwargs):
        self.logger = logger
        self.image_dir = os.path.abspath(os.path.join(logger.log_dir, "images"))
        self.target_modality = None  # "image" or "video" or "molecule"
        self.total_cost = 0

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    @abstractmethod
    def generate(self, prompt, **kwargs):
        """
        get text prompt and return multimodal path
        """
        pass

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @staticmethod
    def _is_invalid_prompt(prompt):
        if prompt is None:
            return True
        if not isinstance(prompt, str):
            return False
        return prompt.strip().lower() in {"", "none", "null", "n/a", "na"}


class OpenAIImageGenerator(MMGenerator):
    def __init__(
        self,
        mm_generator_model_name,
        openai_api_key,
        logger,
        image_size="1024x1024",
        **kwargs,
    ):
        super().__init__(logger=logger)
        self.client = OpenAI(api_key=openai_api_key)
        self.model_config = MM_GENERATION_MODEL_CONFIG[mm_generator_model_name]
        self.model_name = self.model_config["full_name"]
        self.target_modality = self.model_config["target_modality"]

        self.sampling_params = {
            "size": image_size,
            "quality": self.model_config["quality"],
        }
        if self.model_config["response_format"]:
            self.sampling_params["response_format"] = self.model_config["response_format"]

        self.price = self.model_config["price"]

        self.image_size = image_size
        self.total_cost = 0

    def save_b64_image(self, b64_string):
        """Save a base64 string as an image file"""
        if b64_string:
            # Get number of existing images
            existing_images = len([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
            image_path = os.path.join(self.image_dir, f"image_{existing_images+1}.jpg")

            with open(image_path, "wb") as f:
                f.write(base64.b64decode(b64_string))
            return image_path
        return None

    def generate(self, prompt, mm_prompt_path=None, **kwargs):
        if self._is_invalid_prompt(prompt):
            self.logger.error("OpenAIImageGenerator: prompt is empty/None, skipping generation")
            return None
        try:
            if mm_prompt_path is not None:  # Edit
                response = self.client.images.edit(
                    model=self.model_name,
                    prompt=prompt,
                    image=[open(mm_prompt_path, "rb")],
                    **self.sampling_params,
                )
            else:
                response = self.client.images.generate(
                    model=self.model_name,
                    prompt=prompt,
                    **self.sampling_params,
                )

            b64_image = response.data[0].b64_json
            image_path = self.save_b64_image(b64_image)
            cost = self.calculate_cost(response)
            self.total_cost += cost
            self.logger.info(f"OPENAI IMAGE GENERATION COST: {cost:.4f} USD, TOTAL COST: {self.total_cost:.4f} USD")

            return image_path

        except openai.OpenAIError as e:
            self.logger.error(f"Error generating image: {e}")
            self.logger.info(f"Prompt: {prompt}")

            return None

    def multimodal_mixing(self, parents, mm_mix_prompt, **kwargs):
        if self._is_invalid_prompt(mm_mix_prompt):
            self.logger.error("OpenAIImageGenerator: mix prompt is empty/None, skipping generation")
            return None
        try:
            response = self.client.images.edit(
                model=self.model_name,
                prompt=mm_mix_prompt,
                image=[open(parent.mm_prompt_path, "rb") for parent in parents if parent.mm_prompt_path is not None],
                **self.sampling_params,
            )

            b64_image = response.data[0].b64_json
            image_path = self.save_b64_image(b64_image)
            cost = self.calculate_cost(response)
            self.total_cost += cost
            self.logger.info(f"OPENAI IMAGE GENERATION COST: {cost:.4f} USD, TOTAL COST: {self.total_cost:.4f} USD")

            return image_path

        except openai.OpenAIError as e:
            self.logger.error(f"Error generating image: {e}")
            self.logger.info(f"Prompt: {mm_mix_prompt}")

            return None

    def calculate_cost(self, response):
        usage = response.usage
        if usage is None:
            return self.price["output"]
        else:
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cost = (input_tokens * self.price["input"]) + (output_tokens * self.price["output"])
            return cost


class DiffusersImageGenerator(MMGenerator):
    """Local text-to-image / image-to-image via Hugging Face Diffusers."""
    _shared_txt2img_pipe = None
    _shared_img2img_pipe = None
    _shared_model_id = None
    _load_lock = threading.Lock()
    
    def _acquire_file_lock(self):
        """Cross-process lock so only one process loads/runs at a time."""
        f = open(_DIFFUSERS_LOCK_PATH, "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def _release_file_lock(self, f):
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()

    def __init__(
        self,
        mm_generator_model_name,
        logger,
        diffusers_model_id=None,
        diffusers_num_inference_steps=4,
        diffusers_guidance_scale=0.0,
        diffusers_height=1024,
        diffusers_width=1024,
        diffusers_img2img_strength=0.65,
        **kwargs,
    ):
        super().__init__(logger=logger)
        if mm_generator_model_name not in DIFFUSERS_MM_CONFIG:
            raise ValueError(
                f"Unknown Diffusers preset {mm_generator_model_name}. Known: {list(DIFFUSERS_MM_CONFIG.keys())}"
            )
        preset = DIFFUSERS_MM_CONFIG[mm_generator_model_name]
        self.model_id = diffusers_model_id or preset["model_id"]
        self.is_flux = preset.get("flux", False)
        self.pipeline_cls_name = preset.get("pipeline", "StableDiffusionXLPipeline")
        self.num_inference_steps = int(diffusers_num_inference_steps)
        self.guidance_scale = float(diffusers_guidance_scale)
        self.height = int(diffusers_height)
        self.width = int(diffusers_width)
        self.img2img_strength = float(diffusers_img2img_strength)

        self.model_name = mm_generator_model_name
        self.target_modality = "image"
        self.total_cost = 0.0

        self._txt2img_pipe = None
        self._img2img_pipe = None

    def _device_and_dtype(self):
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32

    @staticmethod
    def _assert_no_meta_tensors(pipe, where: str):
        import torch

        for comp_name, component in pipe.components.items():
            if not isinstance(component, torch.nn.Module):
                continue  # schedulers, tokenizers, etc. — not nn.Modules
            for name, t in list(component.named_parameters()) + list(component.named_buffers()):
                if isinstance(t, torch.Tensor) and t.is_meta:
                    raise RuntimeError(
                        f"Diffusers pipeline has meta tensor {where}: "
                        f"{comp_name}.{name}"
                    )

    def _stable_pipeline_from_pretrained(self, pipeline_cls):
        import torch
        from contextlib import contextmanager
        from transformers import CLIPTextModel
        from diffusers import AutoencoderKL
        import accelerate
        import accelerate.big_modeling
        import transformers.modeling_utils

        is_sdxl = self.pipeline_cls_name == "StableDiffusionXLPipeline"

        @contextmanager
        def _no_meta():
            yield

        targets = [
            (accelerate, "init_empty_weights"),
            (accelerate.big_modeling, "init_empty_weights"),
            (transformers.modeling_utils, "init_empty_weights"),
        ]
        originals = [(mod, name, getattr(mod, name)) for mod, name in targets]
        for mod, name in targets:
            setattr(mod, name, _no_meta)

        try:
            device, _ = self._device_and_dtype()

            self.logger.info("DiffusersImageGenerator: pre-loading text encoder")
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_id, subfolder="text_encoder", dtype=torch.float32
            )

            self.logger.info("DiffusersImageGenerator: pre-loading VAE")
            vae = AutoencoderKL.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=torch.float32
            )

            base_kw = {
                "use_safetensors": True,
                "text_encoder": text_encoder,
                "vae": vae,
                "torch_dtype": torch.float32,
            }

            if is_sdxl:
                from transformers import CLIPTextModelWithProjection
                from diffusers import UNet2DConditionModel
                self.logger.info("DiffusersImageGenerator: pre-loading text encoder 2 + UNet (SDXL)")
                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    self.model_id, subfolder="text_encoder_2", dtype=torch.float32
                )
                unet = UNet2DConditionModel.from_pretrained(
                    self.model_id, subfolder="unet", torch_dtype=torch.float32
                )
                base_kw["text_encoder_2"] = text_encoder_2
                base_kw["unet"] = unet

            pipe = pipeline_cls.from_pretrained(self.model_id, **base_kw)
        finally:
            for mod, name, original in originals:
                setattr(mod, name, original)

        self._assert_no_meta_tensors(pipe, "after CPU float32 load")

        if device == "cuda":
            if is_sdxl:
                pipe.unet = pipe.unet.to(dtype=torch.float16)
                pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=torch.float16)
            pipe.vae = pipe.vae.to(dtype=torch.float16)
            pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float16)
            pipe = pipe.to(torch.device("cuda"))
            
            # Cast ALL remaining nn.Module components to float16
            # (catches unet for non-SDXL and anything else that slipped through)
            for comp_name, component in pipe.components.items():
                if isinstance(component, torch.nn.Module):
                    component.to(dtype=torch.float16)
            
            self._assert_no_meta_tensors(pipe, "after CUDA transfer")

        return pipe

    def _flux_pipeline_from_pretrained(self, pipeline_cls):
        import torch
        from contextlib import contextmanager
        import accelerate
        import accelerate.big_modeling
        import transformers.modeling_utils

        @contextmanager
        def _no_meta():
            yield

        original_module_to = torch.nn.Module.to

        def _safe_module_to(self, *args, **kwargs):
            has_meta = any(p.is_meta for p in self.parameters()) or \
                    any(b.is_meta for b in self.buffers())
            if has_meta:
                for arg in args:
                    if isinstance(arg, (str, torch.device)):
                        return original_module_to(
                            self.to_empty(device=arg), *args, **kwargs
                        )
                if "device" in kwargs:
                    return original_module_to(
                        self.to_empty(device=kwargs["device"]), *args, **kwargs
                    )
            return original_module_to(self, *args, **kwargs)

        targets = [
            (accelerate, "init_empty_weights"),
            (accelerate.big_modeling, "init_empty_weights"),
            (transformers.modeling_utils, "init_empty_weights"),
        ]
        try:
            import diffusers.models.transformers.transformer_flux as flux_transformer
            targets.append((flux_transformer, "init_empty_weights"))
        except (ImportError, AttributeError):
            pass

        originals = []
        for mod, name in targets:
            if hasattr(mod, name):
                originals.append((mod, name, getattr(mod, name)))
                setattr(mod, name, _no_meta)

        torch.nn.Module.to = _safe_module_to

        try:
            self.logger.info(f"DiffusersImageGenerator: loading FLUX pipeline {pipeline_cls.__name__}")
            pipe = pipeline_cls.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
            device, _ = self._device_and_dtype()
            pipe = pipe.to(device)

            # ← dtype cast goes here, after pipe.to(device)
            for name, component in pipe.components.items():
                if isinstance(component, torch.nn.Module):
                    try:
                        component.to(dtype=torch.bfloat16)
                    except Exception:
                        for submodule in component.modules():
                            for param in submodule.parameters(recurse=False):
                                if param.dtype == torch.float32:
                                    param.data = param.data.to(torch.bfloat16)
                            for buf in submodule.buffers(recurse=False):
                                if buf.dtype == torch.float32:
                                    buf.data = buf.data.to(torch.bfloat16)

        finally:
            torch.nn.Module.to = original_module_to
            for mod, name, original in originals:
                setattr(mod, name, original)

        return pipe

    def _get_txt2img(self):
        if DiffusersImageGenerator._shared_txt2img_pipe is not None and \
        DiffusersImageGenerator._shared_model_id == self.model_id:
            return DiffusersImageGenerator._shared_txt2img_pipe

        with DiffusersImageGenerator._load_lock:
            if DiffusersImageGenerator._shared_txt2img_pipe is not None and \
            DiffusersImageGenerator._shared_model_id == self.model_id:
                return DiffusersImageGenerator._shared_txt2img_pipe

            self.logger.info(f"DiffusersImageGenerator: is_flux={self.is_flux}, model_id={self.model_id}")

            if self.is_flux:
                from diffusers import FluxPipeline
                pipe = self._flux_pipeline_from_pretrained(FluxPipeline)
            else:
                if self.pipeline_cls_name == "StableDiffusionPipeline":
                    from diffusers import StableDiffusionPipeline as PipelineCls
                else:
                    from diffusers import StableDiffusionXLPipeline as PipelineCls
                pipe = self._stable_pipeline_from_pretrained(PipelineCls)
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()

            DiffusersImageGenerator._shared_txt2img_pipe = pipe
            DiffusersImageGenerator._shared_model_id = self.model_id
            return DiffusersImageGenerator._shared_txt2img_pipe

    def _get_img2img(self):
        if DiffusersImageGenerator._shared_img2img_pipe is not None and \
        DiffusersImageGenerator._shared_model_id == self.model_id:
            return DiffusersImageGenerator._shared_img2img_pipe

        if self.is_flux:
            from diffusers import FluxImg2ImgPipeline
            pipe = self._flux_pipeline_from_pretrained(FluxImg2ImgPipeline)
        else:
            if self.pipeline_cls_name == "StableDiffusionPipeline":
                from diffusers import StableDiffusionImg2ImgPipeline as Img2ImgCls
            else:
                from diffusers import StableDiffusionXLImg2ImgPipeline as Img2ImgCls
            base = self._get_txt2img()
            pipe = Img2ImgCls(**base.components)
            device, _ = self._device_and_dtype()
            pipe.to(device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()

        DiffusersImageGenerator._shared_img2img_pipe = pipe
        return DiffusersImageGenerator._shared_img2img_pipe

    def _alloc_image_path(self):
        existing = len([f for f in os.listdir(self.image_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
        return os.path.join(self.image_dir, f"image_{existing + 1}.jpg")

    def _save_pil(self, image):
        path = self._alloc_image_path()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=92)
        self.logger.info(f"DiffusersImageGenerator: saved {path}")
        return path

    def generate(self, prompt, mm_prompt_path=None, **kwargs):
        if prompt is None:
            self.logger.error("DiffusersImageGenerator: prompt is None, skipping")
            return None

        lock_file = self._acquire_file_lock()
        try:
            from PIL import Image

            if self.is_flux:
                if mm_prompt_path is not None:
                    pipe = self._get_img2img()
                    init_image = Image.open(mm_prompt_path).convert("RGB")
                    init_image = init_image.resize(
                        (self.width, self.height), _pil_resample_lanczos()
                    )
                    out = pipe(
                        prompt=prompt,
                        image=init_image,
                        strength=self.img2img_strength,
                        num_inference_steps=self.num_inference_steps,
                        height=self.height,
                        width=self.width,
                    ).images[0]
                else:
                    pipe = self._get_txt2img()
                    out = pipe(
                        prompt=prompt,
                        num_inference_steps=self.num_inference_steps,
                        height=self.height,
                        width=self.width,
                    ).images[0]
            else:
                if mm_prompt_path is not None:
                    pipe = self._get_img2img()
                    init_image = Image.open(mm_prompt_path).convert("RGB")
                    init_image = init_image.resize(
                        (self.width, self.height), _pil_resample_lanczos()
                    )
                    out = pipe(
                        prompt=prompt,
                        image=init_image,
                        strength=self.img2img_strength,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                    ).images[0]
                else:
                    pipe = self._get_txt2img()
                    out = pipe(
                        prompt=prompt,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        height=self.height,
                        width=self.width,
                    ).images[0]

            return self._save_pil(out)

        except Exception as e:
            self.logger.error(f"DiffusersImageGenerator generate failed: {e}")
            self.logger.info(f"Prompt: {prompt}")
            return None
        finally:
            self._release_file_lock(lock_file)

    def multimodal_mixing(self, parents, mm_mix_prompt, **kwargs):
        try:
            from PIL import Image
            import numpy as np

            paths = [p.mm_prompt_path for p in parents if getattr(p, "mm_prompt_path", None)]
            if not paths:
                self.logger.error("DiffusersImageGenerator multimodal_mixing: no parent images")
                return None

            images = [Image.open(p).convert("RGB") for p in paths]
            w, h = self.width, self.height
            resample = _pil_resample_lanczos()
            images = [im.resize((w, h), resample) for im in images]

            # Equal-weight blending across all parents
            if len(images) == 1:
                blended = images[0]
            else:
                arrays = [np.array(im, dtype=np.float32) for im in images]
                blended = Image.fromarray(
                    np.mean(arrays, axis=0).clip(0, 255).astype(np.uint8)
                )

            pipe = self._get_img2img()

            if self.is_flux:
                out = pipe(
                    prompt=mm_mix_prompt,
                    image=blended,
                    strength=self.img2img_strength,
                    num_inference_steps=self.num_inference_steps,
                    height=self.height,
                    width=self.width,
                ).images[0]
            else:
                out = pipe(
                    prompt=mm_mix_prompt,
                    image=blended,
                    strength=self.img2img_strength,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                ).images[0]

            return self._save_pil(out)

        except Exception as e:
            self.logger.error(f"DiffusersImageGenerator multimodal_mixing failed: {e}")
            self.logger.info(f"Prompt: {mm_mix_prompt}")
            return None


class LocalPatternImageGenerator(MMGenerator):
    """
    Fully local image paths for MPO without torch/diffusers at inference time.

    Writes deterministic JPEGs from the prompt (gradient + light noise). This is not
    photorealistic generation; it exists so local VLMs + MPO can run reliably when
    Diffusers hits environment-specific meta-tensor / CUDA issues. Use
    ``diffusers-sd-turbo`` when that stack works on your machine.
    """

    def __init__(
        self,
        mm_generator_model_name,
        logger,
        diffusers_height=512,
        diffusers_width=512,
        **kwargs,
    ):
        super().__init__(logger=logger)
        self.model_name = mm_generator_model_name
        self.target_modality = "image"
        self.total_cost = 0.0
        self.width = int(diffusers_width)
        self.height = int(diffusers_height)
        self._blend_strength = float(
            kwargs.get("local_pattern_blend", os.environ.get("LOCAL_PATTERN_BLEND", "0.35"))
        )

    def _rng_from_text(self, text: str):
        import numpy as np

        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8", errors="replace")).digest()[:8], "little")
        return np.random.default_rng(seed)

    def _image_from_prompt(self, prompt: str):
        """RGB image: smooth gradients + noise; reproducible given prompt."""
        import numpy as np
        from PIL import Image

        w, h = self.width, self.height
        rng = self._rng_from_text(prompt)
        raw = hashlib.sha256(prompt.encode("utf-8", errors="replace")).digest()
        r0, g0, b0 = raw[0] / 255.0, raw[1] / 255.0, raw[2] / 255.0
        r1, g1, b1 = raw[8] / 255.0, raw[9] / 255.0, raw[10] / 255.0
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
        gv, gu = np.meshgrid(ys, xs, indexing="ij")
        r = (r0 * (1 - gu) + r1 * gu) * (0.85 + 0.15 * gv) + 0.08 * rng.standard_normal((h, w))
        g = (g0 * (1 - gv) + g1 * gv) * (0.85 + 0.15 * gu) + 0.08 * rng.standard_normal((h, w))
        b = (b0 * (1 - gu * gv) + b1 * gu * gv) + 0.08 * rng.standard_normal((h, w))
        rgb = (np.stack([r, g, b], axis=-1) * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    def _alloc_image_path(self):
        existing = len([f for f in os.listdir(self.image_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
        return os.path.join(self.image_dir, f"image_{existing + 1}.jpg")

    def _save_pil(self, image):
        from PIL import Image

        path = self._alloc_image_path()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=92)
        self.logger.info(f"LocalPatternImageGenerator: saved {path}")
        return path

    def generate(self, prompt, mm_prompt_path=None, **kwargs):
        if self._is_invalid_prompt(prompt):
            self.logger.error("LocalPatternImageGenerator: prompt is empty/None, skipping generation")
            return None
        try:
            from PIL import Image

            if mm_prompt_path is None:
                return self._save_pil(self._image_from_prompt(prompt))

            base = Image.open(mm_prompt_path).convert("RGB")
            base = base.resize((self.width, self.height), _pil_resample_lanczos())
            overlay = self._image_from_prompt(prompt)
            alpha = max(0.0, min(1.0, self._blend_strength))
            out = Image.blend(base, overlay, alpha)
            return self._save_pil(out)
        except Exception as e:
            self.logger.error(f"LocalPatternImageGenerator generate failed: {e}")
            self.logger.info(f"Prompt: {prompt}")
            return None

    def multimodal_mixing(self, parents, mm_mix_prompt, **kwargs):
        if self._is_invalid_prompt(mm_mix_prompt):
            self.logger.error("LocalPatternImageGenerator: mix prompt is empty/None, skipping generation")
            return None
        try:
            from PIL import Image

            paths = [p.mm_prompt_path for p in parents if getattr(p, "mm_prompt_path", None)]
            if not paths:
                self.logger.error("LocalPatternImageGenerator multimodal_mixing: no parent images")
                return None

            images = [Image.open(p).convert("RGB") for p in paths]
            w, h = self.width, self.height
            resample = _pil_resample_lanczos()
            images = [im.resize((w, h), resample) for im in images]

            if len(images) == 1:
                blended = images[0]
            else:
                blended = images[0]
                for im in images[1:]:
                    blended = Image.blend(blended, im, 0.5)

            overlay = self._image_from_prompt(mm_mix_prompt)
            alpha = max(0.0, min(1.0, self._blend_strength))
            out = Image.blend(blended, overlay, alpha)
            return self._save_pil(out)
        except Exception as e:
            self.logger.error(f"LocalPatternImageGenerator multimodal_mixing failed: {e}")
            self.logger.info(f"Prompt: {mm_mix_prompt}")
            return None


class DummyImageGenerator(MMGenerator):
    def __init__(self, logger, **kwargs):
        super().__init__(logger=logger)
        self.model_name = "dummy"
        self.target_modality = "image"
        self.total_cost = 0

    def generate(self, prompt, **kwargs):
        self.logger.info("DummyImageGenerator: skipping image generation")
        return None

    def multimodal_mixing(self, parents, mm_mix_prompt, **kwargs):
        self.logger.info("DummyImageGenerator: skipping multimodal mixing")
        return None
