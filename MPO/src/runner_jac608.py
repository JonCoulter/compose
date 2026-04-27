import os
import time
from datetime import timedelta
from .utils import get_pacific_time, create_logger
from .base_model import BaseModel
from .optim_model import OptimizationModel
from .model import get_mm_model
from .tasks import get_task
from .search import MPO
from .evaluators import get_evaluator

SEARCH_ALGORITHMS = {
    "mpo": MPO,
}


class Runner:
    def __init__(self, args) -> None:
        exp_time = get_pacific_time().strftime("%Y%m%d_%H%M%S")
        self.args = args
        self.log_dir = os.path.join(args.log_dir, exp_time)
        self.logger = create_logger(self.log_dir)

        # Retrieve configuration settings
        (
            base_model_setting,
            optim_model_setting,
            mm_generator_setting,
            task_setting,
            search_setting,
            evaluator_setting,
        ) = self._get_config(args)

        self._validate_model_config(args, mm_generator_setting)

        # Initialize task
        self.task = get_task(args.task_name)(**task_setting)

        # Adjust budget per prompt if necessary
        if len(self.task.train_data) < args.train_size and evaluator_setting["evaluation_method"] == "bayes-ucb":
            args.budget_per_prompt = min(len(self.task.train_data) // 3, args.budget_per_prompt)
            evaluator_setting["budget_per_prompt"] = args.budget_per_prompt

        # Log configuration settings
        self._log_settings(
            base_model_setting,
            optim_model_setting,
            mm_generator_setting,
            task_setting,
            search_setting,
            evaluator_setting,
        )

        # Initialize models and evaluator
        self.base_model = BaseModel(base_model_setting, self.task, self.logger)
        self.mm_generator = get_mm_model(mm_generator_setting["mm_generator_model_name"])(
            **mm_generator_setting, logger=self.logger
        )
        self.optim_model = OptimizationModel(optim_model_setting, self.mm_generator, self.task, self.logger)
        self.evaluator = get_evaluator(evaluator_setting["evaluation_method"])(
            self.base_model, self.task, logger=self.logger, **evaluator_setting
        )
        self.search_algorithm = SEARCH_ALGORITHMS[search_setting["method"]](
            task=self.task,
            base_model=self.base_model,
            optim_model=self.optim_model,
            evaluator=self.evaluator,
            log_dir=self.log_dir,
            logger=self.logger,
            **search_setting,
        )

    def run(self):
        """Start searching from initial prompt"""
        start_time = time.time()
        self.search_algorithm.train()
        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        self.logger.info(f"\nDone! Execution time: {exe_time}")
        self.logger.info(
            f"Optimizer Model: {self.optim_model.model.model_name}, Total cost: {self.optim_model.model.total_cost} USD"
        )
        self.logger.info(
            f"MM Generator Model: {self.mm_generator.model_name}, Total cost: {self.mm_generator.total_cost} USD"
        )

    def _get_config(self, args):
        """Retrieve configuration settings"""
        base_model_setting = {
            "model_name": args.base_model_name,
            "temperature": args.base_model_temperature,
            "debug_output": args.debug_output,
            "port": args.base_model_port,
            "openai_api_key": args.openai_api_key,
            "vllm_api_key": args.vllm_api_key,
            "gemini_api_key": args.gemini_api_key,
        }

        optim_port = args.optim_model_port if args.optim_model_port is not None else args.base_model_port
        optim_model_setting = {
            "model_name": args.optim_model_name,
            "temperature": args.optim_model_temperature,
            "port": optim_port,
            "openai_api_key": args.openai_api_key,
            "vllm_api_key": args.vllm_api_key,
            "gemini_api_key": args.gemini_api_key,
        }

        mm_generator_setting = {
            "mm_generator_model_name": args.mm_generator_model_name,
            "openai_api_key": args.openai_api_key,
            "port": args.mm_generator_port,
            "gemini_api_key": args.gemini_api_key,
            "diffusers_model_id": args.diffusers_model_id,
            "diffusers_num_inference_steps": args.diffusers_num_inference_steps,
            "diffusers_guidance_scale": args.diffusers_guidance_scale,
            "diffusers_height": args.diffusers_height,
            "diffusers_width": args.diffusers_width,
            "diffusers_img2img_strength": args.diffusers_img2img_strength,
        }

        task_setting = {
            "task_name": args.task_name,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "seed": args.seed,
            "data_dir": args.data_dir,
        }

        search_setting = {
            "method": args.search_method,
            "iteration": args.iteration,
            "beam_width": args.beam_width,
            "model_responses_num": args.model_responses_num,
            "test_metric_evaluation_mode": args.test_metric_evaluation_mode,
        }

        evaluator_setting = {
            "evaluation_method": args.evaluation_method,
            "budget_per_prompt": args.budget_per_prompt,
            "beam_width": args.beam_width,
            "num_prompts_per_round": args.num_prompts_per_round,
            "ucb_c": args.ucb_c,
            "bayes_prior_strength": args.bayes_prior_strength,
        }

        return (
            base_model_setting,
            optim_model_setting,
            mm_generator_setting,
            task_setting,
            search_setting,
            evaluator_setting,
        )

    def _validate_model_config(self, args, mm_generator_setting):
        def _uses_openai_chat(name: str) -> bool:
            return "gpt" in (name or "").lower()

        if _uses_openai_chat(args.base_model_name) and not args.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when --base_model_name is an OpenAI model (name contains 'gpt'). "
                "Use a vLLM-served name such as Qwen2.5-VL-7B for local runs."
            )
        if _uses_openai_chat(args.optim_model_name) and not args.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when --optim_model_name is an OpenAI model (name contains 'gpt'). "
                "Use a vLLM-served name such as Qwen2.5-VL-7B for local runs."
            )

        mm_name = mm_generator_setting["mm_generator_model_name"].lower()
        if mm_name in ("gpt-image", "gpt-image-medium") and not args.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for mm_generator_model_name gpt-image or gpt-image-medium. "
                "Use a local backend (e.g. diffusers-sd-turbo) or set OPENAI_API_KEY."
            )
        if mm_name.startswith("diffusers"):
            try:
                import diffusers  # noqa: F401
                import torch  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "Local Diffusers MM generator requires `pip install diffusers` (and torch). "
                ) from e

    def _log_settings(
        self,
        base_model_setting,
        optim_model_setting,
        mm_generator_setting,
        task_setting,
        search_setting,
        evaluator_setting,
    ):
        """Log configuration settings"""
        self.logger.info(f"base_model_setting : {base_model_setting}")
        self.logger.info(f"optim_model_setting : {optim_model_setting}")
        self.logger.info(f"mm_generator_setting : {mm_generator_setting}")
        self.logger.info(f"task_setting : {task_setting}")
        self.logger.info(f"train_size : {len(self.task.train_data)} test_size : {len(self.task.test_data)}")
        self.logger.info(f"search_setting : {search_setting}")
        self.logger.info(f"evaluator_setting : {evaluator_setting}")
