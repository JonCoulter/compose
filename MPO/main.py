import argparse
from src.utils import *
from src.runner import Runner


def parse_test_size(value):
    return None if value == "all" else int(value)


# fmt: off
def load_args():
    parser = argparse.ArgumentParser(description='Process prompt search agent arguments')

    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/')

    # Base model settings
    parser.add_argument('--base_model_name', type=str, required=True)
    parser.add_argument('--base_model_temperature', type=float, default=0.0)
    parser.add_argument('--base_model_port', type=int, default=8501, help='port of the base model, for vllm')
    parser.add_argument('--debug_output', action='store_true', help='log the examples and responses')

    # Optimization model settings
    parser.add_argument('--optim_model_name', type=str, default='Qwen2.5-VL-7B')
    parser.add_argument('--optim_model_temperature', type=float, default=0.0)
    parser.add_argument(
        '--optim_model_port',
        type=int,
        default=None,
        help='vLLM port for the optimizer; defaults to --base_model_port',
    )

    # Multimodal Generator (local-pattern = PIL+numpy, no torch I²; diffusers-* = local SD; gpt-image* = OpenAI)
    parser.add_argument(
        '--mm_generator_model_name',
        type=str,
        default='diffusers-sd-turbo',
        help='local-pattern, diffusers-sd-turbo, diffusers-sd-1-4, diffusers-flux-schnell, gpt-image*, or dummy',
    )
    parser.add_argument('--mm_generator_port', type=int, default=8501, help='port of the multimodal generator')
    parser.add_argument('--diffusers_model_id', type=str, default='stabilityai/sd-turbo')
    parser.add_argument('--diffusers_num_inference_steps', type=int, default=2)
    parser.add_argument('--diffusers_guidance_scale', type=float, default=0.0)
    parser.add_argument('--diffusers_height', type=int, default=512)
    parser.add_argument('--diffusers_width', type=int, default=512)
    parser.add_argument('--diffusers_img2img_strength', type=float, default=0.65)

    # Task settings
    parser.add_argument('--train_size', type=int, default=300)
    parser.add_argument('--test_size', type=parse_test_size, default=300, help="Number of test examples (int). Use 'all' to use all test examples.")    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--class_num', type=int, default=5, help='number of classes for butterfly task')

    # Search settings
    parser.add_argument('--search_method', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=12)
    parser.add_argument('--beam_width', type=int, default=3)
    parser.add_argument('--model_responses_num', type=int, default=3, help='number of examples for feedback generation')
    parser.add_argument('--test_metric_evaluation_mode', type=str, default='best', choices=['total', 'updated', 'best'], help='mode to evaluate test metric')
    # GMPO-specific settings
    parser.add_argument(
        '--gmpo_optim_sees_images',
        type=str,
        default='query_only',
        choices=['none', 'query_only'],
        help='Whether the GMPO optimizer sees the actual query images of failed examples '
             'when proposing rules. "none" = text-only failure descriptions; '
             '"query_only" = attach the real wrong-example query images (capped by vLLM '
             'per-prompt image limits). Has no effect when --search_method != gmpo.',
    )
    parser.add_argument(
        '--gmpo_max_rules',
        type=int,
        default=10,
        help='Maximum number of guideline rules carried on a GMPO node.',
    )
    parser.add_argument(
        '--gmpo_rule_mode',
        type=str,
        default='auto',
        choices=['auto', 'general', 'discrimination'],
        help='auto: use class_discrimination when class_choices are detected, else general. '
             'general: force the original WHEN/THEN schema. '
             'discrimination: force per-class feature schema (errors if no class_choices).',
    )
    # Evaluator settings
    parser.add_argument('--evaluation_method', type=str, default='uniform', choices=['successive_halving', 'our_successive_halving', 'successive_rejects', 's_successive_rejects', 'ucb', 'ucb-e', 'uniform', 'bayes-ucb'])
    parser.add_argument('--budget_per_prompt', type=int, default=100, help='budget per prompt for Evaluator')
    parser.add_argument('--num_prompts_per_round', type=int, default=3, help='number of prompts per evaluation round')
    parser.add_argument('--ucb_c', type=float, default=2.0, help='parameter for UCB / Bayes-UCB')
    parser.add_argument('--bayes_prior_strength', type=float, default=10.0, help='prior strength for Bayes-UCB')

    args = parser.parse_args()
    
    # API key should be set in the environment variable.
    args.vllm_api_key = os.environ["VLLM_API_KEY"] if "VLLM_API_KEY" in os.environ else None
    args.openai_api_key = os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else None
    args.gemini_api_key = os.environ["GEMINI_API_KEY"] if "GEMINI_API_KEY" in os.environ else None

    return args

def main(args):
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    args = load_args()
    main(args)
