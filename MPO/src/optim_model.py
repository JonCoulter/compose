import re
from pathlib import Path
from .model import get_language_model
from .tasks import BaseTask
from .utils import check_mm_type
from .model.mmgenerator import MMGenerator


### MPO: Cohesive Backpropagation
def get_multimodal_analysis_prompt(text_prompt, mm_prompt_path, example_prompt, modality="image"):
    before_image_prompt = f"""
You are a Prompt Failure Analysis Agent specialized in multimodal prompt optimization. Your task is to analyze the failure case of a Multimodal Large Language Model (MLLM) and identify the potential reasons in the prompt for the model's incorrect prediction. Based on the given input, output, and ground truth, analyze both the Text Prompt and the Image Prompt used in the task.

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Prompts:
- Text Prompt : {text_prompt}
- Image Prompt : 
""".strip()

    after_image_prompt = f"""
### Wrong Examples:
""".strip()

    after_example_prompt = f"""
### Output Format:
Text Prompt Analysis:
- Identify missing information, vague instructions, or ambiguous wording that could have misled the model.
- Explain how weaknesses in the Text Prompt may have contributed to the wrong output.
- Suggest specific improvements (e.g., clearer task definition, additional constraints, better examples) to help the model produce the correct answer.

Image Prompt Analysis:
- If a image Prompt was used, analyze its effectiveness.
- Identify problems such as lack of clarity, poor composition, irrelevant details, or missing key features.
- If no image Prompt was used, suggest what kind of image (visual content, attributes, composition) would help correct the failure.
""".strip()

    mm_prompt_modality = check_mm_type(mm_prompt_path) if mm_prompt_path else "text"
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_image_prompt},
                (
                    {"type": mm_prompt_modality, mm_prompt_modality: mm_prompt_path}
                    if mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_image_prompt},
                *example_prompt,
                {"type": "text", "text": after_example_prompt},
            ],
        },
    ]

    return prompt


### MPO Generation
def get_multimodal_generation_prompt(text_prompt, mm_prompt_path, example_prompt, analysis, modality="image"):
    before_image_prompt = f"""
You are a Prompt-Improvement Agent specializing in multimodal prompt optimization. Your task is to design improved prompts for both image generation and text instruction, aimed at enhancing the performance of Multimodal Large Language Model (MLLM).

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Provided Material
- Text Prompt: {text_prompt}
- Image Prompt: 
""".strip()
    after_image_prompt = f"""
- Wrong Examples: 
""".strip()
    after_example_prompt = f"""    
- Failure Analysis: {analysis}
### Your Task
Your task is review the failure analysis carefully to understand the issues and create two improved prompts that directly address the issues in the failure analysis:
1. Image Generation Prompt
   - Write a detailed prompt for an image generator.
   - Enhance or redesign the reference image to resolve issues found in the analysis.
   - Ensure the image highlights critical visual features necessary for success.
    - If no reference image is provided, suggest an appropriate one based on the failure analysis.

2. Improved Text Prompt
   - Write a clear, concise, and unambiguous instruction for the MLLM.
   - Resolve ambiguities found in the failure analysis.
   - Elaborate how the reference image should be interpreted.
""".strip()
    output_format = """
### Output Format
<image_generation_prompt>{image_generation_prompt}</image_generation_prompt>
<improved_text_prompt>{improved_text_prompt}</improved_text_prompt>
Return ONLY the two XML tags above with non-empty content. Do not include markdown, code fences, or extra text.
""".strip()
    mm_prompt_modality = check_mm_type(mm_prompt_path) if mm_prompt_path else "text"
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_image_prompt},
                (
                    {"type": mm_prompt_modality, mm_prompt_modality: mm_prompt_path}
                    if mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_image_prompt},
                *example_prompt,
                {"type": "text", "text": after_example_prompt},
                {"type": "text", "text": output_format},
            ],
        },
    ]

    return prompt


### MPO Edit
def get_multimodal_edit_prompt(text_prompt, mm_prompt_path, example_prompt, analysis, modality="image"):
    before_image_prompt = f"""
You are a Prompt-Improvement Agent specializing in multimodal prompt optimization, with a focus on prompt editing. Your task is to design improved prompts for both image editing and text instruction, aimed at enhancing the performance of Multimodal Large Language Model (MLLM).

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Provided Material
- Text Prompt: {text_prompt}
- Image Prompt: 
""".strip()
    after_image_prompt = f"""
- Wrong Examples: 
""".strip()
    after_example_prompt = f"""
- Failure Analysis: {analysis}
### Your Task
Your task is review the failure analysis carefully to understand the issues and create two improved prompts that directly address the issues in the failure analysis:
1. Image Editing Prompt:
   - Write a precise and context-aware prompt instructing the image editor to modify the given reference image.
   - Specify which visual components (e.g., objects, colors, textures, lighting, perspective, composition) should be added, removed, or replaced based on the failure analysis.
   - Clearly identify any undesirable visual elements that led to the failure.
   - Guide the editor on how to retain key features, proportions, or stylistic elements that are critical to the intended outcome.

2. Improved Text Prompt
   - Write a clear, concise, and unambiguous instruction for the MLLM.
   - Resolve ambiguities found in the failure analysis.
   - Elaborate how the reference image should be interpreted.
""".strip()
    output_format = """
### Output Format
<image_edit_prompt>{image_edit_prompt}</image_edit_prompt>
<improved_text_prompt>{improved_text_prompt}</improved_text_prompt>
Return ONLY the two XML tags above with non-empty content. Do not include markdown, code fences, or extra text.
""".strip()

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_image_prompt},
                (
                    {"type": "image", "image": mm_prompt_path}
                    if mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_image_prompt},
                *example_prompt,
                {"type": "text", "text": after_example_prompt},
                {"type": "text", "text": output_format},
            ],
        },
    ]
    return prompt


# MPO Mix
def get_multimodal_improvement_mix_prompt(parents, analyses, example_prompts, modality="image"):
    assert len(parents) == 2 and len(analyses) == 2 and len(example_prompts) == 2
    before_imageA_prompt = f"""
You are a Prompt-Improvement Agent specializing in multimodal prompt optimization, with a focus on cross-prompt fusion. Your task is to create improved, mixed prompts for both image prompt and text instruction, aimed at enhancing the performance of Multimodal Large Language Model (MLLM).

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Provided Material
#### Prompt A
- Text Prompt A: {parents[0].instruction}
- Image Prompt A:
"""

    after_imageA_prompt = f"""
- Wrong Examples from Prompt A: 
""".strip()

    after_imageA_examples_prompt = f"""
- Failure Analysis for Prompt A: {analyses[0]}

#### Prompt B
- Text Prompt B: {parents[1].instruction}
- Image Prompt B: 
""".strip()
    after_imageB_prompt = f"""
- Wrong Examples from Prompt B: 
""".strip()
    after_imageB_examples_prompt = f"""
- Failure Analysis for Prompt B: {analyses[1]}

### Your Task
Your task is review the failure analysis carefully to understand the issues and create two improved prompts that directly address the issues in the failure analysis:
1. Image Mixing Prompt:
    - Write a guidance for the image generator to combine and improve both reference images.
    - Address visual issues identified in both failure analyses.
    - Guide the model to create a new hybrid image that merges key beneficial visual features from both references while mitigating their weaknesses.
    - Explicitly state which visual elements from each image should be retained, modified, or discarded to achieve task success.

2. Improved Text Prompt
   - Write a clear, concise, and unambiguous instruction for the MLLM.
   - Incorporate key visual or task-relevant features identified in both failure analysis.
   - Explain how the reference image should be used to assist the task.
""".strip()
    output_format = """
### Output Format
<image_mixing_prompt>{image_mixing_prompt}</image_mixing_prompt>
<mixed_text_prompt>{mixed_text_prompt}</mixed_text_prompt>
Return ONLY the two XML tags above with non-empty content. Do not include markdown, code fences, or extra text.
""".strip()

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_imageA_prompt},
                (
                    {"type": "image", "image": parents[0].mm_prompt_path}
                    if parents[0].mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_imageA_prompt},
                *example_prompts[0],
                {"type": "text", "text": after_imageA_examples_prompt},
                (
                    {"type": "image", "image": parents[1].mm_prompt_path}
                    if parents[1].mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_imageB_prompt},
                *example_prompts[1],
                {"type": "text", "text": after_imageB_examples_prompt},
                {"type": "text", "text": output_format},
            ],
        },
    ]

    return prompt


class OptimizationModel:
    def __init__(self, optim_model_setting, mm_generator: MMGenerator, task: BaseTask, logger):
        self.model = get_language_model(optim_model_setting["model_name"])(**optim_model_setting)
        self.mm_generator = mm_generator
        self.mm_generator_modality = self.mm_generator.target_modality  # "image" or "video" or "molecule"
        self.task = task
        self.logger = logger

    def _clean_response(self, optim_response, tag_name):
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        matches = re.findall(pattern, optim_response, re.DOTALL)
        matches = [m.strip() for m in matches]
        return matches[0] if matches else None

    @staticmethod
    def _is_missing_prompt(prompt):
        if prompt is None:
            return True
        if not isinstance(prompt, str):
            return False
        normalized = prompt.strip().lower()
        return normalized in {"", "none", "null", "n/a", "na"}

    def _extract_first_tag_value(self, optim_response, candidate_tags):
        for tag in candidate_tags:
            value = self._clean_response(optim_response, tag)
            if not self._is_missing_prompt(value):
                return value
        markdown_fallback = self._extract_markdown_prompt_block(optim_response, candidate_tags)
        if not self._is_missing_prompt(markdown_fallback):
            return markdown_fallback
        plaintext_fallback = self._extract_plaintext_prompt_block(optim_response, candidate_tags)
        if not self._is_missing_prompt(plaintext_fallback):
            return plaintext_fallback
        return None

    def _extract_markdown_prompt_block(self, optim_response, candidate_tags):
        """
        Recover prompt text when the optimizer ignores XML tags and emits markdown headers.
        """
        heading_aliases = {
            "image_generation_prompt": ["image generation prompt", "generation prompt", "image prompt"],
            "image_edit_prompt": ["image edit prompt", "editing prompt", "image prompt"],
            "image_mixing_prompt": ["image mixing prompt", "mixing prompt", "image prompt"],
            "image_mix_prompt": ["image mixing prompt", "mixing prompt", "image prompt"],
            "image_prompt": ["image prompt", "generation prompt", "editing prompt", "mixing prompt"],
            "improved_text_prompt": ["improved text prompt", "text prompt"],
            "mixed_text_prompt": ["mixed text prompt", "improved text prompt", "text prompt"],
            "text_prompt": ["text prompt", "improved text prompt", "mixed text prompt"],
        }

        normalized_targets = set()
        for tag in candidate_tags:
            normalized_targets.add(tag.lower())
            normalized_targets.update(heading_aliases.get(tag.lower(), []))

        lines = optim_response.splitlines()
        start_idx = None
        for idx, line in enumerate(lines):
            clean = line.strip().lower().lstrip("#").strip()
            clean = clean.rstrip(":").strip()
            if clean in normalized_targets:
                start_idx = idx + 1
                break

        if start_idx is None:
            return None

        block = []
        for idx in range(start_idx, len(lines)):
            cur = lines[idx].strip()
            # Next markdown section starts.
            if cur.startswith("#"):
                break
            # Stop at XML section openings if model mixed formats.
            if re.match(r"^<[^/][^>]*>$", cur):
                break
            block.append(lines[idx])

        value = "\n".join(block).strip()
        return value if value else None

    def _extract_plaintext_prompt_block(self, optim_response, candidate_tags):
        """
        Recover prompt text from blocks like:
        (image_edit_prompt)
        ...
        (improved_text_prompt)
        ...
        """
        marker_aliases = {
            "image_generation_prompt": {"image_generation_prompt", "image_prompt"},
            "image_edit_prompt": {"image_edit_prompt", "image_prompt"},
            "image_mixing_prompt": {"image_mixing_prompt", "image_mix_prompt", "image_prompt"},
            "image_mix_prompt": {"image_mixing_prompt", "image_mix_prompt", "image_prompt"},
            "image_prompt": {"image_prompt", "image_generation_prompt", "image_edit_prompt", "image_mixing_prompt"},
            "improved_text_prompt": {"improved_text_prompt", "text_prompt"},
            "mixed_text_prompt": {"mixed_text_prompt", "text_prompt", "improved_text_prompt"},
            "text_prompt": {"text_prompt", "improved_text_prompt", "mixed_text_prompt"},
        }
        markers = set()
        for tag in candidate_tags:
            t = tag.lower()
            markers.add(t)
            markers.update(marker_aliases.get(t, set()))

        lines = optim_response.splitlines()
        start_idx = None
        for i, line in enumerate(lines):
            m = re.match(r"^\(\s*([a-zA-Z0-9_]+)\s*\)\s*$", line.strip())
            if m and m.group(1).lower() in markers:
                start_idx = i + 1
                break
        if start_idx is None:
            return None

        block = []
        for i in range(start_idx, len(lines)):
            cur = lines[i].strip()
            if re.match(r"^\(\s*[a-zA-Z0-9_]+\s*\)\s*$", cur):
                break
            if cur.startswith("```"):
                continue
            block.append(lines[i])
        value = "\n".join(block).strip()
        return value if value else None

    def _repair_structured_output(self, response, text_tag, mm_tag, op_label):
        repair_prompt = [
            {
                "role": "user",
                "content": (
                    "Rewrite the following response to match the required XML format exactly. "
                    "Do not add markdown, code fences, or explanations.\n\n"
                    f"Required format:\n<{mm_tag}>...</{mm_tag}>\n<{text_tag}>...</{text_tag}>\n\n"
                    f"Response to rewrite:\n{response}"
                ),
            }
        ]
        repaired = self.model.generate(repair_prompt)
        self.logger.info(f"{op_label}: attempted structured-output repair pass.")
        return repaired

    def _generate_structured_pair(self, prompt, text_tags, mm_tags, op_label):
        primary_text_tag = text_tags[0]
        primary_mm_tag = mm_tags[0]
        response = self.model.generate(prompt)
        self.log_information(prompt, response)

        for attempt in range(3):
            text_prompt = self._extract_first_tag_value(response, text_tags)
            mm_prompt = self._extract_first_tag_value(response, mm_tags)
            if not self._is_missing_prompt(text_prompt) and not self._is_missing_prompt(mm_prompt):
                return text_prompt, mm_prompt
            if attempt < 2:
                response = self._repair_structured_output(response, primary_text_tag, primary_mm_tag, op_label)
            else:
                raise RuntimeError(
                    f"{op_label}: optimizer output missing required tags after retries. "
                    f"Required tags: <{primary_mm_tag}> and <{primary_text_tag}>."
                )

    # MPO: Cohesive Backpropagation
    def mpo_failure_analysis(self, node, example_prompt):
        analysis_prompt = get_multimodal_analysis_prompt(
            node.instruction,
            node.mm_prompt_path,
            example_prompt,
            modality=self.mm_generator_modality,
        )
        analysis = self.model.generate(analysis_prompt)

        self.log_information(analysis_prompt, analysis)

        return analysis

    # MPO: Generation Operator
    def mpo_optim_generation(self, node, model_responses_num):
        examples = node.get_wrong_examples(model_responses_num)
        # vLLM often limits image items per prompt (e.g. <=2).
        # Generation/Edit prompts may already include one reference image.
        max_example_mm_items = 1 if node.mm_prompt_path else 2
        example_prompt = self.get_example_prompt(
            examples, is_response=True, max_example_mm_items=max_example_mm_items
        )

        # Failure Analysis
        analysis = self.mpo_failure_analysis(node, example_prompt)

        # Prompt Optimization
        generate_prompt = get_multimodal_generation_prompt(
            text_prompt=node.instruction,
            mm_prompt_path=node.mm_prompt_path,
            example_prompt=example_prompt,
            analysis=analysis,
            modality=self.mm_generator_modality,
        )
        improved_text_prompt, mm_condition_prompt = self._generate_structured_pair(
            prompt=generate_prompt,
            text_tags=["improved_text_prompt", "text_prompt"],
            mm_tags=[
                f"{self.mm_generator_modality}_generation_prompt",
                "image_generation_prompt",
                "image_prompt",
            ],
            op_label="Generation operator",
        )

        # Generate MultiModal Data
        generated_mm_data = self.generate_mm(
            mm_condition_prompt,
            text_prompt=improved_text_prompt,
        )
        return improved_text_prompt, generated_mm_data

    def generate_mm(self, mm_condition_prompt: str, text_prompt: str = None) -> dict:
        if self._is_missing_prompt(mm_condition_prompt):
            raise ValueError("generate_mm: invalid multimodal condition prompt from optimizer.")
        generated_mm_path = self.mm_generator(mm_condition_prompt, text_prompt=text_prompt)

        return {
            "mm_condition_prompt": mm_condition_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    # MPO: Edit Operator
    def mpo_optim_edit(self, node, model_responses_num):
        examples = node.get_wrong_examples(model_responses_num)
        max_example_mm_items = 1 if node.mm_prompt_path else 2
        example_prompt = self.get_example_prompt(
            examples, is_response=True, max_example_mm_items=max_example_mm_items
        )

        # Failure Analysis
        analysis = self.mpo_failure_analysis(node, example_prompt)

        # Prompt Optimization
        edit_prompt = get_multimodal_edit_prompt(
            text_prompt=node.instruction,
            mm_prompt_path=node.mm_prompt_path,
            example_prompt=example_prompt,
            analysis=analysis,
            modality=self.mm_generator_modality,
        )

        improved_text_prompt, mm_edit_prompt = self._generate_structured_pair(
            prompt=edit_prompt,
            text_tags=["improved_text_prompt", "text_prompt"],
            mm_tags=[
                f"{self.mm_generator_modality}_edit_prompt",
                "image_edit_prompt",
                "image_prompt",
            ],
            op_label="Edit operator",
        )

        # Generate MultiModal Data
        generated_mm_data = self.edit_mm(
            mm_edit_prompt,
            mm_prompt_path=node.mm_prompt_path,
            text_prompt=improved_text_prompt,
        )
        return improved_text_prompt, generated_mm_data

    def edit_mm(self, mm_edit_prompt: str, mm_prompt_path, text_prompt: str = None):
        assert self.mm_generator_modality in ["image", "molecule"]
        generated_mm_path = self.mm_generator(mm_edit_prompt, mm_prompt_path=mm_prompt_path, text_prompt=text_prompt)

        return {
            "mm_condition_prompt": mm_edit_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    # MPO: Mix Operator
    def mpo_optim_mix(self, parents, model_responses_num):
        analyses, example_prompts = [], []
        for parent in parents:
            examples = parent.get_wrong_examples(model_responses_num)
            # Mix prompts already include two parent images; do not attach example
            # images or we exceed vLLM image-per-prompt limits.
            example_prompt = self.get_example_prompt(examples, is_response=True, max_example_mm_items=0)
            example_prompts.append(example_prompt)

            analysis = self.mpo_failure_analysis(parent, example_prompt)
            analyses.append(analysis)

        mix_prompt = get_multimodal_improvement_mix_prompt(
            parents=parents, analyses=analyses, example_prompts=example_prompts, modality=self.mm_generator_modality
        )

        improved_text_prompts, mm_mix_prompt = self._generate_structured_pair(
            prompt=mix_prompt,
            text_tags=["mixed_text_prompt", "improved_text_prompt", "text_prompt"],
            mm_tags=[
                f"{self.mm_generator_modality}_mixing_prompt",
                "image_mixing_prompt",
                "image_mix_prompt",
                "image_prompt",
            ],
            op_label="Mix operator",
        )

        # Generate Multimodal Data
        generated_mm_data = self.mix_mm(parents, mm_mix_prompt)
        return improved_text_prompts, generated_mm_data

    def mix_mm(
        self,
        parents,
        mm_mix_prompt,
    ):
        generated_mm_path = self.mm_generator.multimodal_mixing(
            parents=parents,
            mm_mix_prompt=mm_mix_prompt,
        )

        return {
            "mm_condition_prompt": mm_mix_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    def log_information(self, generate_prompt, response: str) -> None:
        self.logger.info("=" * 80)
        total_prompt = ""
        for role_content in generate_prompt:
            if isinstance(role_content["content"], list):
                for item in role_content["content"]:
                    item_type = item.get("type")
                    if item_type == "text":
                        total_prompt += f'{item["text"]}\n'
                    elif item_type in {"image", "video"}:
                        abs_path = Path(item[item_type]).resolve()
                        total_prompt += f"{abs_path}\n"
                    elif item_type == "molecule":
                        total_prompt += f"{item['molecule']['smiles'][0]}\n"
            else:
                total_prompt += f'{role_content["role"]}\n{role_content["content"]}\n'

        self.logger.info(f"{total_prompt}\n{'-' * 80}\n{response}\n\n")

    def get_example_prompt(self, examples, is_response=True, max_example_mm_items=0):
        example_prompt = []
        used_mm_items = 0
        for example in examples:
            example_string = self._get_example_string(example, is_response)
            query_mm_path = self.task.get_mm_path(example)
            query_mm_type = check_mm_type(query_mm_path) if query_mm_path else "text"
            include_query_mm = (
                query_mm_path is not None
                and query_mm_type == "image"
                and used_mm_items < max_example_mm_items
            )
            example_content = [
                {"type": "text", "text": f"<Example>\n{self.task.get_query(example)}\n"},
                ({"type": query_mm_type, query_mm_type: query_mm_path} if include_query_mm else None),
                {"type": "text", "text": "\n"},
                {"type": "text", "text": f"{example_string}\n</Example>\n"},
            ]
            example_prompt.extend([item for item in example_content if item is not None])
            if include_query_mm:
                used_mm_items += 1

        return example_prompt

    def _format_answer(self, example):
        answer = self.task.get_answer(example)
        if isinstance(answer, list):
            return ", ".join(map(str, answer))
        return str(answer)

    def _get_example_string(self, example, is_response=True):
        # Format example text content
        if is_response:
            example_string = f'Response: \n{example["response"]}\n\nModel answer: \n{example["model_answer"]}\n\nThe correct answer is : \n{self._format_answer(example)}'
        else:
            example_string = f"The Answer is \n{self._format_answer(example)}"
        return example_string
