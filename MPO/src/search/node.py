import itertools
import os
import random
from typing import List, Optional

from ..tasks import BaseTask


GUIDELINES_PLACEHOLDER = "{{GUIDELINES}}"


def format_guideline(idx: int, g: dict) -> str:
    """Render a single rule dict into a structured display block.

    Supports two rule types:
      - "general"               (default): WHEN/THEN/Wrong/Correct
      - "class_discrimination":  features per class for fine-grained classification
    """
    rule_type = (g.get("type") or "general").strip()
    name = (g.get("name") or f"Rule {idx}").strip()
    priority = (g.get("priority") or "").strip()

    header_bits = [f"Rule {idx}: {name}"]
    meta_bits = []
    if priority:
        meta_bits.append(f"priority={priority}")
    if rule_type and rule_type != "general":
        meta_bits.append(f"type={rule_type}")
    if meta_bits:
        header_bits.append(f"  ({', '.join(meta_bits)})")
    parts = ["".join(header_bits)]

    if rule_type == "class_discrimination":
        distinguishes = g.get("distinguishes") or []
        if isinstance(distinguishes, list) and distinguishes:
            parts.append(f"  Distinguishes: {' vs '.join(map(str, distinguishes))}")
        features = g.get("features") or []
        if isinstance(features, list) and features:
            parts.append("  Features:")
            for feat in features:
                if not isinstance(feat, dict):
                    continue
                feat_name = (feat.get("name") or "feature").strip()
                class_vals = [(k, v) for k, v in feat.items() if k != "name" and v]
                if class_vals:
                    val_str = "; ".join(f"{cls} = {val}" for cls, val in class_vals)
                    parts.append(f"    - {feat_name}: {val_str}")
        decision_order = (g.get("decision_order") or "").strip()
        if decision_order:
            parts.append(f"  Decision order: {decision_order}")
        ex_wrong = (g.get("example_wrong") or "").strip()
        ex_correct = (g.get("example_correct") or "").strip()
        if ex_wrong:
            parts.append(f"  Wrong:   {ex_wrong}")
        if ex_correct:
            parts.append(f"  Correct: {ex_correct}")
        return "\n".join(parts)

    # General rule (the original schema)
    condition = (g.get("condition") or "").strip()
    action = (g.get("action") or "").strip()
    ex_wrong = (g.get("example_wrong") or "").strip()
    ex_correct = (g.get("example_correct") or "").strip()
    if condition:
        parts.append(f"  WHEN {condition}")
    if action:
        parts.append(f"  THEN {action}")
    if ex_wrong:
        parts.append(f"  Wrong:   {ex_wrong}")
    if ex_correct:
        parts.append(f"  Correct: {ex_correct}")
    return "\n".join(parts)


def render_guidelines_block(guidelines: List[dict]) -> str:
    if not guidelines:
        return "(no task-specific guidelines yet)"
    return "\n\n".join(format_guideline(i + 1, g) for i, g in enumerate(guidelines))


def render_with_guidelines(base_instruction: str, guidelines: List[dict]) -> str:
    """
    Insert the rendered guidelines block into the placeholder location of base_instruction.
    If no placeholder is present, append the guidelines as a labeled section at the end.
    """
    if base_instruction is None:
        return ""
    block = render_guidelines_block(guidelines)
    if GUIDELINES_PLACEHOLDER in base_instruction:
        return base_instruction.replace(GUIDELINES_PLACEHOLDER, block)
    return f"{base_instruction.rstrip()}\n\n### Task Guidelines\n{block}"


class Node:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        instruction: str,
        mm_prompt_path: str = None,
        mm_condition_prompt: str = None,
        task: BaseTask = None,
        parents: Optional[List["Node"]] = None,
        train_metric: float = -1,
        test_metric: float = -1,
        action_type: str = None,
        # GMPO additions (optional; unused by the original MPO path).
        base_instruction: str = None,
        guidelines: Optional[List[dict]] = None,
    ):
        self.id = next(Node.id_iter)
        self.instruction = instruction
        self.parents = parents
        self.train_metric = train_metric
        self.test_metric = test_metric
        self.mm_prompt_path = mm_prompt_path
        self.mm_condition_prompt = mm_condition_prompt
        self.action_type = action_type

        # GMPO-specific state. For original MPO nodes these stay at their defaults
        # and are simply ignored.
        self.base_instruction = base_instruction
        self.guidelines = list(guidelines) if guidelines is not None else []

        if parents is None:
            self.depth = 0
            assert task is not None
            self.task = task
        else:
            self.depth = max(parent.depth for parent in parents) + 1
            self.task = parents[0].task

    def update_model_wrong_example(self, examples):
        self.model_wrong_examples = []
        self.model_wrong_examples.extend(examples)

    def update_model_correct_example(self, examples):
        self.model_correct_examples = []
        self.model_correct_examples.extend(examples)

    def get_wrong_examples(self, model_responses_num: int):
        num_wrong_examples = len(self.model_wrong_examples)
        if num_wrong_examples < model_responses_num:
            sampled_examples = self.model_wrong_examples
        else:
            sampled_examples = random.sample(self.model_wrong_examples, model_responses_num)
        return sampled_examples

    def render_instruction(self) -> str:
        """Re-render `instruction` from base_instruction + guidelines (GMPO nodes only)."""
        if self.base_instruction is None:
            return self.instruction
        return render_with_guidelines(self.base_instruction, self.guidelines)

    def to_dict(self):
        if self.mm_prompt_path is not None and isinstance(self.mm_prompt_path, str):
            log_mm_prompt_path = os.path.abspath(self.mm_prompt_path)
        else:
            log_mm_prompt_path = self.mm_prompt_path

        return {
            "id": self.id,
            "task": self.task.task_name,
            "instruction": self.instruction,
            "mm_prompt_path": log_mm_prompt_path,
            "parent_id": [parent.id for parent in self.parents] if self.parents else None,
            "depth": self.depth,
            "train_metric": self.train_metric,
            "test_metric": self.test_metric,
            "mm_condition_prompt": self.mm_condition_prompt,
            "action_type": self.action_type,
            # GMPO fields (None / [] for original MPO nodes — harmless in JSON).
            "base_instruction": self.base_instruction,
            "guidelines": self.guidelines,
        }