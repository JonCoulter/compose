import ast
import json
import re
from typing import List, Optional, Tuple

import numpy as np

from ..base_model import BaseModel
from ..optim_model import OptimizationModel
from ..tasks import BaseTask
from ..utils import check_mm_type
from .base_search import BaseSearch
from .node import GUIDELINES_PLACEHOLDER, Node, render_with_guidelines


DEFAULT_GUIDELINE_OPERATORS = ["append", "update", "mix"]
OPTIMIZER_MAX_EXAMPLE_IMAGES = 2

# Phrases banned from rule bodies. The optimizer's natural failure mode on
# weaker models is to write "re-evaluate carefully" rules that say nothing.
BANNED_VAGUE_PHRASES = (
    "re-evaluate",
    "reevaluate",
    "reconsider",
    "look carefully",
    "look more carefully",
    "pay attention",
    "pay closer attention",
    "select the closest match",
    "select the closest",
    "match the closest",
    "match the given choices",
    "match features to choices",
    "be careful",
    "double-check",
    "double check",
)

# ---------------------------------------------------------------------------
# Class-pair coverage helpers (used when class_choices is non-None)
# ---------------------------------------------------------------------------

def _all_class_pairs(class_choices: List[str]) -> List[tuple]:
    """All unordered pairs of class names."""
    pairs = []
    for i in range(len(class_choices)):
        for j in range(i + 1, len(class_choices)):
            pairs.append(tuple(sorted([class_choices[i], class_choices[j]])))
    return pairs


def _rule_covers_pairs(rule: dict) -> set:
    """Set of unordered class-pairs covered by a single rule, or empty set."""
    if not isinstance(rule, dict):
        return set()
    if rule.get("type") != "class_discrimination":
        return set()
    distinguishes = rule.get("distinguishes") or []
    if not isinstance(distinguishes, list) or len(distinguishes) < 2:
        return set()
    classes = [str(c) for c in distinguishes if isinstance(c, str)]
    pairs = set()
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            pairs.add(tuple(sorted([classes[i], classes[j]])))
    return pairs


def _ruleset_coverage(rules: List[dict]) -> set:
    """Union of class-pairs covered by all rules in the set."""
    covered = set()
    for r in rules:
        covered |= _rule_covers_pairs(r)
    return covered


def _format_coverage_status(rules: List[dict], class_choices: List[str]) -> str:
    """Render a human-readable coverage status block for inclusion in synthesis prompts."""
    all_pairs = set(_all_class_pairs(class_choices))
    covered = _ruleset_coverage(rules)
    uncovered = sorted(all_pairs - covered)
    if not uncovered:
        return (
            "### Class-Pair Coverage\n"
            "All class pairs are already covered by existing rules. Your new rule may sharpen "
            "an existing distinction or address a feature confusion identified by the analysis."
        )
    uncovered_str = "; ".join(f"({a}, {b})" for a, b in uncovered)
    return (
        "### Class-Pair Coverage (CRITICAL)\n"
        f"The following class pairs are NOT yet distinguished by any rule: {uncovered_str}.\n"
        "Your new rule MUST distinguish at least one of these uncovered pairs by including "
        "those classes in `distinguishes` and providing concrete per-class feature values for "
        "each of them. Closing an uncovered pair is the highest priority for this step."
    )

# ---------------------------------------------------------------------------
# Class-choice detection
# ---------------------------------------------------------------------------

def detect_class_choices(task: BaseTask) -> Optional[List[str]]:
    """
    Return a list of class names if this looks like a closed-set classification task,
    else None. Tries (in order):
      1. task.class_choices attribute
      2. task.labels attribute (common on classification tasks)
      3. parsing the first few train examples' queries for a python list literal
      4. parsing the initial prompt for a python list literal
    """
    # 1. Explicit class_choices
    explicit = getattr(task, "class_choices", None)
    if isinstance(explicit, (list, tuple)) and len(explicit) >= 2:
        return [str(c) for c in explicit]

    # 2. task.labels (cuckoo and similar tasks expose this)
    labels = getattr(task, "labels", None)
    if isinstance(labels, (list, tuple)) and len(labels) >= 2 and all(isinstance(v, str) for v in labels):
        return list(labels)

    # 3. Parse train examples' query strings.
    train_data = getattr(task, "train_data", None) or []
    for example in train_data[: min(5, len(train_data))]:
        try:
            query = task.get_query(example)
        except Exception:
            continue
        choices = _parse_choice_list(query)
        if choices is not None:
            return choices

    # 4. Parse initial prompt as last resort.
    prompt = getattr(task, "initial_prompt", None)
    if prompt:
        choices = _parse_choice_list(prompt)
        if choices is not None:
            return choices

    return None


def _parse_choice_list(text: str) -> Optional[List[str]]:
    """
    Find the first python-list-of-strings literal in `text` that has >=2 items.
    Skips short bracketed tokens like '[Choice]' that aren't list literals.
    """
    if not isinstance(text, str):
        return None
    # `[^\[\]]+` ensures we don't match nested brackets; DOTALL lets us span newlines
    # which can appear inside multi-line choice lists.
    for match in re.finditer(r"\[[^\[\]]+\]", text, re.DOTALL):
        snippet = match.group(0)
        try:
            value = ast.literal_eval(snippet)
        except Exception:
            continue
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(v, str) for v in value):
            return value
    return None


# ---------------------------------------------------------------------------
# Schemas and renderers
# ---------------------------------------------------------------------------

GENERAL_RULE_SCHEMA = (
    '{\n'
    '  "type": "general",\n'
    '  "name": "<short, descriptive name>",\n'
    '  "priority": "High|Medium|Low",\n'
    '  "category": "<short category>",\n'
    '  "addresses": "<which failure pattern this addresses>",\n'
    '  "condition": "<WHEN clause: a concrete, observable trigger>",\n'
    '  "action":    "<THEN clause: a concrete action that names what to look at or how to decide>",\n'
    '  "example_wrong":   "<observed wrong behavior>",\n'
    '  "example_correct": "<correct behavior>"\n'
    '}'
)


def class_discrim_rule_schema(class_choices: List[str]) -> str:
    classes_csv = ", ".join(repr(c) for c in class_choices)
    return (
        '{\n'
        '  "type": "class_discrimination",\n'
        '  "name": "<short, descriptive name>",\n'
        '  "priority": "High|Medium|Low",\n'
        f'  "distinguishes": [<a subset of the available class names: {classes_csv}; at least 2>],\n'
        '  "features": [\n'
        '    {\n'
        '      "name": "<observable feature, e.g. bill color, tail length, eye-ring>",\n'
        '      "<exact class name 1>": "<concrete value of this feature for class 1>",\n'
        '      "<exact class name 2>": "<concrete value of this feature for class 2>"\n'
        '    }\n'
        '    // include 2 or more feature objects\n'
        '  ],\n'
        '  "decision_order": "<which feature to check first; how to break ties>",\n'
        '  "addresses": "<which failure pattern this addresses>",\n'
        '  "example_wrong":   "<observed wrong behavior>",\n'
        '  "example_correct": "<correct behavior>"\n'
        '}'
    )


def _format_guidelines_for_prompt(guidelines: List[dict]) -> str:
    if not guidelines:
        return "(none yet)"
    from .node import format_guideline
    return "\n\n".join(format_guideline(i + 1, g) for i, g in enumerate(guidelines))


def _format_single_example_text(idx: int, ex: dict, task: BaseTask) -> str:
    query = task.get_query(ex)
    ans = task.get_answer(ex)
    if isinstance(ans, list):
        ans = ", ".join(map(str, ans))
    return (
        f"<Example {idx}>\n"
        f"Query: {query}\n"
        f"Model Response: {ex.get('response', '')}\n"
        f"Model Answer:   {ex.get('model_answer', '')}\n"
        f"Correct Answer: {ans}\n"
        f"</Example {idx}>"
    )


def _build_examples_content(examples, task: BaseTask, sees_images: str,
                            image_budget: int) -> List[dict]:
    if not examples:
        return [{"type": "text", "text": "(no failure examples available)"}]
    if sees_images == "none":
        text = "\n\n".join(_format_single_example_text(i + 1, ex, task) for i, ex in enumerate(examples))
        return [{"type": "text", "text": text}]
    content: List[dict] = []
    used_images = 0
    for i, ex in enumerate(examples, 1):
        text_block = _format_single_example_text(i, ex, task)
        mm_path = task.get_mm_path(ex)
        attach = False
        if used_images < image_budget and mm_path is not None:
            try:
                if check_mm_type(mm_path) == "image":
                    attach = True
            except Exception:
                attach = False
        if attach:
            content.append({"type": "text", "text": text_block + "\n[query image of this failure]:"})
            content.append({"type": "image", "image": mm_path})
            used_images += 1
        else:
            content.append({"type": "text", "text": text_block})
    return content


# ---------------------------------------------------------------------------
# Prompt builders — stage 1 (failure analysis) and stage 2 (synthesis)
# ---------------------------------------------------------------------------

def _refine_initial_prompt(initial_prompt: str, sample_block: str) -> list:
    return [{
        "role": "user",
        "content": (
            "You are an expert prompt engineer. Given an initial task prompt and a few examples "
            "of the task, rewrite the prompt into a clear, comprehensive instruction for a "
            "multimodal language model.\n\n"
            "### Initial Prompt\n"
            f"{initial_prompt}\n\n"
            "### Task Examples\n"
            f"{sample_block}\n\n"
            "### Your Task\n"
            "1. Read the initial prompt and examples to understand the task.\n"
            "2. Write a clear, complete instruction that explains: what the task is, what input "
            "the model receives, and what output format is expected.\n"
            f"3. Insert the literal token {GUIDELINES_PLACEHOLDER} on its own line at the position "
            "where additional task-specific guidelines should be added later (typically right before "
            "the model is asked to produce its answer).\n\n"
            "### Output Format\n"
            "Return ONLY your refined instruction inside <refined_instruction>...</refined_instruction> "
            "tags. No markdown fences, no extra prose."
        ),
    }]


def _failure_analysis_prompt(base_instruction: str,
                             class_choices: Optional[List[str]],
                             examples_content: List[dict]) -> list:
    classes_line = (
        f"### Available class choices\n{class_choices}\n\n"
        if class_choices else ""
    )
    intro = (
        "You are a Failure Analysis Agent. You will be shown failure cases for a multimodal task. "
        "Your job is to look CAREFULLY at each case and describe what visible features in the input "
        "would have led to the correct answer, and which features the model probably misread.\n\n"
        "### Task Instruction\n"
        f"{base_instruction}\n\n"
        f"{classes_line}"
        "### Failure Cases\n"
    )
    footer = (
        "\n\n### Your Task\n"
        "For EACH failure case, write 3-5 short bullet points covering:\n"
        "- What you actually see in the input image (be CONCRETE: bill color, tail length, plumage "
        "  pattern, body posture, eye-ring, etc.).\n"
        "- Which observable feature(s) most strongly distinguish the correct answer from the "
        "  model's wrong answer. Name the feature and its value for BOTH classes.\n"
        "- What the model probably keyed on to produce the wrong answer.\n\n"
        "Be specific. Vague answers like 'the model misclassified the bird' or 'the model should "
        "look more carefully' are useless and will be rejected.\n\n"
        "### Output Format\n"
        "Return ONLY your analysis inside <analysis>...</analysis> tags. Bullet lists are fine."
    )
    content = [{"type": "text", "text": intro}]
    content.extend(examples_content)
    content.append({"type": "text", "text": footer})
    return [{"role": "user", "content": content}]


def _append_synthesis_prompt(base_instruction: str,
                             guidelines: List[dict],
                             class_choices: Optional[List[str]],
                             rule_mode: str,
                             analysis: str) -> list:
    classes_line = (
        f"### Available class choices\n{class_choices}\n\n"
        if class_choices and rule_mode in ("auto", "discrimination") else ""
    )
    use_discrim = rule_mode in ("auto", "discrimination") and bool(class_choices)
    schema = class_discrim_rule_schema(class_choices) if use_discrim else GENERAL_RULE_SCHEMA
    rule_type_label = "class-discrimination" if use_discrim else "general"

    coverage_block = (
        f"\n\n{_format_coverage_status(guidelines, class_choices)}"
        if use_discrim else ""
    )

    discrim_block = (
        "Write ONE class-discrimination rule that addresses the failures by enumerating concrete "
        "visual features that distinguish at least two of the listed classes.\n\n"
        "Hard requirements:\n"
        "1. The rule MUST name AT LEAST TWO observable features (e.g. bill color, tail length, "
        "eye-ring, body posture, plumage pattern). Behavioral words alone are not features.\n"
        "2. For each feature, you MUST give a concrete description of how that feature looks for "
        "EACH listed class in `distinguishes`. No empty cells.\n"
        "3. `distinguishes` MUST be a subset of the Available class choices listed above. If any "
        "class pair is listed as uncovered above, this rule SHOULD distinguish at least one such "
        "pair (include both classes from that pair in `distinguishes`).\n"
        "4. `decision_order` MUST be a concrete instruction (e.g. 'check bill color first; if "
        "ambiguous, check eye-ring color').\n"
        "5. Vague phrases are FORBIDDEN anywhere in the rule: 're-evaluate', 'reconsider', 'look "
        "carefully', 'pay attention', 'select the closest match', 'be careful', 'match features "
        "to choices'. Rules using these will be rejected.\n"
        "6. The rule MUST NOT duplicate or restate any existing rule.\n"
    ) if use_discrim else (
        "Write ONE rule that addresses the failures.\n\n"
        "Hard requirements:\n"
        "1. The `condition` MUST describe a concrete observable trigger.\n"
        "2. The `action` MUST name a concrete operation, criterion, or feature to check.\n"
        "3. Vague phrases are FORBIDDEN: 're-evaluate', 'reconsider', 'look carefully', 'pay "
        "attention', 'select the closest match', 'be careful', 'match features to choices'.\n"
        "4. The rule MUST NOT duplicate or restate any existing rule.\n"
    )

    return [{
        "role": "user",
        "content": (
            f"You are a Rule-Generation Agent. You will use a prior failure analysis to write ONE "
            f"structured {rule_type_label} rule that, when added to the prompt, would prevent the "
            "kind of failures described.\n\n"
            "### Task Instruction\n"
            f"{base_instruction}\n\n"
            f"{classes_line}"
            "### Existing Rules\n"
            f"{_format_guidelines_for_prompt(guidelines)}"
            f"{coverage_block}\n\n"
            "### Failure Analysis\n"
            f"{analysis}\n\n"
            "### Your Task\n"
            f"{discrim_block}\n"
            "### Output Format\n"
            "Return ONLY the rule as JSON inside <new_rule>...</new_rule> tags. No markdown "
            "fences, no extra text.\n\n"
            f"<new_rule>\n{schema}\n</new_rule>"
        ),
    }]


def _update_synthesis_prompt(base_instruction: str,
                             guidelines: List[dict],
                             class_choices: Optional[List[str]],
                             rule_mode: str,
                             analysis: str) -> list:
    classes_line = (
        f"### Available class choices\n{class_choices}\n\n"
        if class_choices and rule_mode in ("auto", "discrimination") else ""
    )
    use_discrim = rule_mode in ("auto", "discrimination") and bool(class_choices)
    schema = class_discrim_rule_schema(class_choices) if use_discrim else GENERAL_RULE_SCHEMA

    coverage_block = (
        f"\n\n{_format_coverage_status(guidelines, class_choices)}"
        if use_discrim else ""
    )

    upgrade_hint = (
        "If the chosen rule is currently a `general` rule and class choices are available, you "
        "should PREFER to rewrite it as a `class_discrimination` rule with concrete per-class "
        "feature values. That is the strongest form of 'more specific'.\n"
    ) if use_discrim else ""

    coverage_preserve_hint = (
        "IMPORTANT: The updated rule must NOT remove coverage of any class pair currently covered "
        "by the rule you replace. If the original rule's `distinguishes` list is [A, B], the "
        "updated rule's `distinguishes` must include at least [A, B] (it may add more classes).\n"
    ) if use_discrim else ""

    return [{
        "role": "user",
        "content": (
            "You are a Rule-Refinement Agent. Your job is to pick the MOST GENERIC existing rule "
            "and rewrite it so it is STRICTLY MORE SPECIFIC.\n\n"
            "### Task Instruction\n"
            f"{base_instruction}\n\n"
            f"{classes_line}"
            "### Existing Rules (1-indexed)\n"
            f"{_format_guidelines_for_prompt(guidelines)}"
            f"{coverage_block}\n\n"
            "### Failure Analysis\n"
            f"{analysis}\n\n"
            "### Your Task\n"
            "1. Pick the existing rule that is the MOST GENERIC. A rule is generic when it leans "
            "on words like 're-evaluate', 'reconsider', 'be careful', or fails to name concrete "
            "observable features.\n"
            "2. Rewrite it so that the new version:\n"
            "   - names AT LEAST one concrete observable feature that the original did not name, "
            "or sharpens at least one existing feature with class-specific values;\n"
            "   - removes all vague phrasing;\n"
            "   - is STRICTLY MORE INFORMATIVE than the original — a paraphrase is not acceptable.\n"
            f"{upgrade_hint}{coverage_preserve_hint}"
            "### Output Format\n"
            "Return ONLY:\n\n"
            "<target_rule_index>N</target_rule_index>\n"
            f"<updated_rule>\n{schema}\n</updated_rule>"
        ),
    }]


def _mix_synthesis_prompt(parent_a: Node, parent_b: Node,
                          class_choices: Optional[List[str]],
                          rule_mode: str,
                          analysis_a: str, analysis_b: str,
                          max_rules: int) -> list:
    classes_line = (
        f"### Available class choices\n{class_choices}\n\n"
        if class_choices and rule_mode in ("auto", "discrimination") else ""
    )
    use_discrim = rule_mode in ("auto", "discrimination") and bool(class_choices)
    schema = class_discrim_rule_schema(class_choices) if use_discrim else GENERAL_RULE_SCHEMA

    # Coverage status for the *combined* rule set (parent A's union parent B's rules).
    combined_rules = list(parent_a.guidelines) + list(parent_b.guidelines)
    coverage_block = (
        f"\n\n{_format_coverage_status(combined_rules, class_choices)}"
        if use_discrim else ""
    )

    coverage_preserve_hint = (
        "IMPORTANT: The merged rule set must cover EVERY class pair that is covered by either of "
        "the two parent sets. Do not drop coverage for any pair. If a pair is uncovered in both "
        "parents but identified above as needing coverage, your merged set should include a rule "
        "that closes it.\n"
    ) if use_discrim else ""

    return [{
        "role": "user",
        "content": (
            "You are a Rule-Set Fusion Agent. Merge two sets of task guidelines into a single, "
            "improved set, removing redundancy and combining the strongest, most specific rules.\n\n"
            "### Task Instruction\n"
            f"{parent_a.base_instruction}\n\n"
            f"{classes_line}"
            "### Guideline Set A\n"
            f"{_format_guidelines_for_prompt(parent_a.guidelines)}\n\n"
            "### Failure Analysis for Set A\n"
            f"{analysis_a}\n\n"
            "### Guideline Set B\n"
            f"{_format_guidelines_for_prompt(parent_b.guidelines)}\n\n"
            "### Failure Analysis for Set B\n"
            f"{analysis_b}"
            f"{coverage_block}\n\n"
            "### Your Task\n"
            f"Produce a single merged set of guidelines (at most {max_rules} rules) that:\n"
            "- keeps the most concrete, non-redundant rules from both sets;\n"
            "- merges overlapping rules into a single, sharper rule with more class-specific "
            "feature values;\n"
            "- discards rules that are vague or are contradicted by the failure evidence;\n"
            "- adds at most ONE NEW rule if a clear failure pattern is unaddressed.\n"
            f"{coverage_preserve_hint}\n"
            "### Output Format\n"
            "Return ONLY a JSON array of rule objects inside <merged_rules>...</merged_rules> tags. "
            "No markdown fences, no extra text. Each rule object follows this schema:\n"
            f"{schema}"
        ),
    }]

# ---------------------------------------------------------------------------
# Extraction and validation
# ---------------------------------------------------------------------------

def _extract_xml_block(text: str, tag: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _extract_json(text: Optional[str]):
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        idx = 0
        while idx < len(cleaned):
            start = cleaned.find(opener, idx)
            if start == -1:
                break
            depth = 0
            in_str = False
            esc = False
            end = -1
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                break
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                idx = start + 1
                continue
    return None


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def _has_banned_phrase(rule: dict) -> Optional[str]:
    """Return the banned phrase found in the rule, or None if clean."""
    for s in _walk_strings(rule):
        low = s.lower()
        for phrase in BANNED_VAGUE_PHRASES:
            if phrase in low:
                return phrase
    return None


def _validate_general_rule(rule) -> Tuple[bool, str]:
    if not isinstance(rule, dict):
        return False, "not a dict"
    if not rule.get("condition") or not rule.get("action"):
        return False, "missing condition/action"
    bad = _has_banned_phrase(rule)
    if bad:
        return False, f"banned phrase: {bad!r}"
    return True, ""


def _validate_class_discrim_rule(rule, class_choices: Optional[List[str]]) -> Tuple[bool, str]:
    if not isinstance(rule, dict):
        return False, "not a dict"
    distinguishes = rule.get("distinguishes")
    if not isinstance(distinguishes, list) or len(distinguishes) < 2:
        return False, "`distinguishes` must be a list of >= 2 class names"
    if class_choices is not None:
        unknown = [c for c in distinguishes if c not in class_choices]
        if unknown:
            return False, f"`distinguishes` contains unknown classes: {unknown}"
    features = rule.get("features")
    if not isinstance(features, list) or len(features) < 1:
        return False, "`features` must be a non-empty list"
    valid_feats = 0
    for feat in features:
        if not isinstance(feat, dict):
            continue
        per_class = [k for k in feat.keys() if k != "name" and feat.get(k)]
        if feat.get("name") and len(per_class) >= 2:
            valid_feats += 1
    if valid_feats < 1:
        return False, "no feature has >= 2 non-empty per-class values"
    bad = _has_banned_phrase(rule)
    if bad:
        return False, f"banned phrase: {bad!r}"
    return True, ""


def _validate_rule(rule, class_choices: Optional[List[str]]) -> Tuple[bool, str]:
    if not isinstance(rule, dict):
        return False, "not a dict"
    rtype = rule.get("type", "general")
    if rtype == "class_discrimination":
        return _validate_class_discrim_rule(rule, class_choices)
    return _validate_general_rule(rule)


# ---------------------------------------------------------------------------
# GMPO search
# ---------------------------------------------------------------------------

class GMPO(BaseSearch):
    """Guideline-based MPO with two-stage failure analysis + concreteness validation."""

    def __init__(
        self,
        task: BaseTask,
        base_model: BaseModel,
        optim_model: OptimizationModel,
        evaluator,
        log_dir: str,
        logger,
        method: str,
        beam_width: int,
        iteration: int,
        model_responses_num: int,
        gmpo_optim_sees_images: str = "query_only",
        gmpo_max_rules: int = 10,
        gmpo_rule_mode: str = "auto",
        operators: Optional[List[str]] = None,
        max_repair_attempts: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            base_model=base_model,
            optim_model=optim_model,
            log_dir=log_dir,
            logger=logger,
            method=method,
            iteration=iteration,
            model_responses_num=model_responses_num,
            **kwargs,
        )
        self.task = task
        self.evaluator = evaluator
        self.beam_width = beam_width
        self.max_rules = gmpo_max_rules
        self.operators = list(operators) if operators else list(DEFAULT_GUIDELINE_OPERATORS)
        self.max_repair_attempts = max_repair_attempts

        if gmpo_optim_sees_images not in ("none", "query_only"):
            raise ValueError(f"gmpo_optim_sees_images must be 'none' or 'query_only', got {gmpo_optim_sees_images!r}")
        if gmpo_rule_mode not in ("auto", "general", "discrimination"):
            raise ValueError(f"gmpo_rule_mode must be 'auto', 'general', or 'discrimination', got {gmpo_rule_mode!r}")
        self.optim_sees_images = gmpo_optim_sees_images
        self.rule_mode = gmpo_rule_mode

        self.class_choices = detect_class_choices(task)
        if self.rule_mode == "discrimination" and not self.class_choices:
            raise ValueError(
                "gmpo_rule_mode=discrimination but no class_choices could be detected on the task. "
                "Either set task.class_choices, ensure the initial prompt contains a python-list "
                "of class names, or use rule_mode=auto/general."
            )

        self.logger.info(
            f"[GMPO] optim_sees_images={self.optim_sees_images}, "
            f"max_rules={self.max_rules}, operators={self.operators}, "
            f"rule_mode={self.rule_mode}, class_choices={self.class_choices}"
        )

    # ----- entrypoint -----

    def train(self):
        return self.optimize_gmpo(self.task)

    def optimize_gmpo(self, task: BaseTask):
        base_instruction = self._refine_initial_instruction(task)
        self.logger.info(f"[GMPO] Refined base instruction:\n{base_instruction}\n")

        root = Node(
            instruction=render_with_guidelines(base_instruction, []),
            task=task,
            base_instruction=base_instruction,
            guidelines=[],
        )
        self.evaluate_node(node=root, split="train")
        self.logger.info(f"[GMPO] Root train_metric={root.train_metric}")

        nodes_tracker = self.initialize_nodes_tracker(root)

        inputs, action_types = self.get_action_types_and_inputs(it=-1, candidates=nodes_tracker["candidates"])
        self.logger.info(f"[GMPO] iter -1 actions: {action_types}")
        batch = self._generate_nodes_parallel_pairs(
            inputs=inputs,
            action_types=action_types,
            node_generation_func=self.action,
        )
        self.evaluator(batch)
        self.update_candidates(nodes_tracker, batch)

        for it in range(self.iteration - 1):
            inputs, action_types = self.get_action_types_and_inputs(it=it, candidates=nodes_tracker["candidates"])
            self.logger.info(f"[GMPO] iter {it} actions: {action_types}")
            batch = self._generate_nodes_parallel_pairs(
                inputs=inputs,
                action_types=action_types,
                node_generation_func=self.action,
            )
            self.evaluator(batch)
            self.update_candidates(nodes_tracker, batch)

        self.log_node_tracker(nodes_tracker, filename=f"{task.task_name}_gmpo")
        return nodes_tracker["updated"][-1][0].test_metric

    # ----- operator dispatch -----

    def get_action_types_and_inputs(self, it, candidates):
        num_actions = self.beam_width ** 2
        action_types: List[str] = []
        inputs: List[List[Node]] = []

        if it == -1:
            for _ in range(num_actions):
                action_types.append("append")
                inputs.append([candidates[0]])
            return inputs, action_types

        train_metrics = np.array([max(0.0, n.train_metric) for n in candidates], dtype=float)
        total = train_metrics.sum()
        prob = (np.ones_like(train_metrics) / len(train_metrics)) if total <= 0 else (train_metrics / total)

        for i in range(num_actions):
            op = self.operators[i % len(self.operators)]
            if op == "mix":
                if len(candidates) < 2:
                    op = "append"
                    parents = [candidates[0]]
                else:
                    parents = list(np.random.choice(candidates, size=2, p=prob, replace=False))
            else:
                parent = np.random.choice(candidates, p=prob)
                if op == "update" and not parent.guidelines:
                    op = "append"
                parents = [parent]
            action_types.append(op)
            inputs.append(parents)
        return inputs, action_types

    def action(self, inputs: List[Node], action_type: str):
        try:
            if action_type == "append":
                return self._action_append(inputs[0])
            if action_type == "update":
                return self._action_update(inputs[0])
            if action_type == "mix":
                return self._action_mix(inputs[0], inputs[1])
            raise ValueError(f"Unknown GMPO action: {action_type}")
        except Exception as e:
            import traceback
            self.logger.error(f"[GMPO] action {action_type} failed: {e}")
            self.logger.error(f"[GMPO] traceback:\n{traceback.format_exc()}")
            return None

    # ----- two-stage helpers -----

    def _run_failure_analysis(self, parent: Node, label: str) -> Optional[str]:
        examples = parent.get_wrong_examples(self.model_responses_num)
        if not examples:
            return None
        examples_content = _build_examples_content(
            examples, self.task, self.optim_sees_images, OPTIMIZER_MAX_EXAMPLE_IMAGES
        )
        prompt = _failure_analysis_prompt(parent.base_instruction, self.class_choices, examples_content)
        response = self.optim_model.model.generate(prompt)
        self._log_optim_io(prompt, response, f"analysis ({label})")
        for _ in range(self.max_repair_attempts + 1):
            block = _extract_xml_block(response, "analysis")
            if block:
                return block.strip()
            response = self._repair_response(response, f"analysis ({label})")
        # Last resort: return the raw response so synthesis can still proceed
        self.logger.info(f"[GMPO] {label}: analysis tag missing, falling back to raw response.")
        return response.strip() if response else None

    # ----- operators -----

    def _action_append(self, parent: Node):
        analysis = self._run_failure_analysis(parent, f"append parent={parent.id}")
        if not analysis:
            self.logger.info(f"[GMPO] node {parent.id}: no analysis available for append; skipping.")
            return None

        prompt = _append_synthesis_prompt(
            base_instruction=parent.base_instruction,
            guidelines=parent.guidelines,
            class_choices=self.class_choices,
            rule_mode=self.rule_mode,
            analysis=analysis,
        )

        def extractor(response):
            return _extract_json(_extract_xml_block(response, "new_rule"))

        rule = self._extract_with_repair(
            prompt=prompt,
            extractor=extractor,
            validator=lambda v: _validate_rule(v, self.class_choices),
            label=f"append synth (parent={parent.id})",
        )
        if rule is None:
            return None
        new_guidelines = list(parent.guidelines) + [rule]
        if len(new_guidelines) > self.max_rules:
            new_guidelines = new_guidelines[-self.max_rules:]
        return self._make_child_node(parent, [parent], new_guidelines, "append")

    def _action_update(self, parent: Node):
        if not parent.guidelines:
            return self._action_append(parent)
        analysis = self._run_failure_analysis(parent, f"update parent={parent.id}")
        if not analysis:
            self.logger.info(f"[GMPO] node {parent.id}: no analysis available for update; skipping.")
            return None

        prompt = _update_synthesis_prompt(
            base_instruction=parent.base_instruction,
            guidelines=parent.guidelines,
            class_choices=self.class_choices,
            rule_mode=self.rule_mode,
            analysis=analysis,
        )

        def extractor(response):
            idx_str = _extract_xml_block(response, "target_rule_index")
            rule_str = _extract_xml_block(response, "updated_rule")
            if idx_str is None or rule_str is None:
                return None
            m = re.search(r"-?\d+", idx_str)
            if not m:
                return None
            try:
                idx = int(m.group(0))
            except Exception:
                return None
            rule = _extract_json(rule_str)
            if rule is None:
                return None
            if idx < 1 or idx > len(parent.guidelines):
                return None
            return idx, rule

        def validator(value):
            if not isinstance(value, tuple):
                return False, "extractor did not return (idx, rule)"
            idx, rule = value
            ok, reason = _validate_rule(rule, self.class_choices)
            if not ok:
                return False, reason
            # Coverage preservation check: replacing rule must not reduce coverage.
            if self.class_choices:
                old_rule = parent.guidelines[idx - 1]
                old_pairs = _rule_covers_pairs(old_rule)
                new_pairs = _rule_covers_pairs(rule)
                lost = old_pairs - new_pairs
                if lost:
                    lost_str = "; ".join(f"({a}, {b})" for a, b in sorted(lost))
                    return False, f"update would drop coverage of pair(s) {lost_str}"
            return True, ""

        result = self._extract_with_repair(
            prompt=prompt,
            extractor=extractor,
            validator=validator,
            label=f"update synth (parent={parent.id})",
        )
        if result is None:
            return None
        idx, rule = result
        new_guidelines = list(parent.guidelines)
        new_guidelines[idx - 1] = rule
        return self._make_child_node(parent, [parent], new_guidelines, "update")

    def _action_mix(self, parent_a: Node, parent_b: Node):
        analysis_a = self._run_failure_analysis(parent_a, f"mix parent_a={parent_a.id}")
        analysis_b = self._run_failure_analysis(parent_b, f"mix parent_b={parent_b.id}")
        if not analysis_a:
            analysis_a = "(no failure analysis available for parent A)"
        if not analysis_b:
            analysis_b = "(no failure analysis available for parent B)"

        prompt = _mix_synthesis_prompt(
            parent_a, parent_b,
            class_choices=self.class_choices,
            rule_mode=self.rule_mode,
            analysis_a=analysis_a,
            analysis_b=analysis_b,
            max_rules=self.max_rules,
        )

        def extractor(response):
            return _extract_json(_extract_xml_block(response, "merged_rules"))

        parent_coverage = (
            _ruleset_coverage(parent_a.guidelines) | _ruleset_coverage(parent_b.guidelines)
            if self.class_choices else set()
        )

        def validator(value):
            if not isinstance(value, list) or not value:
                return False, "merged_rules must be a non-empty list"
            kept = []
            for r in value:
                ok, _ = _validate_rule(r, self.class_choices)
                if ok:
                    kept.append(r)
            if not kept:
                return False, "no individual rule passed validation"
            # Coverage preservation across the merge.
            if self.class_choices:
                merged_coverage = _ruleset_coverage(kept)
                lost = parent_coverage - merged_coverage
                if lost:
                    lost_str = "; ".join(f"({a}, {b})" for a, b in sorted(lost))
                    return False, f"merge would drop coverage of pair(s) {lost_str}"
            return True, ""

        merged = self._extract_with_repair(
            prompt=prompt,
            extractor=extractor,
            validator=validator,
            label=f"mix synth (a={parent_a.id}, b={parent_b.id})",
        )
        if not merged:
            return None
        merged = [r for r in merged if _validate_rule(r, self.class_choices)[0]]
        merged = merged[: self.max_rules]
        if not merged:
            return None
        return self._make_child_node(parent_a, [parent_a, parent_b], merged, "mix")

    # ----- core helpers -----

    def _make_child_node(self, primary_parent: Node, parents: List[Node],
                         guidelines: list, action_type: str) -> Node:
        rendered = render_with_guidelines(primary_parent.base_instruction, guidelines)
        return Node(
            instruction=rendered,
            task=self.task,
            parents=parents,
            base_instruction=primary_parent.base_instruction,
            guidelines=guidelines,
            action_type=action_type,
        )

    def _extract_with_repair(self, prompt, extractor, label, validator=None):
        """
        Run optim, extract, validate. On failure, request repair up to
        max_repair_attempts times. `validator(value)` should return either
        a (bool, str) tuple, or a bool, or — for backward compat — None for
        valid and a string for invalid.
        """
        response = self.optim_model.model.generate(prompt)
        self._log_optim_io(prompt, response, label)

        def _check(value):
            if value is None:
                return False, "extractor returned None"
            if validator is None:
                return True, ""
            res = validator(value)
            if isinstance(res, tuple):
                return bool(res[0]), str(res[1])
            return bool(res), ""

        for attempt in range(self.max_repair_attempts + 1):
            value = extractor(response)
            ok, reason = _check(value)
            if ok:
                return value
            self.logger.info(f"[GMPO] {label}: validation failed ({reason}); attempt {attempt + 1}.")
            if attempt < self.max_repair_attempts:
                response = self._repair_response(response, label, reason=reason)
        self.logger.info(f"[GMPO] {label}: extraction failed after retries.")
        return None

    def _repair_response(self, response: str, label: str, reason: str = "") -> str:
        extra = f" Reason: {reason}." if reason else ""
        repair_prompt = [{
            "role": "user",
            "content": (
                "Your previous response did not match the required output format or violated a "
                "hard requirement.{} Re-emit your answer using the EXACT XML tag(s) and JSON schema "
                "requested previously. Do not include code fences, prose, or extra tags. Output "
                "only the required XML block(s) and nothing else. If the issue was vague language, "
                "rewrite naming concrete observable features instead.\n\n"
                "Previous response:\n{}"
            ).format(extra, response),
        }]
        return self.optim_model.model.generate(repair_prompt)

    def _refine_initial_instruction(self, task: BaseTask) -> str:
        sample = task.train_data[: min(3, len(task.train_data))]
        sample_lines = []
        for ex in sample:
            q = task.get_query(ex)
            a = task.get_answer(ex)
            if isinstance(a, list):
                a = ", ".join(map(str, a))
            sample_lines.append(f"Input: {q}\nAnswer: {a}")
        sample_block = "\n\n".join(sample_lines) if sample_lines else "(no examples available)"

        prompt = _refine_initial_prompt(task.initial_prompt, sample_block)
        response = self.optim_model.model.generate(prompt)
        self._log_optim_io(prompt, response, "refine_initial")

        for _ in range(self.max_repair_attempts + 1):
            block = _extract_xml_block(response, "refined_instruction")
            if block:
                if GUIDELINES_PLACEHOLDER not in block:
                    block = f"{block.rstrip()}\n\n{GUIDELINES_PLACEHOLDER}"
                return block.strip()
            response = self._repair_response(response, "refine_initial")

        self.logger.info("[GMPO] refine_initial: extraction failed; falling back to original prompt.")
        return f"{task.initial_prompt.strip()}\n\n{GUIDELINES_PLACEHOLDER}"

    def _log_optim_io(self, prompt, response, label):
        self.logger.info("=" * 80)
        self.logger.info(f"[GMPO/{label}] PROMPT:")
        for msg in prompt:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        self.logger.info(item["text"])
                    elif item_type == "image":
                        self.logger.info(f"[image: {item.get('image')}]")
                    else:
                        self.logger.info(f"[{item_type}]")
            else:
                self.logger.info(str(content))
        self.logger.info("-" * 80)
        self.logger.info(f"[GMPO/{label}] RESPONSE:\n{response}\n")