import math
import numpy as np
import random
from tqdm import tqdm
from abc import ABC, abstractmethod
from .search.node import Node
from .base_model import BaseModel
from .tasks import BaseTask
from scipy.stats import beta


def get_evaluator(evaluation_method: str):
    if evaluation_method == "bayes-ucb":
        return BayesianUCBEvaluator
    elif evaluation_method == "ucb":
        return UCBBanditEvaluator
    elif evaluation_method == "uniform":
        return UniformEvaluator
    else:
        raise ValueError(f"Unknown evaluation method: {evaluation_method}")


class Evaluator(ABC):
    """
    Abstract class for evaluators.
    """

    def __init__(self, base_model: BaseModel, task: BaseTask, logger=None, **kwargs):
        self.base_model = base_model
        self.task = task
        self.logger = logger
        self.evaluation_method = kwargs.get("evaluation_method", "uniform")
        # Track which training example indices have been sampled within a single evaluation run
        self._sampling_used_indices = set()

    @abstractmethod
    def __call__(self, nodes_list: list[Node]):
        """
        Evaluate the nodes in the list. Save the evaluation results in the nodes.
        Args:
            nodes_list: list of nodes to evaluate
        Returns:
            list of nodes with their evaluation scores(in node.train_metric)
        """
        pass

    def sample_examples(self, num_examples: int, return_indices: bool = False):
        """
        Sample examples from the task.train_data.
        """
        if num_examples <= 0:
            return ([], []) if return_indices else []

        total = len(self.task.train_data)
        remaining_indices = list(set(range(total)) - self._sampling_used_indices)

        if len(remaining_indices) < num_examples:
            self.logger.info(
                "[Evaluator] sample_examples: not enough remaining examples to sample from. Reset sampling state."
            )
            self._reset_sampling_state()
            remaining_indices = list(set(range(total)) - self._sampling_used_indices)

        k = min(len(remaining_indices), num_examples)

        # Always sample randomly
        chosen = random.sample(remaining_indices, k)
        self._sampling_used_indices.update(chosen)
        chosen_examples = [self.task.train_data[i] for i in chosen]

        return (chosen_examples, chosen) if return_indices else chosen_examples

    def _reset_sampling_state(self):
        self._sampling_used_indices = set()


class UCBBandits:
    """Upper Confidence Bound Bandits"""

    def __init__(self, num_prompts, num_samples=5, c=1.0, mode="ucb"):
        self.c = c
        assert mode in {"ucb"}
        self.mode = mode
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.reset()

    def update(self, chosen, scores):
        for i, score in zip(chosen, scores):
            self.counts[i] += self.num_samples
            self.scores[i] += score * self.num_samples

    def reset(self):
        self.counts = np.zeros(self.num_prompts)
        self.scores = np.zeros(self.num_prompts)

    def get_scores(self):
        # Some counts may be 0, so we need to avoid division by 0.
        return np.divide(self.scores, self.counts, out=np.zeros_like(self.scores), where=self.counts != 0)

    def choose(self, n, t):
        if np.sum(self.counts) == 0:
            # If all counts are 0, choose randomly.
            return random.sample(range(self.num_prompts), n)
        scores = self.get_scores()
        counts = self.counts + 1e-3
        if self.mode == "ucb":
            ucb_scores = scores + self.c * np.sqrt(np.log(t) / counts)
        elif self.mode == "ucb-e":
            ucb_scores = scores + self.c * np.sqrt(self.c / counts)

        # Choose the prompts with the highest UCB scores
        return np.argsort(ucb_scores)[::-1][:n]

    def get_infos(self):
        return self.counts


class UCBBanditEvaluator(Evaluator):
    """Upper Confidence Bound Evaluator"""

    def __init__(self, base_model: BaseModel, task: BaseTask, logger=None, **kwargs):
        super().__init__(base_model, task, logger, **kwargs)
        self.budget_per_prompt = kwargs.get("budget_per_prompt", 25)
        self.num_prompts_per_round = kwargs.get("num_prompts_per_round", 3)
        self.samples_per_eval = self.budget_per_prompt // 5
        self.ucb_c = kwargs.get("ucb_c", 2.0)

    def __call__(self, nodes_list: list[Node]):
        assert self.evaluation_method in {"ucb"}, f"unk evaluator: {self.evaluation_method}"
        self.eval_budget = self.budget_per_prompt * len(nodes_list)
        # reset per-run sampling and per-example stats
        self._reset_sampling_state()

        bandit_algo = UCBBandits(
            len(nodes_list), num_samples=self.samples_per_eval, mode=self.evaluation_method, c=self.ucb_c
        )
        eval_round = math.ceil(self.eval_budget / (self.samples_per_eval * self.num_prompts_per_round))

        num_prompts_per_round = min(self.num_prompts_per_round, len(nodes_list))
        for ri in tqdm(range(1, eval_round + 1), desc=f"Evaluating {len(nodes_list)} nodes"):
            # Sample the nodes
            sampled_nodes_idx = bandit_algo.choose(num_prompts_per_round, ri)
            sampled_nodes = [nodes_list[i] for i in sampled_nodes_idx]
            sampled_data, sampled_indices = self.sample_examples(self.samples_per_eval, return_indices=True)

            # Use base_model.forward_nodes to get scores
            result = self.base_model.forward_nodes(examples=sampled_data, nodes=sampled_nodes)
            scores = result["metrics"]

            bandit_algo.update(sampled_nodes_idx, scores)
        # Get final scores for all nodes
        final_scores = bandit_algo.get_scores().tolist()

        for node, score in zip(nodes_list, final_scores):
            node.train_metric = score

        # Sort nodes by scores (higher is better)
        node_score_pairs = [(node, score) for node, score in zip(nodes_list, final_scores)]
        node_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [node for node, _ in node_score_pairs]


class BayesianUCBBandits:
    """
    Bayesian UCB Bandits with Beta priors/posteriors for Bernoulli-like scores.
    Maintains a Beta(alpha_i, beta_i) posterior per node.
    """

    def __init__(
        self,
        num_prompts: int,
        num_samples: int = 5,
        c: float = 2.0,
        alpha0: np.ndarray | list[float] | tuple = None,
        beta0: np.ndarray | list[float] | tuple = None,
        total_budget: int = None,
    ):
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.c = float(c)
        # Initialize priors; default to uniform Beta(1,1) if not provided
        if alpha0 is None:
            alpha0 = np.ones(num_prompts, dtype=float)
        if beta0 is None:
            beta0 = np.ones(num_prompts, dtype=float)
        self.alpha0 = np.array(alpha0, dtype=float).copy()
        self.beta0 = np.array(beta0, dtype=float).copy()
        self.alpha = self.alpha0.copy()
        self.beta = self.beta0.copy()
        self.total_budget = total_budget

    def update(self, chosen: list[int], scores: list[float]):
        # Update Beta posterior with fractional counts derived from averaged scores
        for i, score in zip(chosen, scores):
            # Clamp score to [0,1] to avoid invalid Beta updates
            s = float(np.clip(score, 0.0, 1.0))
            successes = s * self.num_samples
            failures = (1.0 - s) * self.num_samples
            self.alpha[i] += successes
            self.beta[i] += failures

    def get_scores(self) -> np.ndarray:
        denom = self.alpha + self.beta
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.divide(self.alpha, denom, out=np.zeros_like(self.alpha), where=denom != 0)
        return means

    def choose(self, n: int, t: int) -> np.ndarray:
        """
        Select n nodes to evaluate in round t using Bayesian UCB using the quantile of the Beta posterior.
        """
        log_N = math.log(self.total_budget)
        denominator = t * (log_N**self.c)
        quantile_ratio = 1 - 1.0 / denominator
        quantile_ratio = np.clip(quantile_ratio, 0.5, 0.9999)
        quantile = beta.ppf(quantile_ratio, self.alpha, self.beta)
        return np.argsort(quantile)[::-1][:n]

    def get_infos(self) -> np.ndarray:
        # Return the total observation counts for each node
        total = self.alpha + self.beta
        prior = self.alpha0 + self.beta0
        return np.clip(total - prior, 0.0, None)


class BayesianUCBEvaluator(Evaluator):
    """
    Bayesian UCB Evaluator using Beta priors/posteriors.
    - Prior per node is derived from its parent's train_metric: p0 = node.parent.train_metric (clamped to [0,1]).
    - Prior strength is configurable via kwargs['bayes_prior_strength'] (default: 4.0).
    - Exploration coefficient via kwargs['ucb_c'] (default: 2.0).
    """

    def __init__(self, base_model: BaseModel, task: BaseTask, logger=None, **kwargs):
        super().__init__(base_model, task, logger, **kwargs)
        self.budget_per_prompt = kwargs.get("budget_per_prompt", 25)
        self.num_prompts_per_round = kwargs.get("num_prompts_per_round", 3)

        self.samples_per_eval = self.budget_per_prompt // 5
        self.ucb_c = kwargs.get("ucb_c", 2.0)
        self.prior_strength = kwargs.get("bayes_prior_strength", 4.0) / 100 * self.budget_per_prompt
        self.logger.info(f"[BayesianUCB] Prior strength: {self.prior_strength}, c={self.ucb_c}")

    def _build_beta_priors(self, nodes_list: list[Node]):
        # Build Beta prior parameters for each node based on its parent's train_metric
        alpha0 = np.zeros(len(nodes_list), dtype=float)
        beta0 = np.zeros(len(nodes_list), dtype=float)
        for i, node in enumerate(nodes_list):
            if node.parents is not None:
                train_metrics = np.array([parent.train_metric for parent in node.parents])
                p0 = float(np.mean(train_metrics))
            else:
                p0 = 0.5  # If the candidate hasno parent, use a neutral prior (0.5)
            p0 = np.clip(p0, 0.0, 1.0)
            strength = max(1e-6, float(self.prior_strength))
            # Informative prior: alpha = p0*S + 1, beta = (1-p0)*S + 1
            # Calculate raw alpha and beta values
            raw_alpha = p0 * strength
            raw_beta = (1.0 - p0) * strength

            # Round to one decimal place
            raw_alpha = round(raw_alpha, 2)
            raw_beta = round(raw_beta, 2)

            # Adjust so that their sum is an integer (strength)
            total = raw_alpha + raw_beta
            diff = round(strength - total, 2)
            # Add the difference to beta (arbitrary, could be alpha as well)
            raw_beta += diff

            # Add 1.0 to each as per prior
            alpha0[i] = raw_alpha + 1.0
            beta0[i] = raw_beta + 1.0
        return alpha0, beta0

    def __call__(self, nodes_list: list[Node]):
        assert self.evaluation_method in {"bayes-ucb"}, f"Unknown evaluator: {self.evaluation_method}"
        self.eval_budget = self.budget_per_prompt * len(nodes_list)
        # Reset per-run sampling and per-example stats
        self._reset_sampling_state()
        self.logger.info(
            f"[BayesianUCB] Starting evaluation with {len(nodes_list)} nodes, budget per prompt: {self.budget_per_prompt}"
        )

        alpha0, beta0 = self._build_beta_priors(nodes_list)
        self.logger.info(
            f"[BayesianUCB] Prior strength: {self.prior_strength}, c={self.ucb_c}, alpha0={alpha0}, beta0={beta0}"
        )
        bandit_algo = BayesianUCBBandits(
            num_prompts=len(nodes_list),
            num_samples=self.samples_per_eval,
            c=self.ucb_c,
            alpha0=alpha0,
            beta0=beta0,
            total_budget=self.eval_budget,
        )
        self.logger.info(f"[BayesianUCB] Initialized with prior_strength={self.prior_strength}, c={self.ucb_c}")

        num_prompts_per_round = min(self.num_prompts_per_round, len(nodes_list))
        eval_round = math.ceil(self.eval_budget / (self.samples_per_eval * num_prompts_per_round))
        for ri in tqdm(range(1, eval_round + 1), desc=f"Evaluating {len(nodes_list)} nodes (Bayes-UCB)"):
            # Sample nodes according to Bayesian UCB
            sampled_nodes_idx = bandit_algo.choose(num_prompts_per_round, ri)
            self.logger.info(f"[BayesianUCB] Sampled nodes: {sampled_nodes_idx}")
            sampled_nodes = [nodes_list[i] for i in sampled_nodes_idx]
            sampled_data, sampled_indices = self.sample_examples(self.samples_per_eval, return_indices=True)

            # Evaluate sampled nodes on sampled data
            result = self.base_model.forward_nodes(examples=sampled_data, nodes=sampled_nodes)
            scores = result["metrics"]

            # Update bandit with observed (averaged) scores
            bandit_algo.update(sampled_nodes_idx, scores)
            self.logger.info(
                f"[BayesianUCB] alpha, beta:\n alpha={bandit_algo.alpha}\n beta={bandit_algo.beta}\n total_eval_counts={bandit_algo.get_infos()}"
            )

        # Use posterior means as final scores
        final_scores = bandit_algo.get_scores().tolist()
        for node, score in zip(nodes_list, final_scores):
            node.train_metric = score

        # Sort nodes by scores (higher is better)
        node_score_pairs = [(node, score) for node, score in zip(nodes_list, final_scores)]
        node_score_pairs.sort(key=lambda x: x[1], reverse=True)
        self.logger.info(f"[BayesianUCB] Evaluation completed, returning {len(node_score_pairs)} ranked nodes")

        return [node for node, _ in node_score_pairs]


class UniformEvaluator(Evaluator):
    """
    Uniform Evaluator
    Evaluate all nodes with all train samples.
    """

    def __init__(self, base_model: BaseModel, task: BaseTask, logger=None, **kwargs):
        super().__init__(base_model, task, logger, **kwargs)
        self.budget_per_prompt = kwargs.get("budget_per_prompt", 25)

    def __call__(self, nodes_list: list[Node]):
        # reset per-run sampling and per-example stats
        self._reset_sampling_state()

        # Use base_model.forward_nodes to get scores and update train_metrics
        result = self.base_model.forward_nodes(
            examples=self.task.train_data[: self.budget_per_prompt], nodes=nodes_list
        )
        scores = result["metrics"]
        # Set train_metric directly as only one evaluation round
        for node, score in zip(nodes_list, scores):
            node.train_metric = score

        # Sort nodes by scores (higher is better)
        node_score_pairs = [(node, score) for node, score in zip(nodes_list, scores)]
        node_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [node for node, _ in node_score_pairs]
