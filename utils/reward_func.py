from typing import Optional
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    r"""
    Reward function that checks if the completion is the same as the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
    ```python
    >>> from trl.rewards import accuracy_reward

    >>> solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completion = [
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{2}}"}],
    ... ]
    >>> accuracy_reward(completion, solution)
    [1.0, 0.0]
    ```
    """

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception:
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = float(content.strip().lower() == sol.strip().lower())
        rewards.append(reward)

    return rewards


def trajectory_efficiency_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    trajectory_lengths: list[int] | None = None,
    max_trajectory_length: int = 256,
    efficiency_weight: float = 0.3,
    **kwargs,
) -> list[float]:
    """
    Rewards using a shorter latent trajectory *when the answer is correct*.

    Returns efficiency_weight * (1 - k/max_trajectory_length) for correct
    completions and 0.0 otherwise.  Combined with accuracy_reward this
    incentivises the DifficultyEstimator to choose smaller k for easy problems.

    Only meaningful when trajectory_lengths is provided (i.e. during adaptive
    GRPO training).  Falls back to all-zero if trajectory_lengths is None.
    """
    if trajectory_lengths is None:
        return [0.0] * len(completions)

    acc = accuracy_reward(completions=completions, solution=solution, **kwargs)
    rewards = []
    for a, k in zip(acc, trajectory_lengths):
        if a is None or a == 0.0:
            rewards.append(0.0)
        else:
            rewards.append(float(efficiency_weight * (1.0 - k / max_trajectory_length)))
    return rewards


def length_penalty_reward(
    completions: list[list[dict[str, str]]],
    max_completion_length: int,
    completion_token_lengths: list[int] | None = None,
    **kwargs,
) -> list[float]:
    """Penalizes completions that are close to or at the maximum length.

    Returns 0.0 for short completions and a negative penalty that increases
    as the completion approaches max_completion_length.  The penalty ramps
    linearly once the completion exceeds 80% of max_completion_length.

    If ``completion_token_lengths`` is provided (list of per-sample token
    counts), it is used directly.  Otherwise the function falls back to
    estimating token count from character length (chars / 3.5).
    """
    threshold = int(max_completion_length * 0.8)
    rewards = []
    for i, completion in enumerate(completions):
        if completion_token_lengths is not None:
            length = completion_token_lengths[i]
        else:
            # Rough token estimate when actual counts are unavailable
            content = completion[0]["content"]
            length = int(len(content) / 3.5)
        if length <= threshold:
            rewards.append(0.0)
        else:
            # Linear penalty from 0 to -1 as length goes from threshold to max
            overshoot = (length - threshold) / max(max_completion_length - threshold, 1)
            rewards.append(-min(overshoot, 1.0))
    return rewards
