"""
AdaptiveGRPOTrainer
===================
Extends GRPOTrainer to train the DifficultyEstimator inside
AdaptiveLatentGRPOModel via a REINFORCE auxiliary loss.

Two additions vs. the base trainer:

1. _get_per_token_logps():
   Passes the trajectory_lengths stored in the prepared batch to
   model.forward(), so log-probs are computed with the same trajectory slice
   that was used during generation.

2. compute_loss():
   After the standard GRPO loss, recomputes log P(k | prompt) *with gradient*
   using the cached prompt hidden states and the stored trajectory lengths,
   then adds:

       diff_loss = -mean( advantage * log P(k | prompt) )   [REINFORCE]

   Controlled by diff_reinforce_weight (default 0.05).  Set to 0.0 to
   disable difficulty estimator training and run standard GRPO.
"""

from typing import Any, Union, Optional

import torch

from .grpo_trainer import GRPOTrainer
from trl.trainer.utils import selective_log_softmax


class AdaptiveGRPOTrainer(GRPOTrainer):
    """GRPO trainer with REINFORCE update for the DifficultyEstimator."""

    def __init__(self, *args, diff_reinforce_weight: float = 0.05, **kwargs):
        """
        Args:
            diff_reinforce_weight: Weight λ for the REINFORCE loss term
                diff_loss = -λ * mean(advantage * log_P(k|prompt)).
                Set to 0 to disable.  Typical range: 0.02 – 0.1.
        """
        super().__init__(*args, **kwargs)
        self.diff_reinforce_weight = diff_reinforce_weight

    # ------------------------------------------------------------------
    # Override: pass trajectory_lengths to forward() for consistent logps
    # ------------------------------------------------------------------

    def _get_per_token_logps(
        self,
        model,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        logits_to_keep: int = None,
        cached_prompt_kv=None,
        cached_prompt_hidden_states=None,
        cached_expand_idx=None,
        trajectory_lengths: Optional[torch.Tensor] = None,
    ):
        if trajectory_lengths is None:
            # Backward-compatible fallback for older prepared batches.
            unwrapped = self.accelerator.unwrap_model(model)
            trajectory_lengths = getattr(unwrapped, "_last_trajectory_lengths", None)
        else:
            unwrapped = self.accelerator.unwrap_model(model)

        rn = getattr(unwrapped, "reasoning_network", None)
        if rn is None and hasattr(unwrapped, "_model"):
            rn = getattr(unwrapped._model, "reasoning_network", None)
        if rn is not None and not hasattr(rn, "difficulty_estimator"):
            trajectory_lengths = None

        logits = model(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            logits_to_keep=logits_to_keep,
            cached_prompt_kv=cached_prompt_kv,
            cached_prompt_hidden_states=cached_prompt_hidden_states,
            cached_expand_idx=cached_expand_idx,
            trajectory_lengths=trajectory_lengths,
        )
        if logits.size(1) != completion_ids.size(1):
            raise ValueError(
                f"Expected logits length {completion_ids.size(1)}, got {logits.size(1)}"
            )
        return selective_log_softmax(logits, completion_ids)

    # ------------------------------------------------------------------
    # Override: add REINFORCE term for DifficultyEstimator
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # Standard GRPO loss.
        grpo_loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        if self.diff_reinforce_weight <= 0.0:
            return grpo_loss

        unwrapped = self.accelerator.unwrap_model(model)

        trajectory_lengths: Optional[torch.Tensor] = inputs.get("trajectory_lengths")
        cached_hs: Optional[torch.Tensor] = inputs.get("cached_prompt_hidden_states")
        expand_idx: Optional[torch.Tensor] = inputs.get("cached_expand_idx")
        advantages: torch.Tensor = inputs["advantages"]

        if trajectory_lengths is None or cached_hs is None:
            return grpo_loss

        # Expand unique-prompt hidden states back to full batch dimension.
        B = trajectory_lengths.size(0)
        if expand_idx is not None and cached_hs.size(0) < B:
            expanded_hs = cached_hs[expand_idx]
        else:
            expanded_hs = cached_hs

        # Fetch the difficulty estimator (must be AdaptiveTransformerReasoningNet).
        rn = unwrapped.reasoning_network if hasattr(unwrapped, "reasoning_network") \
            else unwrapped._model.reasoning_network
        if not hasattr(rn, "difficulty_estimator"):
            return grpo_loss  # safety: non-adaptive model, skip

        diff_estimator = rn.difficulty_estimator

        # Recompute log P(k | prompt) *with gradient* for backprop.
        diff_log_probs = diff_estimator.log_probs_for_lengths(
            expanded_hs.detach(), trajectory_lengths
        )  # [B]

        # REINFORCE: -E[advantage * log P(k)]
        sample_has_signal = (advantages.abs() > 1e-8).float()
        num_valid = sample_has_signal.sum().clamp(min=1.0)
        reinforce_loss = (
            -(diff_log_probs * advantages.detach() * sample_has_signal).sum()
            / num_valid
        )

        # Log for monitoring.
        with torch.no_grad():
            self._metrics["diff_reinforce_loss"].append(reinforce_loss.item())
            avg_k = trajectory_lengths.float().mean().item()
            self._metrics["avg_trajectory_length"].append(avg_k)

        return grpo_loss + self.diff_reinforce_weight * reinforce_loss
