"""
Adaptive Latent Trajectory (ALT)
=================================
Difficulty-aware dynamic reasoning path length for Latent Reasoning Models.

Two-phase training strategy:
  Phase 1 – Matryoshka SFT:  AdaptiveLatentSFTModel randomly truncates the
    latent trajectory to k ∈ length_candidates each forward pass, forcing the
    trajectory to be "prefix-consistent": shorter prefixes are already
    sufficient for easier problems.

  Phase 2 – GRPO with difficulty routing:  AdaptiveLatentGRPOModel attaches a
    DifficultyEstimator that predicts k from the prompt hidden state.  During
    rollout generation k is sampled stochastically (temperature > 0) so
    different rollouts of the same prompt explore different budget levels.
    AdaptiveGRPOTrainer adds a REINFORCE term that propagates the GRPO
    advantage back to the DifficultyEstimator, training it to pick shorter k
    for easy problems while preserving accuracy.

At inference (temperature=0) the DifficultyEstimator uses greedy argmax,
reducing the trajectory from 256 to as few as 64 tokens for simple problems
with no accuracy penalty.
"""

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reason import TransformerReasoningNet, LatentTransformerReasoningModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_length_candidates(max_length: int) -> list[int]:
    """[max//4, max//2, 3*max//4, max], e.g. [64, 128, 192, 256] for 256."""
    step = max(max_length // 4, 1)
    return [step * i for i in range(1, 5) if step * i <= max_length]


# ---------------------------------------------------------------------------
# DifficultyEstimator
# ---------------------------------------------------------------------------

class DifficultyEstimator(nn.Module):
    """
    Lightweight two-layer MLP that maps the last prompt hidden state to a
    discrete probability distribution over candidate trajectory lengths.

    Trained end-to-end via REINFORCE using GRPO advantages.
    """

    def __init__(self, hidden_size: int, length_candidates: list[int]):
        super().__init__()
        self.length_candidates = sorted(length_candidates)
        n = len(self.length_candidates)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, n),
        )

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """last_hidden: [B, H] → logits [B, n_candidates]"""
        return self.mlp(last_hidden.to(self.mlp[0].weight.dtype))

    def select_lengths(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory lengths from prompt hidden states.

        Args:
            hidden_states: [B, seq_len, H]
            temperature: > 0 → stochastic (exploration); 0 → greedy (inference)

        Returns:
            lengths:   [B] int64 – selected trajectory lengths per sample
            log_probs: [B] float  – log P(length | prompt), no gradient
        """
        last_hidden = hidden_states[:, -1, :]
        logits = self.forward(last_hidden)

        with torch.no_grad():
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                indices = logits.argmax(dim=-1)

        cands = torch.tensor(
            self.length_candidates, dtype=torch.long, device=hidden_states.device
        )
        lengths = cands[indices]
        log_probs = F.log_softmax(logits, dim=-1).detach().gather(
            1, indices.unsqueeze(1)
        ).squeeze(1)
        return lengths, log_probs

    def log_probs_for_lengths(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recompute log P(lengths | prompts) *with gradient*.

        Used in the REINFORCE update during compute_loss.

        Args:
            hidden_states: [B, seq_len, H]
            lengths:       [B] int64 – lengths chosen during rollout

        Returns:
            log_probs: [B] float – with gradient flowing to estimator params
        """
        last_hidden = hidden_states[:, -1, :]
        logits = self.forward(last_hidden)
        k_indices = torch.tensor(
            [self.length_candidates.index(k.item()) for k in lengths],
            dtype=torch.long,
            device=logits.device,
        )
        return F.log_softmax(logits, dim=-1).gather(1, k_indices.unsqueeze(1)).squeeze(1)


# ---------------------------------------------------------------------------
# AdaptiveTransformerReasoningNet
# ---------------------------------------------------------------------------

class AdaptiveTransformerReasoningNet(TransformerReasoningNet):
    """
    TransformerReasoningNet + DifficultyEstimator with *true* compute savings.

    forward() accepts a trajectory_lengths tensor and:
      1. Truncates the learnable self.latent_trajectory parameter to
         max(trajectory_lengths) tokens.
      2. Runs the reasoning transformer on (prompt + max_k) instead of
         (prompt + 256) tokens, so the small reasoning model also benefits
         from the budget reduction — not just the large slow model.
      3. Applies a per-sample attention mask so samples with k_b < max_k do
         not attend to padding positions.
    """

    def __init__(
        self,
        model_name_or_path: str,
        latent_trajectory_length: int = 256,
        hidden_size: int = 1024,
        length_candidates: Optional[list[int]] = None,
    ):
        super().__init__(model_name_or_path, latent_trajectory_length, hidden_size)

        if length_candidates is None:
            length_candidates = _default_length_candidates(latent_trajectory_length)
        self.length_candidates = sorted(length_candidates)

        self.difficulty_estimator = DifficultyEstimator(hidden_size, length_candidates)

        # Initialise with uniform prior so training starts unbiased.
        nn.init.zeros_(self.difficulty_estimator.mlp[-1].weight)
        nn.init.zeros_(self.difficulty_estimator.mlp[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        trajectory_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:       [B, prompt_len, H]
            attention_mask:      [B, prompt_len], optional
            trajectory_lengths:  [B] int64; if None falls back to full length.

        Returns:
            latent_trajectory:   [B, max_k, H] — max_k is the largest length
                in trajectory_lengths (or self.latent_trajectory_length if None).
                Positions beyond each sample's own k_b are still produced but
                are masked so they cannot influence the slow model downstream.
        """
        hidden_state_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.reasoning_network.dtype)
        B = hidden_states.size(0)

        if trajectory_lengths is not None:
            max_k = int(trajectory_lengths.max().item())
        else:
            max_k = self.latent_trajectory_length

        # ---- Truncate the learnable latent_trajectory parameter ------------
        # This is where the reasoning-network compute saving actually happens:
        # the transformer sees prompt_len + max_k tokens instead of + 256.
        latent_param = self.latent_trajectory[:max_k, :]  # [max_k, H]
        latent_trajectory_base = latent_param.unsqueeze(0).expand(B, -1, -1)
        latent_trajectory_base = latent_trajectory_base.to(self.reasoning_network.dtype)

        last_hidden_state = hidden_states[:, -1:, :]
        latent_trajectory_input = last_hidden_state * latent_trajectory_base  # [B, max_k, H]

        transformed_hidden_states = self.transform_layer(hidden_states)
        transformed_latent_trajectory = self.transform_layer(latent_trajectory_input)
        reasoning_inputs = torch.cat(
            [transformed_hidden_states, transformed_latent_trajectory], dim=1
        )

        reasoning_mask = None
        if attention_mask is not None:
            if trajectory_lengths is not None:
                # Per-sample latent mask: 1 for j < k_b, 0 otherwise.
                traj_range = torch.arange(
                    max_k, device=attention_mask.device
                ).unsqueeze(0)
                latent_mask = (traj_range < trajectory_lengths.unsqueeze(1)).to(
                    attention_mask.dtype
                )
            else:
                latent_mask = torch.ones(
                    B, max_k,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
            reasoning_mask = torch.cat([attention_mask, latent_mask], dim=1)

        outputs = self.reasoning_network(
            inputs_embeds=reasoning_inputs,
            attention_mask=reasoning_mask,
            return_dict=True,
        )
        latent_trajectory_output = outputs.last_hidden_state[:, -max_k:, :]
        latent_trajectory_out = self.reverse_transform_layer(
            latent_trajectory_output
        ).to(hidden_state_dtype)
        return latent_trajectory_out


# ---------------------------------------------------------------------------
# AdaptiveLatentSFTModel  – Phase 1: Matryoshka prefix-consistency training
# ---------------------------------------------------------------------------

class AdaptiveLatentSFTModel(LatentTransformerReasoningModel):
    """
    SFT model with Matryoshka prefix-consistency training.

    Each forward pass randomly samples k ∈ reasoning_network.length_candidates
    and uses only the first k tokens of the latent trajectory.  Over many
    steps this forces every prefix to be independently sufficient, so at GRPO
    time the DifficultyEstimator can safely choose shorter budgets for easy
    problems.
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1) + labels.size(1)),
                device=input_ids.device,
                dtype=torch.long,
            )

        with torch.no_grad():
            prompt_mask = attention_mask[:, : input_ids.size(1)]
            prompt_embeddings, prompt_hidden_states = self._prefill_prompt(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                position_ids=position_ids,
                **kwargs,
            )

        # ---- Matryoshka prefix length sampled BEFORE reasoning network -----
        # Truncate before the reasoning forward so the small reasoning
        # transformer also pays only O(prompt + k), not O(prompt + 256).
        trajectory_lengths = None
        if self.training:
            k = random.choice(self.reasoning_network.length_candidates)
            trajectory_lengths = torch.full(
                (input_ids.size(0),), k, dtype=torch.long, device=input_ids.device,
            )
        # --------------------------------------------------------------------

        latent_trajectory = self.reasoning_network(
            prompt_hidden_states,
            attention_mask=prompt_mask,
            trajectory_lengths=trajectory_lengths,
        )

        latent_trajectory_mask = torch.ones(
            latent_trajectory.size(0), latent_trajectory.size(1),
            device=input_ids.device, dtype=torch.long,
        )

        label_embeddings = self.get_input_embeddings(labels).to(
            self.slow_reasoning_model.dtype
        )
        labels_mask = attention_mask[:, input_ids.size(1):]

        input_embeddings = torch.cat(
            [prompt_embeddings, latent_trajectory, label_embeddings], dim=1
        )
        input_mask = torch.cat(
            [prompt_mask, latent_trajectory_mask, labels_mask], dim=1
        ).long()

        labels = labels.long()
        labels = labels.masked_fill(labels_mask.to(labels.device) == 0, -100)
        labels = torch.cat(
            (
                prompt_embeddings.new_ones(
                    labels.size(0), prompt_embeddings.size(1)
                ).long() * -100,
                latent_trajectory.new_ones(
                    labels.size(0), latent_trajectory.size(1)
                ).long() * -100,
                labels,
            ),
            dim=1,
        ).long()

        outputs = self.slow_reasoning_model(
            inputs_embeds=input_embeddings,
            attention_mask=input_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )
        return outputs


# ---------------------------------------------------------------------------
# AdaptiveLatentGRPOModel  – Phase 2: difficulty-aware GRPO generation
# ---------------------------------------------------------------------------

class AdaptiveLatentGRPOModel(nn.Module):
    """
    GRPO policy wrapper that adds difficulty-aware trajectory routing on top of
    LatentTransformerReasoningForGRPO (which is defined in rft.py).

    Key changes vs. the base GRPO model:

    generate()
      * Calls DifficultyEstimator.select_lengths() to pick k per sample.
      * Truncates the latent trajectory to k before the KV-cache forward pass.
      * Stores chosen lengths in self._last_trajectory_lengths so the trainer
        can pass them back to forward() and compute_loss().

    forward() / _build_adaptive_grpo_inputs()
      * Accepts trajectory_lengths kwarg; truncates trajectory and builds a
        per-sample attention mask so variable-length batches are handled.
    """

    def __init__(self, base_grpo_model, diff_sample_temperature: float = 1.0):
        """
        Args:
            base_grpo_model: A LatentTransformerReasoningForGRPO instance whose
                reasoning_network is an AdaptiveTransformerReasoningNet.
            diff_sample_temperature: Temperature for DifficultyEstimator during
                rollout generation.  Set to 0 for deterministic inference.
        """
        super().__init__()
        self._model = base_grpo_model
        self.diff_sample_temperature = diff_sample_temperature

        # Expose attributes expected by the TRL / HF ecosystem.
        self.config = base_grpo_model.config
        self.name_or_path = base_grpo_model.name_or_path
        self._model_tags = base_grpo_model._model_tags
        self.warnings_issued = base_grpo_model.warnings_issued
        self.prepare_inputs_for_generation = (
            base_grpo_model.prepare_inputs_for_generation
        )

        # Trajectory lengths chosen during the most recent generate() call.
        # Shape: [B]. Cleared at the start of each generate() call.
        self._last_trajectory_lengths: Optional[torch.Tensor] = None

    # ---- Delegate attribute access to the wrapped model -------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)

    @property
    def device(self):
        return next(self._model.parameters()).device

    def parameters(self, recurse=True):
        return self._model.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self._model.named_parameters(prefix, recurse, remove_duplicate)

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._model.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        self._model.train(mode)
        return super().train(mode)

    def eval(self):
        self._model.eval()
        return super().eval()

    def gradient_checkpointing_enable(self, **kwargs):
        self._model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self._model.gradient_checkpointing_disable()

    def add_model_tags(self, tags):
        self._model.add_model_tags(tags)

    def disable_adapter(self):
        return self._model.disable_adapter()

    # ---- Core logic -------------------------------------------------------

    def _get_difficulty_estimator(self):
        """Returns DifficultyEstimator; raises if reasoning_network is not adaptive."""
        rn = self._model.reasoning_network
        if not hasattr(rn, "difficulty_estimator"):
            raise AttributeError(
                "reasoning_network must be an AdaptiveTransformerReasoningNet "
                "to use AdaptiveLatentGRPOModel."
            )
        return rn.difficulty_estimator

    def _get_prompt_cache(
        self,
        prompt_ids,
        prompt_mask,
        position_ids=None,
        cached_prompt_kv=None,
        cached_prompt_hidden_states=None,
        cached_expand_idx=None,
        **kwargs,
    ):
        m = self._model
        if cached_prompt_kv is not None and cached_prompt_hidden_states is not None:
            B = prompt_ids.size(0)
            U = cached_prompt_hidden_states.size(0)
            if U < B and cached_expand_idx is not None:
                return (
                    m._expand_kv_cache(cached_prompt_kv, cached_expand_idx),
                    cached_prompt_hidden_states[cached_expand_idx],
                )
            return cached_prompt_kv, cached_prompt_hidden_states

        with torch.no_grad():
            prompt_emb = m.get_input_embeddings(prompt_ids).to(
                m.slow_reasoning_model.dtype
            )
            prompt_out = m.slow_reasoning_model(
                inputs_embeds=prompt_emb,
                attention_mask=prompt_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
                use_cache=True,
                **kwargs,
            )
            return prompt_out.past_key_values, prompt_out.hidden_states[-1].detach()

    def _forward_grouped_by_trajectory(
        self,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask=None,
        position_ids=None,
        cached_prompt_kv=None,
        cached_prompt_hidden_states=None,
        cached_expand_idx=None,
        trajectory_lengths=None,
        **kwargs,
    ):
        m = self._model
        if completion_mask is None:
            completion_mask = (completion_ids != m.pad_token_id).long()

        prompt_mask = prompt_mask.long()
        completion_mask = completion_mask.long()
        trajectory_lengths = trajectory_lengths.to(prompt_ids.device).long()

        prompt_kv, prompt_hidden_states = self._get_prompt_cache(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            position_ids=position_ids,
            cached_prompt_kv=cached_prompt_kv,
            cached_prompt_hidden_states=cached_prompt_hidden_states,
            cached_expand_idx=cached_expand_idx,
            **kwargs,
        )

        B = prompt_ids.size(0)
        C = completion_ids.size(1)
        logits_out = None

        for k_tensor in torch.unique(trajectory_lengths, sorted=True):
            k = int(k_tensor.item())
            group_idx = (trajectory_lengths == k).nonzero(as_tuple=False).flatten()
            group_lengths = trajectory_lengths[group_idx]
            group_prompt_kv = m._expand_kv_cache(prompt_kv, group_idx)
            group_prompt_hidden = prompt_hidden_states[group_idx]
            group_prompt_mask = prompt_mask[group_idx]
            group_completion_ids = completion_ids[group_idx]
            group_completion_mask = completion_mask[group_idx]

            latent_trajectory = m.reasoning_network(
                group_prompt_hidden,
                attention_mask=group_prompt_mask,
                trajectory_lengths=group_lengths,
            ).to(m.slow_reasoning_model.dtype)

            latent_mask = torch.ones(
                latent_trajectory.size(0),
                latent_trajectory.size(1),
                device=group_prompt_mask.device,
                dtype=group_prompt_mask.dtype,
            )
            completion_embeddings = m.get_input_embeddings(group_completion_ids).to(
                m.slow_reasoning_model.dtype
            )

            input_embeddings = torch.cat(
                [latent_trajectory, completion_embeddings], dim=1
            )
            input_mask = torch.cat(
                [group_prompt_mask, latent_mask, group_completion_mask], dim=1
            ).long()

            outputs = m.slow_reasoning_model(
                inputs_embeds=input_embeddings,
                attention_mask=input_mask,
                past_key_values=group_prompt_kv,
                return_dict=True,
                **kwargs,
            )
            group_logits = outputs.logits[:, k - 1 : k - 1 + C, :]

            if logits_out is None:
                logits_out = group_logits.new_empty(B, C, group_logits.size(-1))
            logits_out[group_idx] = group_logits

        return logits_out

    def _build_adaptive_grpo_inputs(
        self,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask=None,
        position_ids=None,
        cached_prompt_kv=None,
        cached_prompt_hidden_states=None,
        cached_expand_idx=None,
        trajectory_lengths=None,
        **kwargs,
    ):
        """Like LatentTransformerReasoningForGRPO._build_grpo_inputs but with
        per-sample trajectory truncation controlled by trajectory_lengths."""
        m = self._model  # base GRPO model

        if completion_mask is None:
            completion_mask = (completion_ids != m.pad_token_id).long()

        prompt_mask = prompt_mask.long()
        completion_mask = completion_mask.long()

        if cached_prompt_kv is not None and cached_prompt_hidden_states is not None:
            B = prompt_ids.size(0)
            U = cached_prompt_hidden_states.size(0)
            if U < B and cached_expand_idx is not None:
                prompt_kv = m._expand_kv_cache(cached_prompt_kv, cached_expand_idx)
                prompt_hidden_states = cached_prompt_hidden_states[cached_expand_idx]
            else:
                prompt_kv = cached_prompt_kv
                prompt_hidden_states = cached_prompt_hidden_states
        else:
            with torch.no_grad():
                prompt_emb = m.get_input_embeddings(prompt_ids).to(
                    m.slow_reasoning_model.dtype
                )
                prompt_out = m.slow_reasoning_model(
                    inputs_embeds=prompt_emb,
                    attention_mask=prompt_mask,
                    position_ids=position_ids,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=True,
                    **kwargs,
                )
                prompt_hidden_states = prompt_out.hidden_states[-1].detach()
                prompt_kv = prompt_out.past_key_values

        # Compute trajectory at the truncated length (saves compute on the
        # reasoning network too — it sees prompt + max_k, not prompt + 256).
        latent_trajectory = m.reasoning_network(
            prompt_hidden_states,
            attention_mask=prompt_mask,
            trajectory_lengths=trajectory_lengths,
        ).to(m.slow_reasoning_model.dtype)

        B, L, H = latent_trajectory.shape

        if trajectory_lengths is not None:
            max_k = int(trajectory_lengths.max().item())
            # latent_trajectory is already at length max_k (no further slicing).
            traj_range = torch.arange(max_k, device=prompt_mask.device).unsqueeze(0)
            latent_trajectory_mask = (traj_range < trajectory_lengths.unsqueeze(1)).long()
        else:
            max_k = L
            latent_trajectory_mask = torch.ones(B, L, device=prompt_mask.device, dtype=torch.long)

        completion_embeddings = m.get_input_embeddings(completion_ids).to(
            m.slow_reasoning_model.dtype
        )
        return (
            prompt_mask,
            prompt_kv,
            latent_trajectory,
            latent_trajectory_mask,
            completion_embeddings,
            completion_mask,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        completion_ids: Optional[torch.LongTensor] = None,
        completion_mask: Optional[torch.Tensor] = None,
        cached_prompt_kv=None,
        cached_prompt_hidden_states=None,
        cached_expand_idx=None,
        trajectory_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with optional trajectory_lengths for consistent log-prob
        computation with what was used during generation."""
        if prompt_ids is None:
            # SFT-style forward (shouldn't be called in GRPO, but handle gracefully).
            return self._model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                **kwargs,
            )

        kwargs.pop("return_dict", None)
        kwargs.pop("logits_to_keep", None)

        m = self._model

        if trajectory_lengths is not None:
            return self._forward_grouped_by_trajectory(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                position_ids=position_ids,
                cached_prompt_kv=cached_prompt_kv,
                cached_prompt_hidden_states=cached_prompt_hidden_states,
                cached_expand_idx=cached_expand_idx,
                trajectory_lengths=trajectory_lengths,
                **kwargs,
            )

        (
            prompt_mask,
            prompt_kv,
            latent_trajectory,
            latent_trajectory_mask,
            completion_embeddings,
            completion_mask,
        ) = self._build_adaptive_grpo_inputs(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            position_ids=position_ids,
            cached_prompt_kv=cached_prompt_kv,
            cached_prompt_hidden_states=cached_prompt_hidden_states,
            cached_expand_idx=cached_expand_idx,
            trajectory_lengths=trajectory_lengths,
            **kwargs,
        )

        input_embeddings = torch.cat([latent_trajectory, completion_embeddings], dim=1)
        input_mask = torch.cat(
            [prompt_mask, latent_trajectory_mask, completion_mask], dim=1
        ).long()

        outputs = m.slow_reasoning_model(
            inputs_embeds=input_embeddings,
            attention_mask=input_mask,
            past_key_values=prompt_kv,
            return_dict=True,
            **kwargs,
        )

        logits_start = latent_trajectory.size(1) - 1
        logits_end = logits_start + completion_ids.size(1)
        return outputs.logits[:, logits_start:logits_end, :]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        generation_config=None,
        return_logps: bool = False,
        return_cached_states: bool = False,
        **kwargs,
    ):
        """
        Generate completions using difficulty-adaptive trajectory length.

        The DifficultyEstimator samples k stochastically during training
        (self.diff_sample_temperature > 0) so different rollouts of the same
        prompt can explore different budget levels.  At inference use
        self.diff_sample_temperature = 0 for deterministic greedy selection.

        Stores self._last_trajectory_lengths [B] for AdaptiveGRPOTrainer.
        """
        m = self._model

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if generation_config is None:
            generation_config = m.slow_reasoning_model.generation_config

        max_new_tokens = getattr(generation_config, "max_new_tokens", None)
        if max_new_tokens is None:
            raise ValueError("generation_config.max_new_tokens must be set.")

        eos_token_id = getattr(generation_config, "eos_token_id", None)
        eos_token_ids = (
            [] if eos_token_id is None
            else ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
        )
        pad_token_id = getattr(generation_config, "pad_token_id", None) or m.pad_token_id

        B = input_ids.size(0)
        prompt_mask = attention_mask.long()

        # Deduplicate prompts for efficiency.
        unique_ids, unique_mask, expand_idx = m._dedup_prompts(input_ids, prompt_mask)
        U = unique_ids.size(0)

        # Prefill unique prompts.
        unique_emb = m.get_input_embeddings(unique_ids).to(m.slow_reasoning_model.dtype)
        prompt_out = m.slow_reasoning_model(
            inputs_embeds=unique_emb,
            attention_mask=unique_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )
        unique_hidden = prompt_out.hidden_states[-1].detach()
        unique_kv = prompt_out.past_key_values

        cached_unique_hidden = unique_hidden
        cached_unique_kv = unique_kv

        # Expand prompt states before sampling k so repeated GRPO rollouts of
        # the same prompt can explore different trajectory budgets.
        if U < B:
            full_hidden = unique_hidden[expand_idx]
            full_prompt_kv = m._expand_kv_cache(unique_kv, expand_idx)
        else:
            full_hidden = unique_hidden
            full_prompt_kv = unique_kv

        # ---- Difficulty-aware trajectory length selection ------------------
        diff_estimator = self._get_difficulty_estimator()
        temperature = self.diff_sample_temperature if self.training else 0.0
        trajectory_lengths, _ = diff_estimator.select_lengths(full_hidden, temperature)

        # Store for the trainer to retrieve.
        self._last_trajectory_lengths = trajectory_lengths

        all_tokens = torch.full(
            (B, max_new_tokens), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        all_logps = (
            torch.zeros((B, max_new_tokens), dtype=torch.float32, device=input_ids.device)
            if return_logps else None
        )
        eos_tensor = (
            torch.tensor(eos_token_ids, device=input_ids.device) if eos_token_ids else None
        )
        actual_length = 0

        # Decode each selected trajectory length separately. This avoids using
        # padded latent positions as the first completion-token context.
        for k_tensor in torch.unique(trajectory_lengths, sorted=True):
            k = int(k_tensor.item())
            group_idx = (trajectory_lengths == k).nonzero(as_tuple=False).flatten()
            group_lengths = trajectory_lengths[group_idx]
            group_hidden = full_hidden[group_idx]
            group_mask = prompt_mask[group_idx]
            group_prompt_kv = m._expand_kv_cache(full_prompt_kv, group_idx)

            latent = m.reasoning_network(
                group_hidden,
                attention_mask=group_mask,
                trajectory_lengths=group_lengths,
            ).to(unique_emb.dtype)
            latent_mask = torch.ones(
                latent.size(0),
                latent.size(1),
                dtype=group_mask.dtype,
                device=group_mask.device,
            )
            model_mask = torch.cat([group_mask, latent_mask], dim=1).long()
            latent_out = m.slow_reasoning_model(
                inputs_embeds=latent,
                attention_mask=model_mask,
                past_key_values=group_prompt_kv,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = latent_out.past_key_values
            next_token_logits = latent_out.logits[:, -1, :]

            group_size = group_idx.size(0)
            full_attention_mask = torch.ones(
                group_size,
                group_mask.size(1) + k + max_new_tokens,
                dtype=group_mask.dtype,
                device=group_mask.device,
            )
            full_attention_mask[:, : group_mask.size(1)] = group_mask
            current_mask_len = group_mask.size(1) + k
            finished = torch.zeros(group_size, dtype=torch.bool, device=input_ids.device)
            group_actual_length = 0

            for step in range(max_new_tokens):
                if return_logps:
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)

                next_tokens = m._sample_next_token(next_token_logits, generation_config)

                if return_logps:
                    token_logp = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
                    token_logp = torch.where(finished, torch.zeros_like(token_logp), token_logp)
                    all_logps[group_idx, step] = token_logp

                if eos_tensor is not None:
                    is_eos = (next_tokens.unsqueeze(1) == eos_tensor.unsqueeze(0)).any(dim=1)
                else:
                    is_eos = torch.zeros_like(finished)

                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, pad_token_id), next_tokens
                )
                all_tokens[group_idx, step] = next_tokens
                group_actual_length = step + 1
                finished = finished | is_eos
                if finished.all():
                    break

                current_mask_len += 1
                out = m.slow_reasoning_model(
                    input_ids=next_tokens.unsqueeze(1),
                    attention_mask=full_attention_mask[:, :current_mask_len],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                next_token_logits = out.logits[:, -1, :]

            actual_length = max(actual_length, group_actual_length)

        completion_ids = all_tokens[:, :actual_length]

        if actual_length == 0:
            empty = input_ids.new_full((B, 0), pad_token_id)
            result: tuple = (empty,)
            if return_logps:
                result += (empty.new_zeros((B, 0), dtype=torch.float32),)
            if return_cached_states:
                result += (cached_unique_kv, cached_unique_hidden, expand_idx)
            return result if len(result) > 1 else result[0]

        if return_logps and return_cached_states:
            return (
                completion_ids,
                all_logps[:, :actual_length],
                cached_unique_kv,
                cached_unique_hidden,
                expand_idx,
            )
        elif return_logps:
            return completion_ids, all_logps[:, :actual_length]
        elif return_cached_states:
            return completion_ids, cached_unique_kv, cached_unique_hidden, expand_idx
        return completion_ids
