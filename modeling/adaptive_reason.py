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
    TransformerReasoningNet + DifficultyEstimator.

    The DifficultyEstimator shares the input dtype / device with the
    reasoning network and is initialised to a uniform prior (equal logits).
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

        latent_trajectory = self.reasoning_network(
            prompt_hidden_states, attention_mask=prompt_mask
        )

        # ---- Matryoshka truncation ----------------------------------------
        if self.training:
            k = random.choice(self.reasoning_network.length_candidates)
            latent_trajectory = latent_trajectory[:, :k, :]
        # ----------------------------------------------------------------------

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

        labels = labels.masked_fill(labels == self.pad_token_id, -100).long()
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

        # Compute full trajectory then truncate per-sample.
        latent_trajectory = m.reasoning_network(
            prompt_hidden_states, attention_mask=prompt_mask
        ).to(m.slow_reasoning_model.dtype)

        B, L, H = latent_trajectory.shape

        if trajectory_lengths is not None:
            max_k = int(trajectory_lengths.max().item())
            latent_trajectory = latent_trajectory[:, :max_k, :]
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

        # ---- Difficulty-aware trajectory length selection ------------------
        diff_estimator = self._get_difficulty_estimator()
        temperature = self.diff_sample_temperature if self.training else 0.0
        unique_lengths, _ = diff_estimator.select_lengths(unique_hidden, temperature)

        # Expand from unique prompts to full batch.
        if U < B:
            trajectory_lengths = unique_lengths[expand_idx]
        else:
            trajectory_lengths = unique_lengths

        # Store for the trainer to retrieve.
        self._last_trajectory_lengths = trajectory_lengths

        # Compute full trajectory for unique prompts, then truncate.
        unique_latent_full = m.reasoning_network(
            unique_hidden, attention_mask=unique_mask
        ).to(unique_emb.dtype)

        unique_max_k = int(unique_lengths.max().item())
        unique_latent = unique_latent_full[:, :unique_max_k, :]

        # Build variable-length per-unique-sample mask.
        traj_range = torch.arange(unique_max_k, device=unique_mask.device).unsqueeze(0)
        unique_traj_mask_part = (traj_range < unique_lengths.unsqueeze(1))

        base_mask_len = unique_mask.size(1) + unique_max_k
        unique_model_mask = torch.ones(
            U, base_mask_len, dtype=unique_mask.dtype, device=unique_mask.device
        )
        unique_model_mask[:, : unique_mask.size(1)] = unique_mask
        unique_model_mask[:, unique_mask.size(1):] = unique_traj_mask_part.long()
        # --------------------------------------------------------------------

        # Forward latent trajectory through the slow model (fill KV cache).
        latent_out = m.slow_reasoning_model(
            inputs_embeds=unique_latent,
            attention_mask=unique_model_mask,
            past_key_values=unique_kv,
            use_cache=True,
            return_dict=True,
        )
        unique_full_kv = latent_out.past_key_values
        unique_next_logits = latent_out.logits[:, -1, :]

        cached_unique_hidden = unique_hidden
        cached_unique_kv = unique_kv

        # Expand KV cache to full batch.
        if U < B:
            past_key_values = m._expand_kv_cache(unique_full_kv, expand_idx)
            next_token_logits = unique_next_logits[expand_idx]
        else:
            past_key_values = unique_full_kv
            next_token_logits = unique_next_logits

        # Build batch-level attention mask (accounts for variable traj lengths).
        batch_max_k = int(trajectory_lengths.max().item())
        total_mask_len = unique_mask.size(1) + batch_max_k + max_new_tokens
        full_attention_mask = torch.ones(
            B, total_mask_len, dtype=prompt_mask.dtype, device=prompt_mask.device
        )
        full_attention_mask[:, : prompt_mask.size(1)] = prompt_mask
        # Mark trajectory padding positions as 0 for samples with k < batch_max_k.
        prompt_len = prompt_mask.size(1)
        for b in range(B):
            k_b = trajectory_lengths[b].item()
            if k_b < batch_max_k:
                full_attention_mask[b, prompt_len + k_b : prompt_len + batch_max_k] = 0

        current_mask_len = unique_mask.size(1) + batch_max_k

        # Auto-regressive decoding.
        all_tokens = torch.full(
            (B, max_new_tokens), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        all_logps = (
            torch.zeros((B, max_new_tokens), dtype=torch.float32, device=input_ids.device)
            if return_logps else None
        )
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        eos_tensor = (
            torch.tensor(eos_token_ids, device=input_ids.device) if eos_token_ids else None
        )
        actual_length = 0

        for step in range(max_new_tokens):
            if return_logps:
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

            next_tokens = m._sample_next_token(next_token_logits, generation_config)

            if return_logps:
                token_logp = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
                token_logp = torch.where(finished, torch.zeros_like(token_logp), token_logp)
                all_logps[:, step] = token_logp

            if eos_tensor is not None:
                is_eos = (next_tokens.unsqueeze(1) == eos_tensor.unsqueeze(0)).any(dim=1)
            else:
                is_eos = torch.zeros_like(finished)

            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_token_id), next_tokens
            )
            all_tokens[:, step] = next_tokens
            actual_length = step + 1
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
