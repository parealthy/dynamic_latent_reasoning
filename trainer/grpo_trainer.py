from typing import Any, Union

import torch
from accelerate.utils import gather, gather_object
from torch import nn

from trl import GRPOTrainer as TRLGRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import RepeatRandomSampler
from trl.trainer.utils import selective_log_softmax


class GRPOTrainer(TRLGRPOTrainer):
    """Repo-local GRPO trainer that works with the latent reasoning policy model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DAPO-style asymmetric clip ratio [1 - eps_low, 1 + eps_high]
        self.clip_eps_low = getattr(self.args, "clip_eps_low", 0.2)
        self.clip_eps_high = getattr(self.args, "clip_eps_high", 0.28)

        if self.beta == 0.0 and getattr(self, "ref_model", None) is not None:
            del self.ref_model
            self.ref_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_train_sampler(self, train_dataset=None):
        data_source = train_dataset if train_dataset is not None else self.train_dataset
        return RepeatRandomSampler(data_source, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

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
    ):
        logits = model(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            logits_to_keep=logits_to_keep,
            cached_prompt_kv=cached_prompt_kv,
            cached_prompt_hidden_states=cached_prompt_hidden_states,
            cached_expand_idx=cached_expand_idx,
        )
        if logits.size(1) != completion_ids.size(1):
            raise ValueError(
                f"Expected logits length {completion_ids.size(1)}, but got {logits.size(1)}"
            )
        return selective_log_softmax(logits, completion_ids)

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [example["prompt"] for example in inputs]
        prompts_text = [
            self.processing_class.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = prompt_inputs.to(device)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            completion_ids, old_per_token_logps, cached_prompt_kv, cached_prompt_hidden_states, cached_expand_idx = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                return_logps=True,
                return_cached_states=True,
            )

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0),
            -1,
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        prompt_ids,
                        prompt_mask,
                        completion_ids,
                        completion_mask,
                    )
                else:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    if hasattr(unwrapped_model, "disable_adapter"):
                        with unwrapped_model.disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                self.model,
                                prompt_ids,
                                prompt_mask,
                                completion_ids,
                                completion_mask,
                            )
                    else:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            prompt_ids,
                            prompt_mask,
                            completion_ids,
                            completion_mask,
                        )
            # KV cache is invalidated after ref model forward; don't reuse
            cached_prompt_kv = None
            cached_prompt_hidden_states = None
            cached_expand_idx = None
        else:
            ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        completions = []
        for prompt, completion_text in zip(prompts, completions_text):
            bootstrap = prompt[-1]["content"] if prompt and prompt[-1]["role"] == "assistant" else ""
            completions.append([{"role": "assistant", "content": bootstrap + completion_text}])

        # Compute per-sample completion token lengths for reward functions
        completion_token_lengths = completion_mask.sum(dim=1).tolist()

        rewards_per_func = torch.zeros(
            len(prompts),
            len(self.reward_funcs),
            device=device,
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            del reward_processing_class
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            reward_kwargs["completion_token_lengths"] = completion_token_lengths
            output_reward_func = reward_func(
                prompts=prompts,
                completions=completions,
                **reward_kwargs,
            )
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func,
                dtype=torch.float32,
                device=device,
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)
        std_grouped_rewards = grouped_rewards.std(dim=1)

        # Mask out zero-variance groups (e.g. all rewards are 0) to prevent dead gradients.
        valid_group_mask = std_grouped_rewards > 1e-8
        num_valid_groups = valid_group_mask.sum().item()
        num_total_groups = grouped_rewards.size(0)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        valid_mask = valid_group_mask.repeat_interleave(self.num_generations, dim=0)

        if num_valid_groups > 0:
            # Standard: normalize within groups that have variance
            advantages = torch.where(
                valid_mask,
                (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4),
                torch.zeros_like(rewards),
            )
        else:
            # Fallback: ALL groups have zero variance. Use batch-level normalization
            # so the model still gets gradient from samples where reward differs
            # from the batch mean.
            batch_mean = rewards.mean()
            batch_std = rewards.std()
            if batch_std > 1e-8:
                advantages = (rewards - batch_mean) / (batch_std + 1e-4)
            else:
                # Truly no signal at all (e.g., all rewards are 0) — nothing to learn
                advantages = torch.zeros_like(rewards)

        # Clip advantages to prevent extreme gradient magnitudes
        advantages = torch.clamp(advantages, -5.0, 5.0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["valid_groups_ratio"].append(
            num_valid_groups / max(num_total_groups, 1)
        )

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            pd.DataFrame(table)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "old_per_token_logps": old_per_token_logps,
            "advantages": advantages,
            "cached_prompt_kv": cached_prompt_kv,
            "cached_prompt_hidden_states": cached_prompt_hidden_states,
            "cached_expand_idx": cached_expand_idx,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        cached_prompt_kv = inputs.get("cached_prompt_kv")
        cached_prompt_hidden_states = inputs.get("cached_prompt_hidden_states")
        cached_expand_idx = inputs.get("cached_expand_idx")

        per_token_logps = self._get_per_token_logps(
            model,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            cached_prompt_kv=cached_prompt_kv,
            cached_prompt_hidden_states=cached_prompt_hidden_states,
            cached_expand_idx=cached_expand_idx,
        )

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"]

        # DAPO-style asymmetric PPO clipping
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_eps_low,
            1.0 + self.clip_eps_high,
        )
        advantages_expanded = advantages.unsqueeze(1)
        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded
        per_token_loss = -torch.min(surr1, surr2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Mask out samples with zero advantage (from zero-variance groups) so they
        # don't dilute the loss or contribute noise.
        sample_has_signal = (advantages.abs() > 1e-8).float()
        per_sample_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        num_valid = sample_has_signal.sum().clamp(min=1.0)
        loss = (per_sample_loss * sample_has_signal).sum() / num_valid

        # Log clip fraction for diagnostics
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.clip_eps_low).float().mean().item()
            self._metrics["clip_fraction"].append(clip_frac)

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        self._metrics["completion_length"].append(completion_length)

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss