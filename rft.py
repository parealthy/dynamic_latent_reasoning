import os
import sys
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from glob import glob
from typing import Callable, Optional

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

sys.path.append("trl/")
LOCAL_TRL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trl")
if LOCAL_TRL_PATH not in sys.path:
    sys.path.append(LOCAL_TRL_PATH)

from trl import GRPOConfig

from modeling.reason import LatentTransformerReasoningModel, TransformerReasoningNet
from modeling.adaptive_reason import (
    AdaptiveTransformerReasoningNet,
    AdaptiveLatentGRPOModel,
)
from trainer.grpo_trainer import GRPOTrainer
from trainer.adaptive_grpo_trainer import AdaptiveGRPOTrainer
from utils.load_data import load_train_data
from utils.reward_func import accuracy_reward, length_penalty_reward, trajectory_efficiency_reward


REWARD_FNS: dict[str, Callable[..., list[Optional[float]]]] = {
    "accuracy": accuracy_reward,
}

LENGTH_PENALTY_WEIGHT = 0.2


@dataclass
class CustomGRPOConfig(GRPOConfig):
    slow_thinking_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    reasoning_net_path: str = "Qwen/Qwen3-Embedding-0.6B"
    latent_trajectory_length: int = 256
    # ---- Adaptive Latent Trajectory (ALT) ----
    use_adaptive_length: bool = False
    length_candidates: str = ""          # e.g. "64,128,192,256"
    diff_sample_temperature: float = 1.0 # exploration temp for DifficultyEstimator
    diff_reinforce_weight: float = 0.05  # λ for REINFORCE aux loss
    # Trajectory cost reward weight. Controls the strength of the "short-k
    # preference" signal injected into GRPO advantage. Without this, the
    # DifficultyEstimator only learns "which k is easier to answer correctly",
    # NOT "the shortest k that still answers correctly".
    trajectory_efficiency_weight: float = 0.3
    resume_from_checkpoint: Optional[str] = None

    dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k"
    reward_metric: str = "accuracy"

    # DAPO-style asymmetric PPO clipping [1 - eps_low, 1 + eps_high] = [0.8, 1.28]
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28

    output_dir: str = "./checkpoints/"
    logging_dir: str = "logs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False

    learning_rate: float = 1e-5
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    logging_steps: int = 100
    eval_strategy: str = "no"
    save_steps: int = 100
    save_strategy: str = "steps"
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = False
    push_to_hub: bool = False

    bf16: bool = True
    tf32: bool = False
    remove_unused_columns: bool = False

    report_to: str = "tensorboard"
    run_name: Optional[str] = "latent-reasoning-rft"

    def __post_init__(self):
        super().__post_init__()
        if self.reward_metric not in REWARD_FNS:
            valid_metrics = ", ".join(sorted(REWARD_FNS))
            raise ValueError(
                f"Unsupported reward_metric: {self.reward_metric}. Choose from: {valid_metrics}."
            )


def parse_training_args() -> CustomGRPOConfig:
    parser = HfArgumentParser(CustomGRPOConfig)
    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.ddp_find_unused_parameters = False
    return training_args


def _build_device_map() -> dict[str, int]:
    return {"": int(os.environ.get("LOCAL_RANK") or 0)}


def _resolve_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_checkpoint_if_needed(model, resume_path: Optional[str]) -> None:
    if not resume_path or not os.path.exists(resume_path):
        return

    print(f"Loading checkpoint from {resume_path}")
    safetensor_files = glob(os.path.join(resume_path, "*.safetensors"))
    if safetensor_files:
        state_dict = {}
        for filename in safetensor_files:
            with safe_open(filename, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    state_dict[key] = handle.get_tensor(key)
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint from safetensors")
        return

    pytorch_model = os.path.join(resume_path, "pytorch_model.bin")
    if os.path.exists(pytorch_model):
        state_dict = torch.load(pytorch_model, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint from pytorch_model.bin")
        return

    print(f"Warning: no checkpoint weights found in {resume_path}")


def _build_prompt_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": problem}]


def build_reward_function(metric_name: str):
    reward_fn = REWARD_FNS[metric_name]

    def reward_wrapper(prompts, completions, solution, **kwargs):
        del prompts
        rewards = reward_fn(completions=completions, solution=solution, **kwargs)
        return [0.0 if reward is None else float(reward) for reward in rewards]

    reward_wrapper.__name__ = f"{metric_name}_reward"
    return reward_wrapper


def build_length_penalty(max_completion_length: int):
    def length_penalty_wrapper(prompts, completions, **kwargs):
        del prompts
        return length_penalty_reward(
            completions=completions,
            max_completion_length=max_completion_length,
            completion_token_lengths=kwargs.get("completion_token_lengths"),
        )

    length_penalty_wrapper.__name__ = "length_penalty"
    return length_penalty_wrapper


def build_trajectory_efficiency(
    model,
    max_trajectory_length: int,
    efficiency_weight: float = 0.3,
):
    """
    Trajectory cost reward that closes over the adaptive model.

    At call time it pulls model._last_trajectory_lengths (set during
    generate() in AdaptiveLatentGRPOModel) and combines with accuracy.
    Adding this reward to the GRPO objective is what teaches the
    DifficultyEstimator to actually prefer *shorter* k — without it,
    GRPO's group-normalised advantage cannot distinguish
    "correct with k=64" from "correct with k=256".

    Returns:
        per-sample reward: efficiency_weight * (1 - k/max) if correct, else 0
    """

    def reward_wrapper(prompts, completions, solution, **kwargs):
        del prompts
        traj_lens = getattr(model, "_last_trajectory_lengths", None)
        if traj_lens is None:
            return [0.0] * len(completions)
        if hasattr(traj_lens, "tolist"):
            traj_lens = traj_lens.tolist()
        else:
            traj_lens = list(traj_lens)
        return trajectory_efficiency_reward(
            completions=completions,
            solution=solution,
            trajectory_lengths=traj_lens,
            max_trajectory_length=max_trajectory_length,
            efficiency_weight=efficiency_weight,
            **kwargs,
        )

    reward_wrapper.__name__ = "trajectory_efficiency"
    return reward_wrapper


class LatentTransformerReasoningForGRPO(LatentTransformerReasoningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warnings_issued = {}
        self._model_tags: list[str] = []
        self.name_or_path = getattr(self.config, "_name_or_path", "")
        self.prepare_inputs_for_generation = self.slow_reasoning_model.prepare_inputs_for_generation

    def add_model_tags(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        self._model_tags.extend(tags)

    def __deepcopy__(self, memo):
        copied_model = type(self)(
            slow_reasoning_model=copy.deepcopy(self.slow_reasoning_model, memo),
            processor=self.processor,
            reasoning_network=copy.deepcopy(self.reasoning_network, memo),
        )
        copied_model.warnings_issued = dict(self.warnings_issued)
        copied_model._model_tags = list(self._model_tags)
        copied_model.name_or_path = self.name_or_path
        memo[id(self)] = copied_model
        return copied_model

    @property
    def device(self):
        return next(self.parameters()).device

    @contextmanager
    def disable_adapter(self):
        yield self

    def _sample_next_token(self, next_token_logits, generation_config):
        temperature = getattr(generation_config, "temperature", 1.0)
        do_sample = getattr(generation_config, "do_sample", True)
        if not do_sample or temperature is None or temperature == 0:
            return next_token_logits.argmax(dim=-1)

        next_token_logits = next_token_logits / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _build_grpo_inputs(
        self,
        prompt_ids: torch.LongTensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.LongTensor,
        completion_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cached_prompt_kv=None,
        cached_prompt_hidden_states=None,
        cached_expand_idx=None,
        **kwargs,
    ):
        if completion_mask is None:
            completion_mask = (completion_ids != self.pad_token_id).long()

        prompt_mask = prompt_mask.long()
        completion_mask = completion_mask.long()

        if cached_prompt_kv is not None and cached_prompt_hidden_states is not None:
            B = prompt_ids.size(0)
            U = cached_prompt_hidden_states.size(0)

            if U < B and cached_expand_idx is not None:
                prompt_kv = self._expand_kv_cache(cached_prompt_kv, cached_expand_idx)
                prompt_hidden_states = cached_prompt_hidden_states[cached_expand_idx]
            else:
                prompt_kv = cached_prompt_kv
                prompt_hidden_states = cached_prompt_hidden_states
        else:
            with torch.no_grad():
                prompt_embeddings = self.get_input_embeddings(prompt_ids).to(
                    self.slow_reasoning_model.dtype
                )
                prompt_outputs = self.slow_reasoning_model(
                    inputs_embeds=prompt_embeddings,
                    attention_mask=prompt_mask,
                    position_ids=position_ids,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=True,
                    **kwargs,
                )
                prompt_hidden_states = prompt_outputs.hidden_states[-1].detach()
                prompt_kv = prompt_outputs.past_key_values

        latent_trajectory = self.reasoning_network(
            prompt_hidden_states,
            attention_mask=prompt_mask,
        ).to(self.slow_reasoning_model.dtype)

        latent_trajectory_mask = torch.ones(
            latent_trajectory.size(0),
            latent_trajectory.size(1),
            device=prompt_mask.device,
            dtype=prompt_mask.dtype,
        )
        completion_embeddings = self.get_input_embeddings(completion_ids).to(
            self.slow_reasoning_model.dtype
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
        **kwargs,
    ):
        if prompt_ids is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                **kwargs,
            )

        kwargs.pop("return_dict", None)
        kwargs.pop("logits_to_keep", None)

        (
            prompt_mask,
            prompt_kv,
            latent_trajectory,
            latent_trajectory_mask,
            completion_embeddings,
            completion_mask,
        ) = self._build_grpo_inputs(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            position_ids=position_ids,
            cached_prompt_kv=cached_prompt_kv,
            cached_prompt_hidden_states=cached_prompt_hidden_states,
            cached_expand_idx=cached_expand_idx,
            **kwargs,
        )

        input_embeddings = torch.cat(
            [latent_trajectory, completion_embeddings],
            dim=1,
        )
        input_mask = torch.cat(
            [prompt_mask, latent_trajectory_mask, completion_mask],
            dim=1,
        ).long()

        outputs = self.slow_reasoning_model(
            inputs_embeds=input_embeddings,
            attention_mask=input_mask,
            past_key_values=prompt_kv,
            return_dict=True,
            **kwargs,
        )

        logits_start = latent_trajectory.size(1) - 1
        logits_end = logits_start + completion_ids.size(1)
        return outputs.logits[:, logits_start:logits_end, :]

    @staticmethod
    def _dedup_prompts(input_ids, attention_mask):
        """Find unique prompts and return indices for reconstruction."""
        B = input_ids.size(0)
        if B <= 1:
            return input_ids, attention_mask, torch.zeros(B, dtype=torch.long, device=input_ids.device)

        all_same = (input_ids[0:1] == input_ids).all()
        if all_same:
            return (
                input_ids[:1],
                attention_mask[:1],
                torch.zeros(B, dtype=torch.long, device=input_ids.device),
            )

        unique_indices = [0]
        expand_idx = torch.zeros(B, dtype=torch.long, device=input_ids.device)
        for i in range(1, B):
            found = False
            for j, ui in enumerate(unique_indices):
                if (input_ids[i] == input_ids[ui]).all():
                    expand_idx[i] = j
                    found = True
                    break
            if not found:
                expand_idx[i] = len(unique_indices)
                unique_indices.append(i)

        unique_indices_t = torch.tensor(unique_indices, device=input_ids.device)
        return input_ids[unique_indices_t], attention_mask[unique_indices_t], expand_idx

    @staticmethod
    def _expand_kv_cache(past_key_values, expand_idx):
        """Expand KV cache from unique prompts to full batch using expand_idx."""
        from transformers.cache_utils import DynamicCache

        if isinstance(past_key_values, DynamicCache):
            new_cache = DynamicCache()
            for layer_idx in range(len(past_key_values)):
                keys, values = past_key_values[layer_idx]
                new_cache.update(keys[expand_idx], values[expand_idx], layer_idx)
            return new_cache

        expanded = []
        for layer_kv in past_key_values:
            expanded.append(tuple(t[expand_idx] for t in layer_kv))
        return type(past_key_values)(expanded) if not isinstance(past_key_values, list) else expanded

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
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if generation_config is None:
            generation_config = self.slow_reasoning_model.generation_config

        max_new_tokens = getattr(generation_config, "max_new_tokens", None)
        if max_new_tokens is None:
            raise ValueError("generation_config.max_new_tokens must be set for GRPO generation.")

        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_ids = []
        elif isinstance(eos_token_id, int):
            eos_token_ids = [eos_token_id]
        else:
            eos_token_ids = list(eos_token_id)

        pad_token_id = getattr(generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.pad_token_id

        B = input_ids.size(0)
        prompt_mask = attention_mask.long()

        with torch.no_grad():
            unique_ids, unique_mask, expand_idx = self._dedup_prompts(input_ids, prompt_mask)
            U = unique_ids.size(0)

            unique_embeddings = self.get_input_embeddings(unique_ids).to(
                self.slow_reasoning_model.dtype
            )
            prompt_outputs = self.slow_reasoning_model(
                inputs_embeds=unique_embeddings,
                attention_mask=unique_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
                **kwargs,
            )
            unique_hidden_states = prompt_outputs.hidden_states[-1].detach()
            unique_kv = prompt_outputs.past_key_values

            unique_latent = self.reasoning_network(
                unique_hidden_states,
                attention_mask=unique_mask,
            ).to(unique_embeddings.dtype)

            base_mask_len = unique_mask.size(1) + unique_latent.size(1)
            unique_model_mask = torch.ones(
                U, base_mask_len,
                dtype=unique_mask.dtype,
                device=unique_mask.device,
            )
            unique_model_mask[:, :unique_mask.size(1)] = unique_mask

            outputs = self.slow_reasoning_model(
                inputs_embeds=unique_latent,
                attention_mask=unique_model_mask,
                past_key_values=unique_kv,
                use_cache=True,
                return_dict=True,
            )
            unique_full_kv = outputs.past_key_values
            unique_next_logits = outputs.logits[:, -1, :]

            cached_unique_hidden_states = unique_hidden_states
            cached_unique_kv = unique_kv

            if U < B:
                past_key_values = self._expand_kv_cache(unique_full_kv, expand_idx)
                next_token_logits = unique_next_logits[expand_idx]
            else:
                past_key_values = unique_full_kv
                next_token_logits = unique_next_logits

            total_mask_len = base_mask_len + max_new_tokens
            full_attention_mask = torch.ones(
                B, total_mask_len,
                dtype=prompt_mask.dtype,
                device=prompt_mask.device,
            )
            full_attention_mask[:, :prompt_mask.size(1)] = prompt_mask
            current_mask_len = base_mask_len

            all_tokens = torch.full(
                (B, max_new_tokens), pad_token_id,
                dtype=input_ids.dtype, device=input_ids.device,
            )
            all_logps = torch.zeros(
                (B, max_new_tokens), dtype=torch.float32, device=input_ids.device,
            ) if return_logps else None

            finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
            eos_token_tensor = (
                torch.tensor(eos_token_ids, device=input_ids.device)
                if eos_token_ids
                else None
            )
            actual_length = 0

            for step in range(max_new_tokens):
                if return_logps:
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)

                next_tokens = self._sample_next_token(
                    next_token_logits, generation_config
                )

                if return_logps:
                    token_logp = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
                    token_logp = torch.where(finished, torch.zeros_like(token_logp), token_logp)
                    all_logps[:, step] = token_logp

                if eos_token_tensor is not None:
                    is_eos = (next_tokens.unsqueeze(1) == eos_token_tensor.unsqueeze(0)).any(dim=1)
                else:
                    is_eos = torch.zeros_like(finished)

                next_tokens = torch.where(
                    finished,
                    torch.full_like(next_tokens, pad_token_id),
                    next_tokens,
                )
                all_tokens[:, step] = next_tokens
                actual_length = step + 1
                finished = finished | is_eos
                if finished.all():
                    break

                current_mask_len += 1
                outputs = self.slow_reasoning_model(
                    input_ids=next_tokens.unsqueeze(1),
                    attention_mask=full_attention_mask[:, :current_mask_len],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

        completion_ids = all_tokens[:, :actual_length]

        if actual_length == 0:
            empty_ids = input_ids.new_full((B, 0), pad_token_id)
            result = (empty_ids,)
            if return_logps:
                result += (empty_ids.new_zeros((B, 0), dtype=torch.float32),)
            if return_cached_states:
                result += (cached_unique_kv, cached_unique_hidden_states, expand_idx)
            return result if len(result) > 1 else result[0]

        if return_logps and return_cached_states:
            return completion_ids, all_logps[:, :actual_length], cached_unique_kv, cached_unique_hidden_states, expand_idx
        elif return_logps:
            return completion_ids, all_logps[:, :actual_length]
        elif return_cached_states:
            return completion_ids, cached_unique_kv, cached_unique_hidden_states, expand_idx
        return completion_ids


def _parse_length_candidates(args: CustomGRPOConfig) -> list[int]:
    if args.length_candidates:
        return sorted(int(x.strip()) for x in args.length_candidates.split(","))
    step = max(args.latent_trajectory_length // 4, 1)
    return [step * i for i in range(1, 5) if step * i <= args.latent_trajectory_length]


def build_model(training_args: CustomGRPOConfig, tokenizer):
    device_map = _build_device_map()
    slow_reasoning_model = AutoModelForCausalLM.from_pretrained(
        training_args.slow_thinking_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    slow_reasoning_model.config.use_cache = False

    reasoning_network = TransformerReasoningNet(
        training_args.reasoning_net_path,
        latent_trajectory_length=training_args.latent_trajectory_length,
        hidden_size=_resolve_hidden_size(slow_reasoning_model.config),
    ).to(next(slow_reasoning_model.parameters()).device)

    if training_args.use_adaptive_length:
        length_candidates = _parse_length_candidates(training_args)
        print(f"[ALT-GRPO] Adaptive length with candidates={length_candidates}, "
              f"diff_temperature={training_args.diff_sample_temperature}")
        reasoning_network = AdaptiveTransformerReasoningNet(
            training_args.reasoning_net_path,
            latent_trajectory_length=training_args.latent_trajectory_length,
            hidden_size=_resolve_hidden_size(slow_reasoning_model.config),
            length_candidates=length_candidates,
        ).to(next(slow_reasoning_model.parameters()).device)

        base_model = LatentTransformerReasoningForGRPO(
            slow_reasoning_model=slow_reasoning_model,
            processor=tokenizer,
            reasoning_network=reasoning_network,
        )
        model = AdaptiveLatentGRPOModel(
            base_grpo_model=base_model,
            diff_sample_temperature=training_args.diff_sample_temperature,
        )
    else:
        model = LatentTransformerReasoningForGRPO(
            slow_reasoning_model=slow_reasoning_model,
            processor=tokenizer,
            reasoning_network=reasoning_network,
        )
    load_checkpoint_if_needed(model, training_args.resume_from_checkpoint)
    return model


def build_train_dataset(training_args: CustomGRPOConfig):
    train_dataset = load_train_data(dataset_name=training_args.dataset_name)

    def format_example(example):
        return {
            "prompt": _build_prompt_messages(example["problem"]),
            "solution": example["solution"],
        }

    return train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names,
    )


def main():
    training_args = parse_training_args()
    print(f"training_args: {training_args}")

    torch.backends.cudnn.benchmark = True

    tokenizer = load_tokenizer(training_args.slow_thinking_model_path)
    model = build_model(training_args, tokenizer)
    train_dataset = build_train_dataset(training_args)

    reward_fns = [build_reward_function(training_args.reward_metric)]
    reward_weights = [1.0]

    length_penalty_fn = build_length_penalty(training_args.max_completion_length)
    reward_fns.append(length_penalty_fn)
    reward_weights.append(LENGTH_PENALTY_WEIGHT)

    # ALT: trajectory cost reward — essential for teaching DifficultyEstimator
    # to prefer the shortest k that still answers correctly. Without it,
    # GRPO advantage cannot distinguish (correct, k=64) from (correct, k=256).
    if training_args.use_adaptive_length:
        eff_fn = build_trajectory_efficiency(
            model=model,
            max_trajectory_length=training_args.latent_trajectory_length,
            efficiency_weight=training_args.trajectory_efficiency_weight,
        )
        reward_fns.append(eff_fn)
        reward_weights.append(1.0)  # absolute scale already controlled by efficiency_weight

    training_args.reward_weights = reward_weights

    TrainerClass = AdaptiveGRPOTrainer if training_args.use_adaptive_length else GRPOTrainer
    trainer_kwargs = {}
    if training_args.use_adaptive_length:
        trainer_kwargs["diff_reinforce_weight"] = training_args.diff_reinforce_weight

    trainer = TrainerClass(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        **trainer_kwargs,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
