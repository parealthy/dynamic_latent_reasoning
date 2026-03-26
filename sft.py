import os
import sys
from glob import glob
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)

from safetensors import safe_open

sys.path.append("trl/")
LOCAL_TRL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trl")
if LOCAL_TRL_PATH not in sys.path:
    sys.path.append(LOCAL_TRL_PATH)

from trl import SFTConfig

from modeling.reason import TransformerReasoningNet, LatentTransformerReasoningModel
from utils.load_data import load_train_data


@dataclass
class CustomSFTConfig(SFTConfig):
    slow_thinking_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    reasoning_net_path: str = "Qwen/Qwen3-Embedding-0.6B"
    latent_trajectory_length: int = 256
    resume_from_checkpoint: Optional[str] = None

    dataset_name: str = "open-r1/OpenR1-Math-220k"
    prompt_max_length: int = 2048
    completion_max_length: int = 2048
    dataset_kwargs: dict = field(default_factory=lambda: {"skip_prepare_dataset": True})

    output_dir: str = "./checkpoints/"
    logging_dir: str = "logs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    average_tokens_across_devices: bool = False

    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    logging_steps: int = 100
    eval_steps: Optional[int] = None
    eval_strategy: str = "no"
    save_steps: int = 100
    save_strategy: str = "steps"
    save_total_limit: Optional[int] = None
    metric_for_best_model: Optional[str] = "eval_loss"
    load_best_model_at_end: bool = False
    push_to_hub: bool = False

    bf16: bool = True
    tf32: bool = False
    remove_unused_columns: bool = False

    report_to: str = "tensorboard"
    run_name: Optional[str] = "latent-reasoning-sft"
    use_liger: bool = False


def parse_training_args() -> CustomSFTConfig:
    parser = HfArgumentParser(CustomSFTConfig)
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


def _tokenize_with_padding_side(
    tokenizer,
    texts,
    *,
    max_length: int,
    padding_side: str,
):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        return tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding="longest",
        )
    finally:
        tokenizer.padding_side = original_padding_side


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


def load_checkpoint_if_needed(model, resume_path: str) -> None:
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


def build_model(training_args: CustomSFTConfig, tokenizer):
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

    model = LatentTransformerReasoningModel(
        slow_reasoning_model=slow_reasoning_model,
        processor=tokenizer,
        reasoning_network=reasoning_network,
    )
    load_checkpoint_if_needed(model, training_args.resume_from_checkpoint)

    return model


def _build_prompt_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": problem}]


def build_data_collator(tokenizer, training_args: CustomSFTConfig):
    eos_token = tokenizer.eos_token or ""

    def data_collator(batch):
        prompt_messages = [_build_prompt_messages(item["problem"]) for item in batch]
        prompt_texts = [
            tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )
            for message in prompt_messages
        ]

        prompt_inputs = _tokenize_with_padding_side(
            tokenizer,
            prompt_texts,
            max_length=training_args.prompt_max_length,
            padding_side="left",
        )

        completions = [f'{item["solution"]}{eos_token}' for item in batch]
        completion_inputs = _tokenize_with_padding_side(
            tokenizer,
            completions,
            max_length=training_args.completion_max_length,
            padding_side="right",
        )

        return {
            "input_ids": prompt_inputs["input_ids"].long(),
            "labels": completion_inputs["input_ids"].long(),
            "attention_mask": torch.cat(
                (
                    prompt_inputs["attention_mask"],
                    completion_inputs["attention_mask"],
                ),
                dim=-1,
            ).long(),
        }

    return data_collator


def main():
    training_args = parse_training_args()
    print(f"training_args: {training_args}")

    tokenizer = load_tokenizer(training_args.slow_thinking_model_path)
    model = build_model(training_args, tokenizer)
    train_dataset = load_train_data(dataset_name=training_args.dataset_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=build_data_collator(tokenizer, training_args),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
