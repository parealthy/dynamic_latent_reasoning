"""
Latent Reasoning Model — Interactive Inference

Enters an interactive REPL loop: user types a question, the model generates
an answer.  Single-turn (no conversation history between questions).

The base model and reasoning network are loaded once and stay resident on
GPU for the entire session.

Usage:
    python inference/run_inference.py \
        --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --reasoning_net_path Qwen/Qwen3-Embedding-0.6B \
        --checkpoint_path checkpoints/checkpoint-500
"""

import os
import sys
import argparse
from glob import glob
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modeling.reason import TransformerReasoningNet
from modeling.adaptive_reason import AdaptiveTransformerReasoningNet


def _resolve_hidden_size(config) -> int:
    """Extract hidden_size from a HF model config."""
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError("Failed to resolve hidden_size from the base model config.")


def _load_reasoning_weights(reasoning_network, checkpoint_path: str):
    """Load reasoning_network.* weights from a checkpoint directory."""
    safetensor_files = glob(os.path.join(checkpoint_path, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No safetensors files found in {checkpoint_path}"
        )

    state_dict = {}
    for filename in safetensor_files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("reasoning_network."):
                    new_key = key[len("reasoning_network."):]
                    state_dict[new_key] = handle.get_tensor(key)

    if not state_dict:
        raise ValueError(
            f"No reasoning_network.* keys found in checkpoint at {checkpoint_path}"
        )

    reasoning_network.load_state_dict(state_dict, strict=True)
    print(f"  Loaded {len(state_dict)} reasoning network weight tensors from checkpoint.")


def _checkpoint_has_adaptive_weights(checkpoint_path: str) -> bool:
    """Return True if the checkpoint contains DifficultyEstimator weights."""
    safetensor_files = glob(os.path.join(checkpoint_path, "*.safetensors"))
    for filename in safetensor_files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("reasoning_network.difficulty_estimator."):
                    return True
    return False


def _parse_bool_auto(value: str) -> Optional[bool]:
    value = value.lower()
    if value == "auto":
        return None
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(
        "--use_adaptive_length must be one of: auto, true, false"
    )


def _parse_length_candidates(
    length_candidates: str,
    latent_trajectory_length: int,
) -> list[int]:
    if length_candidates:
        candidates = sorted(
            {int(item.strip()) for item in length_candidates.split(",") if item.strip()}
        )
    else:
        step = max(latent_trajectory_length // 4, 1)
        candidates = [
            step * i for i in range(1, 5)
            if step * i <= latent_trajectory_length
        ]

    if not candidates:
        raise ValueError("length_candidates must contain at least one length.")
    invalid = [
        length for length in candidates
        if length <= 0 or length > latent_trajectory_length
    ]
    if invalid:
        raise ValueError(
            "length_candidates must be positive and no larger than "
            f"latent_trajectory_length={latent_trajectory_length}; got {invalid}"
        )
    return candidates


def _last_layer_hidden_states(model, inputs_embeds, attention_mask):
    """
    Extract the same final hidden states used during training.

    The training model reads ``output.hidden_states[-1]``. Avoid forward hooks
    here because they can capture a decoder block output before the model's
    final normalization layer.
    """
    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,
    )
    return output.hidden_states[-1]


class LatentReasoningInteractive:
    """
    Interactive inference wrapper.

    Loads the HF base model + reasoning network once, keeps them on GPU,
    and uses HF generate() for autoregressive decoding.
    """

    def __init__(
        self,
        model_path: str,
        reasoning_net_path: str,
        checkpoint_path: str,
        latent_trajectory_length: int = 256,
        use_adaptive_length: Optional[bool] = None,
        length_candidates: str = "",
        prompt_max_length: int = 1024,
        max_new_tokens: int = 2048,
        device: str = "cuda",
    ):
        self.prompt_max_length = prompt_max_length
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.last_trajectory_lengths: Optional[torch.Tensor] = None

        checkpoint_is_adaptive = _checkpoint_has_adaptive_weights(checkpoint_path)
        if use_adaptive_length is None:
            use_adaptive_length = checkpoint_is_adaptive
        if checkpoint_is_adaptive and not use_adaptive_length:
            raise ValueError(
                "Checkpoint contains adaptive DifficultyEstimator weights, but "
                "--use_adaptive_length=false was requested."
            )
        if use_adaptive_length and not checkpoint_is_adaptive:
            raise ValueError(
                "--use_adaptive_length=true was requested, but the checkpoint "
                "does not contain reasoning_network.difficulty_estimator.* weights."
            )
        self.use_adaptive_length = use_adaptive_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model
        print("  Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Reasoning network
        hidden_size = _resolve_hidden_size(self.model.config)
        print("  Loading reasoning network...")
        if self.use_adaptive_length:
            parsed_candidates = _parse_length_candidates(
                length_candidates,
                latent_trajectory_length,
            )
            print(
                "  Adaptive trajectory length enabled "
                f"(candidates={parsed_candidates})."
            )
            self.reasoning_network = AdaptiveTransformerReasoningNet(
                reasoning_net_path,
                latent_trajectory_length=latent_trajectory_length,
                hidden_size=hidden_size,
                length_candidates=parsed_candidates,
            )
        else:
            self.reasoning_network = TransformerReasoningNet(
                reasoning_net_path,
                latent_trajectory_length=latent_trajectory_length,
                hidden_size=hidden_size,
            )
        self.reasoning_network.to(device)
        self.reasoning_network.eval()
        _load_reasoning_weights(self.reasoning_network, checkpoint_path)

    @torch.no_grad()
    def generate(self, user_input: str, temperature: float = 0.0) -> str:
        """
        Run latent-reasoning inference for a single user input.
        Returns the generated text.
        """
        self.last_trajectory_lengths = None

        messages = [{"role": "user", "content": user_input}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_max_length,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        prompt_embeddings = self.model.get_input_embeddings()(input_ids)
        prompt_embeddings = prompt_embeddings.to(self.model.dtype)

        hidden_states = _last_layer_hidden_states(
            self.model, prompt_embeddings, attention_mask,
        )

        trajectory_lengths = None
        if self.use_adaptive_length:
            trajectory_lengths, _ = (
                self.reasoning_network.difficulty_estimator.select_lengths(
                    hidden_states,
                    temperature=0.0,
                )
            )
            self.last_trajectory_lengths = trajectory_lengths.detach().cpu()

        if trajectory_lengths is None:
            latent_trajectory = self.reasoning_network(
                hidden_states,
                attention_mask=attention_mask,
            ).to(prompt_embeddings.dtype)
        else:
            latent_trajectory = self.reasoning_network(
                hidden_states,
                attention_mask=attention_mask,
                trajectory_lengths=trajectory_lengths,
            ).to(prompt_embeddings.dtype)

        combined_embeds = torch.cat(
            [prompt_embeddings, latent_trajectory], dim=1,
        )
        latent_mask = torch.ones(
            latent_trajectory.size(0),
            latent_trajectory.size(1),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        if trajectory_lengths is not None:
            traj_range = torch.arange(
                latent_trajectory.size(1),
                device=self.device,
            ).unsqueeze(0)
            latent_mask = (
                traj_range < trajectory_lengths.to(self.device).unsqueeze(1)
            ).to(attention_mask.dtype)
        combined_mask = torch.cat([attention_mask, latent_mask], dim=1).long()

        generate_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95

        output_ids = self.model.generate(**generate_kwargs)
        return self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Reasoning Interactive Inference",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reasoning_net_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--latent_trajectory_length", type=int, default=256)
    parser.add_argument(
        "--use_adaptive_length",
        type=_parse_bool_auto,
        default=None,
        help=(
            "auto/true/false. auto enables adaptive length when the checkpoint "
            "contains DifficultyEstimator weights."
        ),
    )
    parser.add_argument(
        "--length_candidates",
        type=str,
        default="",
        help="Comma-separated adaptive trajectory candidates, e.g. 64,128,192,256.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--show_trajectory_length",
        action="store_true",
        help="Print the selected adaptive latent trajectory length before each answer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\nLoading model...")
    model = LatentReasoningInteractive(
        model_path=args.model_path,
        reasoning_net_path=args.reasoning_net_path,
        checkpoint_path=args.checkpoint_path,
        latent_trajectory_length=args.latent_trajectory_length,
        use_adaptive_length=args.use_adaptive_length,
        length_candidates=args.length_candidates,
        prompt_max_length=args.prompt_max_length,
        max_new_tokens=args.max_new_tokens,
    )
    print("Model loaded. Type your question (or 'exit' to quit).\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        try:
            answer = model.generate(user_input, temperature=args.temperature)
            if args.show_trajectory_length and model.last_trajectory_lengths is not None:
                lengths = model.last_trajectory_lengths.tolist()
                print(f"\n[latent_trajectory_length={lengths[0]}]")
            print(f"\n{answer}\n")
        except KeyboardInterrupt:
            print("\n[Generation interrupted]\n")
            continue


if __name__ == "__main__":
    main()
