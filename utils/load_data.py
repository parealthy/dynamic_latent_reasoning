import os
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

SUPPORTED_DATASETS = [
    "open-r1/OpenR1-Math-220k",
    "agentica-org/DeepScaler-Preview-Dataset",
    "BytedTsinghua-SIA/DAPO-Math-17k",
    "stepfun-ai/Step-3.5-Flash-SFT",
    "MATH-DATA",
]

DEFAULT_DATA_ROOT = Path(os.environ.get("LRT_DATA_ROOT", "./data/datasets")).expanduser()
MATH_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def _resolve_split(dataset_obj, split: str):
    if isinstance(dataset_obj, DatasetDict):
        if split not in dataset_obj:
            raise KeyError(f"Split '{split}' not found. Available splits: {list(dataset_obj.keys())}")
        return dataset_obj[split]
    return dataset_obj


def _load_from_local_or_hub(
    dataset_name: str,
    *,
    split: str = "train",
    config_name: str | None = None,
):
    local_path = DEFAULT_DATA_ROOT / dataset_name
    if local_path.exists():
        try:
            return _resolve_split(load_from_disk(str(local_path)), split)
        except Exception:
            if config_name is None:
                return load_dataset(str(local_path), split=split)
            return load_dataset(str(local_path), config_name, split=split)

    if config_name is None:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, config_name, split=split)


def _load_saved_dataset_from_disk(*path_parts: str, split: str = "train"):
    local_path = DEFAULT_DATA_ROOT.joinpath(*path_parts)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Expected a saved dataset under {local_path}, but it was not found. "
            f"Set LRT_DATA_ROOT to your local dataset cache or use a public HF dataset id."
        )
    return _resolve_split(load_from_disk(str(local_path)), split)


def _project_dataset(dataset, transform_fn):
    dataset = dataset.map(transform_fn, batched=False)
    columns = set(dataset.column_names) - {"problem", "solution"}
    if columns:
        dataset = dataset.remove_columns(list(columns))
    return dataset


def load_train_data(dataset_name: str):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. Choose from {SUPPORTED_DATASETS}."
        )

    match dataset_name:
        case "open-r1/OpenR1-Math-220k":
            dataset = _load_from_local_or_hub(
                dataset_name,
                config_name="default",
                split="train",
            )
            return _project_dataset(
                dataset,
                lambda example: {
                    "problem": example["problem"] + MATH_SUFFIX,
                    "solution": example["solution"],
                },
            )

        case "agentica-org/DeepScaler-Preview-Dataset":
            dataset = _load_from_local_or_hub(dataset_name, split="train")
            return _project_dataset(
                dataset,
                lambda example: {
                    "problem": example["problem"] + MATH_SUFFIX,
                    "solution": example["answer"],
                },
            )

        case "BytedTsinghua-SIA/DAPO-Math-17k":
            dataset = _load_from_local_or_hub(dataset_name, split="train")
            prefix = (
                "Solve the following math problem step by step. The last line of your response should be "
                "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            )
            return _project_dataset(
                dataset,
                lambda example: {
                    "problem": example["prompt"][0]["content"].removeprefix(prefix) + MATH_SUFFIX,
                    "solution": example["reward_model"]["ground_truth"],
                },
            )

        case "stepfun-ai/Step-3.5-Flash-SFT":
            local_processed_dir = DEFAULT_DATA_ROOT / dataset_name / "single_turn_processed"
            if local_processed_dir.exists():
                dataset = _resolve_split(load_from_disk(str(local_processed_dir)), "train")
            else:
                dataset = _load_from_local_or_hub(dataset_name, split="train")
            return _project_dataset(
                dataset,
                lambda example: {
                    "problem": example["question"],
                    "solution": example["answer"],
                },
            )

        case "MATH-DATA":
            dataset = _load_saved_dataset_from_disk(
                "Deepseek-R1-Distill-Math-Reasoning",
                split="train",
            )
            return _project_dataset(
                dataset,
                lambda example: {
                    "problem": example["problem"],
                    "solution": example["answer"],
                },
            )
