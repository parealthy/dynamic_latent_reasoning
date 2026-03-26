import ast
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def map(self, fn, batched=False):
        if batched:
            raise AssertionError("This test double only supports batched=False.")
        return FakeDataset([fn(row) for row in self.rows])

    def remove_columns(self, columns):
        return FakeDataset(
            [{key: value for key, value in row.items() if key not in columns} for row in self.rows]
        )


class FakeDatasetDict(dict):
    pass


def load_load_data_module(load_dataset_impl, load_from_disk_impl, data_root: str):
    module_name = "test_release_load_data_module"
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.DatasetDict = FakeDatasetDict
    fake_datasets.load_dataset = load_dataset_impl
    fake_datasets.load_from_disk = load_from_disk_impl

    module_path = REPO_ROOT / "utils" / "load_data.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        with patch.dict(os.environ, {"LRT_DATA_ROOT": data_root}, clear=False):
            spec.loader.exec_module(module)

    return module


class ReleaseSanityTests(unittest.TestCase):
    def test_openr1_falls_back_to_public_hub_dataset(self):
        calls = []

        def fake_load_dataset(name, config_name=None, split=None):
            calls.append((name, config_name, split))
            return FakeDataset(
                [{"problem": "2+2=?", "solution": "\\boxed{4}", "extra": "drop-me"}]
            )

        def fake_load_from_disk(path):
            raise AssertionError(f"load_from_disk should not be called for missing local path: {path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            module = load_load_data_module(fake_load_dataset, fake_load_from_disk, tmpdir)
            dataset = module.load_train_data("open-r1/OpenR1-Math-220k")

        self.assertEqual(
            calls,
            [("open-r1/OpenR1-Math-220k", "default", "train")],
        )
        self.assertEqual(
            dataset.rows,
            [
                {
                    "problem": "2+2=? Let's think step by step and output the final answer within \\boxed{}.",
                    "solution": "\\boxed{4}",
                }
            ],
        )

    def test_stepfun_prefers_local_processed_dataset_when_available(self):
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("load_dataset should not be used when single_turn_processed exists locally.")

        def fake_load_from_disk(path):
            self.assertTrue(path.endswith("stepfun-ai/Step-3.5-Flash-SFT/single_turn_processed"))
            return FakeDataset(
                [{"question": "hello", "answer": "world", "metadata": "drop-me"}]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "stepfun-ai" / "Step-3.5-Flash-SFT" / "single_turn_processed"
            processed_dir.mkdir(parents=True)

            module = load_load_data_module(fake_load_dataset, fake_load_from_disk, tmpdir)
            dataset = module.load_train_data("stepfun-ai/Step-3.5-Flash-SFT")

        self.assertEqual(calls, [])
        self.assertEqual(dataset.rows, [{"problem": "hello", "solution": "world"}])

    def test_math_data_alias_projects_columns_without_self_rename(self):
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("MATH-DATA should load from a saved local dataset, not load_dataset().")

        def fake_load_from_disk(path):
            self.assertTrue(path.endswith("Deepseek-R1-Distill-Math-Reasoning"))
            return FakeDataset(
                [{"problem": "prove it", "answer": "done", "unused": 1}]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            math_dir = Path(tmpdir) / "Deepseek-R1-Distill-Math-Reasoning"
            math_dir.mkdir(parents=True)

            module = load_load_data_module(fake_load_dataset, fake_load_from_disk, tmpdir)
            dataset = module.load_train_data("MATH-DATA")

        self.assertEqual(calls, [])
        self.assertEqual(dataset.rows, [{"problem": "prove it", "solution": "done"}])

    def test_rft_default_dataset_is_public(self):
        tree = ast.parse((REPO_ROOT / "rft.py").read_text())

        dataset_default = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "CustomGRPOConfig":
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == "dataset_name":
                        dataset_default = ast.literal_eval(stmt.value)

        self.assertEqual(dataset_default, "BytedTsinghua-SIA/DAPO-Math-17k")

    def test_sft_scripts_default_to_public_dataset(self):
        expected = 'DATASET_NAME="open-r1/OpenR1-Math-220k"'
        script_paths = [
            REPO_ROOT / "scripts" / "train_sft_1.5B.sh",
            REPO_ROOT / "scripts" / "train_sft_7B.sh",
        ]

        for script_path in script_paths:
            with self.subTest(script=script_path.name):
                self.assertIn(expected, script_path.read_text())

    def test_requirements_include_reward_dependencies(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()
        self.assertIn("latex2sympy2-extended", requirements)
        self.assertIn("math-verify", requirements)


if __name__ == "__main__":
    unittest.main()
