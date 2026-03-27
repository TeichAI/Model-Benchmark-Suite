import ast
import datetime
import gc
import io
import json
import os
import sys
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import quote

import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
RUNNER_PATH = ROOT / "benchmarks" / "run_lm_eval.py"


def load_symbols(path, function_names, assignment_names, extra_globals):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected_nodes = []
    function_names = set(function_names)
    assignment_names = set(assignment_names)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if assignment_names & target_names:
                selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace = dict(extra_globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_app_namespace():
    namespace = load_symbols(
        APP_PATH,
        function_names=[
            "is_valid_lm_eval_payload",
            "get_model_filename_keys",
            "get_cache_path_candidates",
            "get_raw_result_path",
            "get_deepeval_result_path_candidates",
            "benchmark_supports_chat_template",
            "get_effective_chat_template_settings",
            "get_task_config",
            "get_primary_metric_name",
            "format_score_percent",
            "build_eval_results_note",
            "build_eval_results_yaml",
            "build_eval_results_bundle",
            "export_plotly_figure",
            "summarize_results",
            "get_run_config",
        ],
        assignment_names=[
            "EVAL_RESULTS_FALLBACKS",
            "CHAT_TEMPLATE_UNSAFE_BENCHMARKS",
        ],
        extra_globals={
            "os": os,
            "json": json,
            "io": io,
            "zipfile": zipfile,
            "datetime": datetime,
            "quote": quote,
            "go": go,
            "pio": pio,
        },
    )
    namespace.update(
        {
            "apply_chat_template": True,
            "backend": "hf",
            "quantization": "none",
            "vllm_max_model_len": 8192,
            "allow_code_eval": False,
            "chat_template_kwargs": {"enable_thinking": True},
            "num_fewshot": None,
            "override_gen_kwargs": False,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "batch_size": 1,
        }
    )
    return namespace


def load_runner_namespace(fake_evaluator, fake_torch):
    return load_symbols(
        RUNNER_PATH,
        function_names=["task_supports_chat_template", "run_lm_eval"],
        assignment_names=["CHAT_TEMPLATE_UNSAFE_TASKS"],
        extra_globals={
            "os": os,
            "gc": gc,
            "sys": sys,
            "torch": fake_torch,
            "evaluator": fake_evaluator,
            "clear_torch_cache": lambda: None,
        },
    )


class AppHelperTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_app_namespace()

    def test_model_filename_keys_include_encoded_and_legacy_forms(self):
        keys = self.ns["get_model_filename_keys"]("TeichAI/Qwen3.5-4B")
        self.assertEqual(keys[0], "TeichAI%2FQwen3.5-4B")
        self.assertIn("TeichAI_Qwen3.5-4B", keys)

        candidates = self.ns["get_cache_path_candidates"](
            "TeichAI/Qwen3.5-4B", "gpqa_diamond_zeroshot"
        )
        self.assertIn(
            os.path.join(
                "saved_results",
                "results_TeichAI%2FQwen3.5-4B_gpqa_diamond_zeroshot.json",
            ),
            candidates,
        )
        self.assertIn(
            os.path.join(
                "saved_results",
                "results_TeichAI_Qwen3.5-4B_gpqa_diamond_zeroshot.json",
            ),
            candidates,
        )

    def test_highlighted_rows_use_dark_foreground_text(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("background-color: #dcfce7; color: #111827;", source)
        self.assertIn("background-color: #fee2e2; color: #111827;", source)
        self.assertIn("background-color: #fef3c7; color: #111827;", source)

    def test_get_run_config_auto_disables_chat_template_for_unsafe_benchmark(self):
        config = self.ns["get_run_config"](
            "TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning",
            "gpqa_diamond_zeroshot",
        )
        self.assertTrue(config["apply_chat_template_requested"])
        self.assertFalse(config["apply_chat_template"])
        self.assertEqual(config["chat_template_disabled_reason"], "task_incompatible")
        self.assertIsNone(config["chat_template_kwargs"])

    def test_get_run_config_keeps_chat_template_for_safe_benchmark(self):
        config = self.ns["get_run_config"](
            "TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning",
            "gsm8k",
        )
        self.assertTrue(config["apply_chat_template_requested"])
        self.assertTrue(config["apply_chat_template"])
        self.assertIsNone(config["chat_template_disabled_reason"])
        self.assertEqual(config["chat_template_kwargs"], {"enable_thinking": True})

    def test_summarize_results_aggregates_group_subtasks(self):
        data = {
            "lm_eval": {
                "results": {"mmlu": {"acc,none": 0.25}},
                "group_subtasks": {"mmlu": ["mmlu_math", "mmlu_history"]},
                "n-samples": {
                    "mmlu_math": {"effective": 10},
                    "mmlu_history": {"effective": 20},
                },
            }
        }
        score, total_questions, total_correct = self.ns["summarize_results"](
            "model", "mmlu", data
        )
        self.assertEqual(score, 0.25)
        self.assertEqual(total_questions, 30)
        self.assertEqual(total_correct, 7)

    def test_eval_results_yaml_and_bundle_include_expected_metadata(self):
        entry = {
            "Benchmark": "gpqa_diamond_zeroshot",
            "Score": 0.2828282828,
            "Run Diagnostics": {
                "Comparable": "No",
                "Few-shot": "0",
                "Sampling": "On",
                "Temperature": "1.0",
                "Quantization": "none",
            },
            "Details": {},
        }
        yaml_text = self.ns["build_eval_results_yaml"](
            "TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning",
            [entry],
        )
        self.assertIn("# NOTE: One or more exported runs use non-standard settings.", yaml_text)
        self.assertIn("id: Idavidrein/gpqa", yaml_text)
        self.assertIn("task_id: gpqa_diamond", yaml_text)
        self.assertIn('notes: "acc, 0-shot, temp=1.0, sampling (non-standard)"', yaml_text)

        bundle_bytes = self.ns["build_eval_results_bundle"](
            "TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning",
            [entry],
        )
        with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
            self.assertIn(".eval_results/benchmarks.yml", zf.namelist())
            bundled_yaml = zf.read(".eval_results/benchmarks.yml").decode("utf-8")
        self.assertEqual(bundled_yaml, yaml_text)

    def test_export_plotly_figure_prefers_pio_to_image(self):
        figure = go.Figure(data=[go.Bar(x=["a"], y=[1])])
        config = {"toImageButtonOptions": {"width": 200, "height": 100, "scale": 2}}
        with mock.patch.object(self.ns["pio"], "to_image", return_value=b"primary") as mocked_to_image:
            result = self.ns["export_plotly_figure"](figure, config)
        self.assertEqual(result, b"primary")
        self.assertTrue(mocked_to_image.called)

    def test_export_plotly_figure_falls_back_to_legacy_engine_path(self):
        figure = go.Figure(data=[go.Bar(x=["a"], y=[1])])
        config = {"toImageButtonOptions": {"width": 200, "height": 100, "scale": 2}}
        with mock.patch.object(self.ns["pio"], "to_image", side_effect=TypeError("deprecated path")):
            with mock.patch.object(go.Figure, "to_image", return_value=b"fallback") as mocked_figure_to_image:
                result = self.ns["export_plotly_figure"](figure, config)
        self.assertEqual(result, b"fallback")
        _, kwargs = mocked_figure_to_image.call_args
        self.assertEqual(kwargs["engine"], "kaleido")


class RunLmEvalTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.fake_evaluator = SimpleNamespace(simple_evaluate=self.fake_simple_evaluate)
        self.fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        self.ns = load_runner_namespace(self.fake_evaluator, self.fake_torch)

    def fake_simple_evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": {}}

    def test_task_supports_chat_template_matches_expected_policy(self):
        self.assertFalse(self.ns["task_supports_chat_template"]("gpqa_diamond_zeroshot"))
        self.assertFalse(self.ns["task_supports_chat_template"]("mmlu"))
        self.assertTrue(self.ns["task_supports_chat_template"]("ifeval"))
        self.assertTrue(self.ns["task_supports_chat_template"]("humaneval"))

    def test_run_lm_eval_disables_chat_template_for_unsafe_task(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False):
            self.ns["run_lm_eval"](
                "TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning",
                tasks_list=["gpqa_diamond_zeroshot"],
                apply_chat_template=True,
                chat_template_kwargs={"enable_thinking": True},
                backend="hf",
            )
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["tasks"], ["gpqa_diamond_zeroshot"])
        self.assertFalse(call["apply_chat_template"])
        self.assertNotIn("chat_template_args", call["model_args"])
        self.assertNotIn("enable_thinking", call["model_args"])

    def test_run_lm_eval_uses_humaneval_instruct_for_safe_chat_template_task(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False):
            self.ns["run_lm_eval"](
                "TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning",
                tasks_list=["humaneval"],
                apply_chat_template=True,
                chat_template_kwargs={"enable_thinking": True},
                backend="hf",
            )
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["tasks"], ["humaneval_instruct"])
        self.assertTrue(call["apply_chat_template"])
        self.assertEqual(
            call["model_args"]["chat_template_args"], {"enable_thinking": True}
        )
        self.assertTrue(call["model_args"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
