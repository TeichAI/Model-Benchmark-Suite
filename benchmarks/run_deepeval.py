import os
import argparse
import json
from typing import List, Optional
from deepeval.metrics import (
    AnswerRelevancyMetric,
    HallucinationMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI


class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="x-ai/grok-4.1-fast"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def run_deepeval(results_file: str, limit: int = 10):
    print(f"Running DeepEval on {results_file} (Limit: {limit})...")

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        return

    # Load results
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        return

    # Initialize Judge Model
    judge_model = OpenRouterLLM()

    # Initialize Metrics
    # Note: Faithfulness and Hallucination require 'context' which we might not always have in simple QA
    # For now, let's focus on Answer Relevancy if we have the question and answer.
    # If we have context (e.g. from RAG or the prompt itself), we can use Faithfulness.

    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)
    # hallucination_metric = HallucinationMetric(threshold=0.5, model=judge_model) # Needs context

    eval_results = []

    # Determine format of results
    # 1) Legacy formats: list of dicts or dict with "details" list
    # 2) New lm_eval format: dict with "lm_eval" containing per-sample data under "samples"
    items = []
    item_format = "unknown"

    if isinstance(data, list):
        items = data
        item_format = "legacy_list"
    elif isinstance(data, dict):
        # AIME-style legacy format
        if "details" in data and isinstance(data["details"], list):
            items = data["details"]
            item_format = "legacy_details"
        # lm_eval-style results
        elif "lm_eval" in data and isinstance(data["lm_eval"], dict):
            lm = data["lm_eval"]
            samples_by_task = lm.get("samples") or {}

            if isinstance(samples_by_task, dict) and samples_by_task:
                task_name = None

                # Try to infer task name from filename: results_raw_<model>_<task>.json
                base = os.path.basename(results_file)
                if base.startswith("results_raw_") and base.endswith(".json"):
                    core = base[len("results_raw_") : -len(".json")]
                    if "_" in core:
                        # Split once from the right: everything before is model, last part is task
                        _, maybe_task = core.rsplit("_", 1)
                        task_name = maybe_task

                # Fallback: if there is exactly one task in lm_eval.results, use that
                if task_name is None:
                    task_keys = list((lm.get("results") or {}).keys())
                    if len(task_keys) == 1:
                        task_name = task_keys[0]

                if task_name and task_name in samples_by_task:
                    items = samples_by_task[task_name]
                    item_format = "lm_eval_mc"
                else:
                    print(
                        "lm_eval results found but could not determine a task with per-sample 'samples' for DeepEval."
                    )
            else:
                print(
                    "lm_eval results do not contain per-sample 'samples' needed for DeepEval. "
                    "Ensure simple_evaluate was called with log_samples=True."
                )

    if not items:
        print("Could not find a list of items to evaluate in the results file.")
        return

    print(f"Found {len(items)} items. Evaluating first {limit}...")

    for i, item in enumerate(items[:limit]):
        # Adapt to the detected format
        input_text = None
        actual_output = None
        expected_output = None

        if item_format == "lm_eval_mc":
            # lm_eval multiple-choice sample structure
            doc = item.get("doc") or {}

            # 1) Input text: prefer processed query/question
            input_text = doc.get("query") or doc.get("question") or doc.get("ctx")

            # 2) Choices: try common lm_eval MC schemas
            choices = (
                doc.get("choices")
                or (doc.get("mc2_targets") or {}).get("choices")
                or (doc.get("mc1_targets") or {}).get("choices")
            )

            # 3) Predicted choice from filtered_resps (highest log-likelihood)
            filtered_resps = item.get("filtered_resps")
            pred_idx = None
            if isinstance(filtered_resps, list) and filtered_resps:
                scores = []
                for fr in filtered_resps:
                    # Handle [score, flag] or [[score, flag]]
                    if isinstance(fr, list) and fr:
                        if isinstance(fr[0], (int, float)):
                            scores.append(fr[0])
                        elif (
                            isinstance(fr[0], list)
                            and fr[0]
                            and isinstance(fr[0][0], (int, float))
                        ):
                            scores.append(fr[0][0])
                if scores:
                    max_score = max(scores)
                    pred_idx = scores.index(max_score)

            if choices and pred_idx is not None and 0 <= pred_idx < len(choices):
                actual_output = str(choices[pred_idx])

            # 4) Gold answer (optional, for completeness)
            gold_idx = None
            if "gold" in doc:
                gold_idx = doc["gold"]
            else:
                labels = (doc.get("mc2_targets") or {}).get("labels") or (
                    doc.get("mc1_targets") or {}
                ).get("labels")
                if isinstance(labels, list) and 1 in labels:
                    gold_idx = labels.index(1)

            if (
                choices
                and gold_idx is not None
                and isinstance(gold_idx, int)
                and 0 <= gold_idx < len(choices)
            ):
                expected_output = str(choices[gold_idx])

        else:
            # Legacy AIME/GPQA-style formats
            input_text = (
                item.get("problem") or item.get("question") or item.get("prompt")
            )
            actual_output = (
                item.get("generated_answer")
                or item.get("generated_text")
                or item.get("model_patch")
            )
            expected_output = (
                item.get("ground_truth")
                or item.get("correct_answer")
                or item.get("solution")
            )

        if not input_text or not actual_output:
            print(f"Skipping item {i}: Missing input or output.")
            continue

        test_case = LLMTestCase(
            input=input_text,
            actual_output=str(actual_output),
            expected_output=str(expected_output) if expected_output else None,
        )

        print(f"Evaluating item {i+1}...")
        relevancy_metric.measure(test_case)

        eval_results.append(
            {
                "input": input_text[:50] + "...",
                "score": relevancy_metric.score,
                "reason": relevancy_metric.reason,
            }
        )
        print(f"  Score: {relevancy_metric.score} - Reason: {relevancy_metric.reason}")

    # Save DeepEval Results
    output_file = results_file.replace(".json", "_deepeval.json")
    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"DeepEval results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=str, required=True, help="Path to the results JSON file"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of items to evaluate"
    )
    args = parser.parse_args()

    run_deepeval(args.results, args.limit)
