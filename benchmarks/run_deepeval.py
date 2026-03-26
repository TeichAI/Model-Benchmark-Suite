import os
import argparse
import json
from deepeval.metrics import AnswerRelevancyMetric
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

    def generate(self, prompt: str, schema=None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if schema is None:
            return content
        if hasattr(schema, "model_validate_json"):
            return schema.model_validate_json(content)
        raise TypeError("Unsupported schema type for DeepEval custom model")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return self.model_name


def _first_non_empty(*values):
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        elif value is not None:
            return value
    return None


def _extract_first_text(value, allow_scalars=True):
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[invalid]":
            return None
        return text

    if allow_scalars and isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        for key in ("content", "text", "response", "output"):
            if key in value:
                text = _extract_first_text(value[key], allow_scalars=allow_scalars)
                if text:
                    return text
        for nested_value in value.values():
            text = _extract_first_text(nested_value, allow_scalars=allow_scalars)
            if text:
                return text
        return None

    if isinstance(value, list):
        for item in value:
            text = _extract_first_text(item, allow_scalars=allow_scalars)
            if text:
                return text

    return None


def _extract_prompt_from_arguments(arguments):
    if not isinstance(arguments, list):
        return None

    for argument in arguments:
        if isinstance(argument, str):
            text = argument.strip()
            if text:
                return text
        elif isinstance(argument, list):
            for element in argument:
                if isinstance(element, str):
                    text = element.strip()
                    if text:
                        return text
                elif isinstance(element, list) and element:
                    text = _extract_first_text(element[0], allow_scalars=False)
                    if text:
                        return text

    return None


def _extract_choices(doc):
    if not isinstance(doc, dict):
        return None

    direct_choices = doc.get("choices")
    if isinstance(direct_choices, list) and direct_choices:
        return [str(choice) for choice in direct_choices]

    indexed_choices = []
    index = 1
    while f"choice{index}" in doc:
        indexed_choices.append(str(doc[f"choice{index}"]))
        index += 1
    if indexed_choices:
        return indexed_choices

    for key in ("mc2_targets", "mc1_targets"):
        target_block = doc.get(key) or {}
        target_choices = target_block.get("choices")
        if isinstance(target_choices, list) and target_choices:
            return [str(choice) for choice in target_choices]

    return None


def _normalize_choice_label(value):
    if not isinstance(value, str):
        return None

    text = value.strip()
    if len(text) == 1 and text.isalpha():
        return text.upper()
    if len(text) == 3 and text.startswith("(") and text.endswith(")") and text[1].isalpha():
        return text[1].upper()
    return None


def _label_to_index(label):
    if not label or not label.isalpha():
        return None
    index = ord(label.upper()) - ord("A")
    if index < 0:
        return None
    return index


def _extract_prediction_index(filtered_resps):
    if not isinstance(filtered_resps, list) or not filtered_resps:
        return None

    scored_candidates = []
    for idx, item in enumerate(filtered_resps):
        score = None
        if isinstance(item, list) and item:
            if isinstance(item[0], (int, float)):
                score = item[0]
            elif isinstance(item[0], list) and item[0] and isinstance(item[0][0], (int, float)):
                score = item[0][0]
        if score is not None:
            scored_candidates.append((idx, score))

    if not scored_candidates:
        return None

    return max(scored_candidates, key=lambda pair: pair[1])[0]


def _extract_mc_outputs(item, doc):
    choices = _extract_choices(doc)
    pred_idx = _extract_prediction_index(item.get("filtered_resps"))

    actual_output = None
    if choices and pred_idx is not None and 0 <= pred_idx < len(choices):
        actual_output = str(choices[pred_idx])
    if actual_output is None:
        actual_output = _extract_first_text(item.get("filtered_resps"), allow_scalars=False)
    if actual_output is None:
        actual_output = _extract_first_text(item.get("resps"), allow_scalars=False)

    expected_output = None
    target_value = item.get("target")
    if choices and isinstance(target_value, int) and 0 <= target_value < len(choices):
        expected_output = str(choices[target_value])
    elif isinstance(target_value, str):
        label = _normalize_choice_label(target_value)
        if choices and label is not None:
            label_index = _label_to_index(label)
            if label_index is not None and 0 <= label_index < len(choices):
                expected_output = str(choices[label_index])
        if expected_output is None:
            expected_output = target_value.strip() or None

    if expected_output is None:
        gold_idx = doc.get("gold")
        if choices and isinstance(gold_idx, int) and 0 <= gold_idx < len(choices):
            expected_output = str(choices[gold_idx])

    if expected_output is None:
        labels = (doc.get("mc2_targets") or {}).get("labels") or (doc.get("mc1_targets") or {}).get("labels")
        if choices and isinstance(labels, list) and 1 in labels:
            gold_idx = labels.index(1)
            if 0 <= gold_idx < len(choices):
                expected_output = str(choices[gold_idx])

    if expected_output is None and isinstance(doc.get("answer"), str):
        answer = doc.get("answer", "").strip()
        label = _normalize_choice_label(answer)
        if choices and label is not None:
            label_index = _label_to_index(label)
            if label_index is not None and 0 <= label_index < len(choices):
                expected_output = str(choices[label_index])
        elif answer:
            expected_output = answer

    return actual_output, expected_output


def _extract_generation_outputs(item, doc):
    actual_output = _extract_first_text(item.get("resps"), allow_scalars=False)
    if actual_output is None:
        actual_output = _extract_first_text(item.get("filtered_resps"), allow_scalars=False)

    expected_output = _extract_first_text(item.get("target"), allow_scalars=False)
    if expected_output is None:
        expected_output = _extract_first_text(doc.get("answer"), allow_scalars=False)
    if expected_output is None:
        expected_output = _extract_first_text(doc.get("solution"), allow_scalars=False)

    return actual_output, expected_output


def _collect_eval_items(data):
    items = []

    if isinstance(data, list):
        for item in data:
            items.append({"task": None, "format": "legacy", "config": {}, "item": item})
        return items

    if not isinstance(data, dict):
        return items

    if "details" in data and isinstance(data["details"], list):
        for item in data["details"]:
            items.append({"task": None, "format": "legacy", "config": {}, "item": item})
        return items

    lm_data = data.get("lm_eval")
    if not isinstance(lm_data, dict):
        return items

    samples_by_task = lm_data.get("samples") or {}
    configs_by_task = lm_data.get("configs") or {}
    if not isinstance(samples_by_task, dict):
        return items

    for task_name, task_items in samples_by_task.items():
        if not isinstance(task_items, list):
            continue
        task_config = configs_by_task.get(task_name) or {}
        for item in task_items:
            items.append(
                {
                    "task": task_name,
                    "format": "lm_eval",
                    "config": task_config,
                    "item": item,
                }
            )

    return items


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

    eval_results = []

    items = _collect_eval_items(data)

    if not items:
        print("Could not find a list of items to evaluate in the results file.")
        return

    print(f"Found {len(items)} items. Evaluating first {limit}...")

    for i, entry in enumerate(items[:limit]):
        item = entry.get("item") or {}
        task_name = entry.get("task")
        item_format = entry.get("format")
        task_config = entry.get("config") or {}
        doc = item.get("doc") or {}

        input_text = None
        actual_output = None
        expected_output = None

        if item_format == "lm_eval":
            input_text = _first_non_empty(
                doc.get("query"),
                doc.get("question"),
                doc.get("Question"),
                doc.get("prompt"),
                doc.get("ctx"),
                _extract_prompt_from_arguments(item.get("arguments")),
            )

            if task_config.get("output_type") == "multiple_choice" or _extract_choices(doc):
                actual_output, expected_output = _extract_mc_outputs(item, doc)
            else:
                actual_output, expected_output = _extract_generation_outputs(item, doc)
        else:
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
                "task": task_name,
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
