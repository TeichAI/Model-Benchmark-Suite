import streamlit as st
import subprocess
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import io
import zipfile
import plotly.io as pio
import tempfile
import time
import signal
import datetime
from collections import deque
from urllib.parse import quote, unquote
from xml.sax.saxutils import escape

PLOTLY_DOWNLOAD_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "height": 800,
        "width": 1400,
        "scale": 2,
    }
}

PLOTLY_MMLU_DOWNLOAD_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "height": 1200,
        "width": 2400,
        "scale": 3,
    }
}

EVAL_RESULTS_FALLBACKS = {
    "arc_challenge": {
        "dataset_id": "allenai/ai2_arc",
        "task_id": "ARC-Challenge",
        "metric_name": "acc_norm",
    },
    "gpqa_diamond_zeroshot": {
        "dataset_id": "Idavidrein/gpqa",
        "task_id": "gpqa_diamond",
        "metric_name": "acc",
    },
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
        "task_id": "main",
        "metric_name": "exact_match",
    },
    "hellaswag": {
        "dataset_id": "Rowan/hellaswag",
        "task_id": "default",
        "metric_name": "acc_norm",
    },
    "humaneval": {
        "dataset_id": "openai/openai_humaneval",
        "task_id": "openai_humaneval",
        "metric_name": "pass@1",
    },
    "ifeval": {
        "dataset_id": "google/IFEval",
        "task_id": "ifeval",
        "metric_name": "prompt_level_strict_acc",
    },
    "mmlu": {
        "dataset_id": "cais/mmlu",
        "task_id": "default",
        "metric_name": "acc",
    },
    "truthfulqa_mc2": {
        "dataset_id": "truthfulqa/truthful_qa",
        "task_id": "multiple_choice",
        "metric_name": "mc2 acc",
    },
    "winogrande": {
        "dataset_id": "allenai/winogrande",
        "task_id": "winogrande_xl",
        "metric_name": "acc",
    },
}

CHAT_TEMPLATE_UNSAFE_BENCHMARKS = {
    "arc_challenge",
    "gpqa_diamond_zeroshot",
    "hellaswag",
    "mmlu",
    "truthfulqa_mc2",
    "winogrande",
}

def parse_json_object_input(value, field_name):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed


def benchmark_supports_chat_template(benchmark):
    return benchmark not in CHAT_TEMPLATE_UNSAFE_BENCHMARKS


def get_effective_chat_template_settings(benchmark):
    requested = bool(apply_chat_template)
    enabled = requested and benchmark_supports_chat_template(benchmark)
    disabled_reason = "task_incompatible" if requested and not enabled else None
    return enabled, disabled_reason

st.set_page_config(page_title="TeichAI Benchmark Suite", layout="wide")

st.title("TeichAI Model Benchmark Suite")

results_placeholder = st.empty()
native_windows = sys.platform.startswith("win")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Model Selection
default_model = "TeichAI/Qwen3-4B-Thinking-2507-Gemini-2.5-Flash-Distill"
models_input = st.sidebar.text_area(
    "Models (one per line)", value=default_model, height=100
)
models = [m.strip() for m in models_input.split("\n") if m.strip()]

# Benchmark (lm_eval task) Selection
benchmarks = st.sidebar.multiselect(
    "Benchmarks",
    [
        "gpqa_diamond_zeroshot",
        "gsm8k",
        "winogrande",
        "arc_challenge",
        "hellaswag",
        "truthfulqa_mc2",
        "mmlu",
        "ifeval",
        "humaneval",
    ],
    default=["gpqa_diamond_zeroshot"],
)

# DeepEval
run_deepeval = st.sidebar.checkbox("Run DeepEval (Qualitative Metrics)", value=False)
if run_deepeval and not os.getenv("OPENROUTER_API_KEY"):
    st.sidebar.warning("OPENROUTER_API_KEY not set. DeepEval may fail.")

# HuggingFace Token (for private models)
with st.sidebar.expander("HuggingFace Token"):
    hf_token = st.text_input(
        "HF Token",
        type="password",
        help="Enter your HuggingFace token to access private/gated models. "
        "Get one at https://huggingface.co/settings/tokens",
    )
    if hf_token:
        st.success("Token provided")

# Settings
backend_options = ["hf"] if native_windows else ["hf", "vllm"]
backend = st.sidebar.selectbox(
    "Inference Backend",
    backend_options,
    index=0,
    help="'hf' = HuggingFace Transformers (works everywhere). "
    "'vllm' = vLLM (Linux/WSL only, much faster for generation tasks like ifeval/humaneval).",
)
if native_windows:
    st.sidebar.info(
        "Native Windows detected: `vllm` is disabled in the UI. Use `hf` or run the suite under Linux/WSL for `vllm`."
    )
quantization = st.sidebar.selectbox(
    "Quantization",
    ["4bit", "8bit", "none"],
    index=2,
    help="Use `none` for the most trustworthy and leaderboard-comparable results. Low-bit quantization is mainly a speed/memory tradeoff.",
)
if quantization != "none":
    st.sidebar.warning(
        "Low-bit quantization can materially change benchmark scores. Use `none` if you care about fair comparison."
    )
vllm_max_model_len_default = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
vllm_max_model_len = st.sidebar.number_input(
    "vLLM Max Context Length",
    min_value=512,
    max_value=262144,
    value=vllm_max_model_len_default,
    step=512,
    disabled=backend != "vllm",
    help="Sets vLLM max_model_len to limit KV cache memory."
    " Only applies to vLLM backend.",
)
allow_code_eval = st.sidebar.checkbox(
    "Allow code execution (Humaneval/code_eval)",
    value=False,
    disabled="humaneval" not in benchmarks,
    help=(
        "Required for Humaneval/code_eval. Runs untrusted model code; "
        "enable only in a sandboxed environment."
    ),
)
apply_chat_template = st.sidebar.checkbox(
    "Apply chat template",
    value=True,
    help="Recommended for instruct/chat models to format prompts correctly.",
)
chat_template_kwargs_input = st.sidebar.text_area(
    "Chat template kwargs (JSON)",
    value="",
    height=80,
    disabled=not apply_chat_template,
    help='Optional JSON object passed into the model chat template, e.g. {"enable_thinking": true}.',
)
chat_template_kwargs = None
chat_template_kwargs_error = None
if apply_chat_template:
    try:
        chat_template_kwargs = parse_json_object_input(
            chat_template_kwargs_input,
            "Chat template kwargs",
        )
    except ValueError as exc:
        chat_template_kwargs_error = str(exc)
        if chat_template_kwargs_input.strip():
            st.sidebar.error(chat_template_kwargs_error)
if apply_chat_template:
    incompatible_chat_template_benchmarks = [
        benchmark for benchmark in benchmarks if not benchmark_supports_chat_template(benchmark)
    ]
    if incompatible_chat_template_benchmarks:
        st.sidebar.warning(
            "Chat template will be auto-disabled for "
            f"{', '.join(incompatible_chat_template_benchmarks)} because assistant-prefill/reasoning prefixes can corrupt multiple-choice likelihood benchmarking."
        )
overwrite_saved = st.sidebar.checkbox("Overwrite saved results", value=False)

fewshot_mode = st.sidebar.selectbox(
    "Few-shot",
    ["Task default (recommended)", "Zero-shot (0)", "Custom"],
    index=0,
)
if fewshot_mode == "Zero-shot (0)":
    num_fewshot = 0
elif fewshot_mode == "Custom":
    num_fewshot = st.sidebar.number_input("num_fewshot", min_value=0, value=5)
else:
    num_fewshot = None

override_gen_kwargs = st.sidebar.checkbox(
    "Override generation settings (advanced)",
    value=False,
)

# Sampling Parameters
with st.sidebar.expander("Sampling Parameters"):
    do_sample = st.checkbox(
        "Enable sampling (not recommended for reproducible benchmarks)",
        value=False,
        disabled=not override_gen_kwargs,
    )
    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.0,
    )
    top_p = st.slider(
        "Top P",
        0.0,
        1.0,
        1.0,
    )
    top_k = st.number_input(
        "Top K",
        value=0,
        min_value=0,
    )
    repetition_penalty = st.slider(
        "Repetition Penalty",
        1.0,
        2.0,
        1.0,
    )
    batch_size = st.number_input("Batch size (lm_eval)", min_value=1, value=1)

if override_gen_kwargs:
    st.sidebar.warning(
        "Manual generation overrides make runs harder to compare across models and across reruns."
    )
if do_sample:
    st.sidebar.error(
        "Sampling is enabled. This run is experimental and should not be treated as a stable benchmark result."
    )

# Run / View Controls
view_saved_only = st.sidebar.checkbox(
    "View saved results only (no new runs)", value=False
)
run_clicked = st.sidebar.button("Run Benchmarks", type="primary")


def is_valid_lm_eval_payload(data):
    if not isinstance(data, dict):
        return False

    lm_data = data.get("lm_eval")
    if not isinstance(lm_data, dict):
        return False

    lm_results = lm_data.get("results")
    return isinstance(lm_results, dict) and bool(lm_results)


def get_model_filename_keys(model):
    model_text = str(model)
    keys = [quote(model_text, safe="")]
    legacy_key = model_text.replace("/", "_")
    if legacy_key not in keys:
        keys.append(legacy_key)
    return keys


def get_cache_path(model, benchmark):
    return os.path.join(
        "saved_results", f"results_{get_model_filename_keys(model)[0]}_{benchmark}.json"
    )


def get_cache_path_candidates(model, benchmark):
    return [
        os.path.join("saved_results", f"results_{key}_{benchmark}.json")
        for key in get_model_filename_keys(model)
    ]


def get_raw_result_path(model, benchmark):
    return os.path.join(
        "saved_results",
        f"results_raw_{get_model_filename_keys(model)[0]}_{benchmark}.json",
    )


def get_deepeval_result_path_candidates(model, benchmark):
    candidates = [
        os.path.join("saved_results", f"results_raw_{key}_{benchmark}_deepeval.json")
        for key in get_model_filename_keys(model)
    ]
    candidates.extend(
        f"results_{key}_{benchmark}_deepeval.json"
        for key in get_model_filename_keys(model)
    )
    return list(dict.fromkeys(candidates))


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def unpack_result_payload(payload):
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("config"), payload.get("data")
    return None, payload


def coerce_scalar(value):
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        try:
            if all(ch not in lowered for ch in [".", "e"]):
                return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text
    return value


def parse_model_args(model_args):
    if isinstance(model_args, dict):
        return dict(model_args)
    if not isinstance(model_args, str):
        return {}

    parsed = {}
    for item in model_args.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = coerce_scalar(value.strip())
    return parsed


def is_probably_chat_model(model_name):
    lowered = str(model_name).lower()
    return any(token in lowered for token in ["instruct", "chat", "assistant", "reasoning"])


def load_saved_run_config(model, benchmark):
    for candidate in get_cache_path_candidates(model, benchmark):
        if not os.path.exists(candidate):
            continue
        try:
            candidate_config, candidate_data = unpack_result_payload(load_json_file(candidate))
        except Exception:
            continue
        if isinstance(candidate_config, dict) and is_valid_lm_eval_payload(candidate_data):
            return candidate_config
    return None


def extract_run_diagnostics(model, benchmark, data, saved_config=None):
    lm_data = data.get("lm_eval", {}) if isinstance(data, dict) else {}
    lm_config = lm_data.get("config", {}) if isinstance(lm_data, dict) else {}
    if not isinstance(lm_config, dict):
        lm_config = {}

    saved_config = saved_config if isinstance(saved_config, dict) else {}
    model_args = parse_model_args(lm_config.get("model_args"))
    gen_kwargs = lm_config.get("gen_kwargs")
    if not isinstance(gen_kwargs, dict):
        gen_kwargs = {}

    backend_name = saved_config.get("backend") or lm_config.get("model") or "-"
    quantization_name = saved_config.get("quantization")
    if not quantization_name:
        if model_args.get("load_in_4bit") is True:
            quantization_name = "4bit"
        elif model_args.get("load_in_8bit") is True:
            quantization_name = "8bit"
        elif model_args.get("quantization") == "bitsandbytes":
            quantization_name = "low-bit"
        else:
            quantization_name = "none"

    override_generation = bool(saved_config.get("override_gen_kwargs")) or bool(gen_kwargs)
    if override_generation:
        sampling_enabled = bool(saved_config.get("do_sample")) if saved_config else bool(gen_kwargs.get("do_sample"))
        temperature_value = saved_config.get("temperature") if saved_config else gen_kwargs.get("temperature")
        top_p_value = saved_config.get("top_p") if saved_config else gen_kwargs.get("top_p")
        top_k_value = saved_config.get("top_k") if saved_config else gen_kwargs.get("top_k")
        repetition_penalty_value = saved_config.get("repetition_penalty") if saved_config else gen_kwargs.get("repetition_penalty")
    else:
        sampling_enabled = False
        temperature_value = None
        top_p_value = None
        top_k_value = None
        repetition_penalty_value = None

    apply_chat_template_value = saved_config.get("apply_chat_template")
    chat_template_disabled_reason = saved_config.get("chat_template_disabled_reason")
    if apply_chat_template_value is None:
        if "chat_template_args" in model_args or "enable_thinking" in model_args:
            apply_chat_template_value = True

    thinking_value = None
    chat_template_kwargs_value = saved_config.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs_value, dict):
        thinking_value = chat_template_kwargs_value.get("enable_thinking")
    if thinking_value is None:
        chat_template_args_value = model_args.get("chat_template_args")
        if isinstance(chat_template_args_value, dict):
            thinking_value = chat_template_args_value.get("enable_thinking")
    if thinking_value is None:
        thinking_value = model_args.get("enable_thinking")

    limit_value = lm_config.get("limit")
    limit_display = "full" if limit_value is None else str(limit_value)
    fewshot_value = saved_config.get("num_fewshot")
    fewshot_display = "task default" if fewshot_value is None else str(fewshot_value)
    batch_size_value = saved_config.get("batch_size", lm_config.get("batch_size", "-"))
    precision_value = model_args.get("dtype") or lm_config.get("model_dtype") or "-"
    device_value = lm_config.get("device") or "-"
    seed_parts = [
        lm_config.get("random_seed"),
        lm_config.get("numpy_seed"),
        lm_config.get("torch_seed"),
    ]
    seed_values = [str(seed) for seed in seed_parts if seed is not None]
    seeds_display = ", ".join(seed_values) if seed_values else "-"

    notes = []
    if limit_value is not None:
        notes.append(f"partial dataset (limit={limit_display})")
    if quantization_name in {"4bit", "8bit", "low-bit"}:
        notes.append(f"{quantization_name} quantization")
    if sampling_enabled:
        notes.append("sampling enabled")
    if fewshot_value is not None:
        notes.append(f"few-shot override={fewshot_display}")
    if (
        apply_chat_template_value is False
        and is_probably_chat_model(model)
        and chat_template_disabled_reason != "task_incompatible"
    ):
        notes.append("chat template disabled")

    comparable = not notes
    if apply_chat_template_value is True:
        chat_template_display = "On"
    elif chat_template_disabled_reason == "task_incompatible":
        chat_template_display = "Auto-disabled"
    elif apply_chat_template_value is False:
        chat_template_display = "Off"
    else:
        chat_template_display = "Unknown"

    if thinking_value is True:
        thinking_display = "On"
    elif thinking_value is False:
        thinking_display = "Off"
    else:
        thinking_display = "-"

    return {
        "Model": model,
        "Benchmark": benchmark,
        "Comparable": "Yes" if comparable else "No",
        "Backend": backend_name,
        "Quantization": quantization_name,
        "Sampling": "On" if sampling_enabled else "Off",
        "Temperature": "-" if temperature_value is None else str(temperature_value),
        "Top P": "-" if top_p_value is None else str(top_p_value),
        "Top K": "-" if top_k_value is None else str(top_k_value),
        "Repetition Penalty": "-" if repetition_penalty_value is None else str(repetition_penalty_value),
        "Few-shot": fewshot_display,
        "Limit": limit_display,
        "Chat Template": chat_template_display,
        "Thinking": thinking_display,
        "Batch Size": str(batch_size_value),
        "Precision": str(precision_value),
        "Device": str(device_value),
        "Seeds": seeds_display,
        "Notes": "; ".join(notes) if notes else "None detected",
    }


def build_result_entry(model, benchmark, data, saved_config=None):
    score, total_questions, total_correct = summarize_results(model, benchmark, data)
    diagnostics = extract_run_diagnostics(model, benchmark, data, saved_config=saved_config)
    return {
        "Model": model,
        "Benchmark": benchmark,
        "Score": score,
        "Total Questions": int(total_questions),
        "Total Correct": int(total_correct),
        "Comparable": diagnostics["Comparable"],
        "Notes": diagnostics["Notes"],
        "Run Config": saved_config,
        "Run Diagnostics": diagnostics,
        "Details": data,
    }


def render_results(results_data):
    if not results_data:
        return

    st.divider()
    st.header("Results Comparison")

    df = pd.DataFrame(results_data)

    all_models = sorted(df["Model"].unique())
    all_benchmarks = sorted(df["Benchmark"].unique())

    selected_models = st.multiselect(
        "Models to display",
        options=all_models,
        default=all_models,
        key="results_filter_models",
    )
    selected_benchmarks = st.multiselect(
        "Benchmarks to display",
        options=all_benchmarks,
        default=all_benchmarks,
        key="results_filter_benchmarks",
    )

    st.session_state.selected_models_for_mmlu = selected_models

    filtered_df = df[
        df["Model"].isin(selected_models) & df["Benchmark"].isin(selected_benchmarks)
    ]

    if filtered_df.empty:
        st.info("No data for current selection.")
        return

    filtered_results = [
        item
        for item in results_data
        if item["Model"] in selected_models and item["Benchmark"] in selected_benchmarks
    ]
    diagnostics_df = pd.DataFrame(
        [item.get("Run Diagnostics", {}) for item in filtered_results]
    )
    provenance_columns = [
        "Model",
        "Benchmark",
        "Comparable",
        "Backend",
        "Quantization",
        "Sampling",
        "Temperature",
        "Few-shot",
        "Limit",
        "Chat Template",
        "Thinking",
        "Batch Size",
        "Precision",
        "Device",
        "Seeds",
        "Notes",
    ]
    provenance_export_df = (
        diagnostics_df[provenance_columns].copy() if not diagnostics_df.empty else pd.DataFrame()
    )
    long_form_df = pd.DataFrame(
        [
            {
                "Model": item["Model"],
                "Benchmark": item["Benchmark"],
                "Score": item["Score"],
                "Total Questions": item["Total Questions"],
                "Total Correct": item["Total Correct"],
                "Comparable": (item.get("Run Diagnostics") or {}).get("Comparable", "Unknown"),
                "Backend": (item.get("Run Diagnostics") or {}).get("Backend", "-"),
                "Quantization": (item.get("Run Diagnostics") or {}).get("Quantization", "-"),
                "Sampling": (item.get("Run Diagnostics") or {}).get("Sampling", "-"),
                "Chat Template": (item.get("Run Diagnostics") or {}).get("Chat Template", "-"),
                "Limit": (item.get("Run Diagnostics") or {}).get("Limit", "-"),
                "Notes": (item.get("Run Diagnostics") or {}).get("Notes", "-"),
            }
            for item in filtered_results
        ]
    )

    if not diagnostics_df.empty:
        displayed_runs = int(len(diagnostics_df))
        comparable_runs = int((diagnostics_df["Comparable"] == "Yes").sum())
        experimental_runs = displayed_runs - comparable_runs
        partial_runs = int((diagnostics_df["Limit"] != "full").sum())
        metric_cols = st.columns(4)
        metric_cols[0].metric("Displayed runs", displayed_runs)
        metric_cols[1].metric("Comparable runs", comparable_runs)
        metric_cols[2].metric("Experimental runs", experimental_runs)
        metric_cols[3].metric("Partial runs", partial_runs)

        if experimental_runs:
            st.warning(
                "Some displayed results were produced under settings that make direct comparison unreliable. Check the run provenance table before trusting rank order."
            )

        st.subheader("Run Provenance")
        st.caption(
            "These fields are extracted from saved run configs and lm_eval output so you can see what was actually run."
        )
        st.dataframe(provenance_export_df, use_container_width=True)

    score_matrix = (
        filtered_df.pivot_table(
            index="Benchmark", columns="Model", values="Score", aggfunc="mean"
        )
        .reindex(columns=selected_models)
        .sort_index()
    )

    def _highlight_row_winners(row):
        valid = row.dropna()
        if valid.empty:
            return ["" for _ in row]
        winner = valid.max()
        return [
            "background-color: #dcfce7; color: #166534; font-weight: 700;"
            if pd.notna(v) and v == winner
            else ""
            for v in row
        ]

    def _highlight_outcome_row(row):
        outcome = row.get("Outcome")
        if outcome == "Primary model ahead":
            style = "background-color: #dcfce7; color: #111827;"
        elif outcome == "Comparison models ahead":
            style = "background-color: #fee2e2; color: #111827;"
        elif outcome == "Tie":
            style = "background-color: #fef3c7; color: #111827;"
        else:
            style = ""
        return [style for _ in row]

    def _highlight_delta_column(values):
        styles = []
        for value in values:
            if pd.isna(value):
                styles.append("")
            elif float(value) > 0:
                styles.append("color: #166534; font-weight: 700;")
            elif float(value) < 0:
                styles.append("color: #991b1b; font-weight: 700;")
            else:
                styles.append("color: #92400e; font-weight: 700;")
        return styles

    def _score_to_text(v):
        return "-" if pd.isna(v) else f"{float(v):.3f}"

    def _score_to_md(v, is_winner):
        if pd.isna(v):
            return "-"
        value = f"{float(v):.3f}"
        return f"**{value}**" if is_winner else value

    def _safe_to_markdown(table_df):
        try:
            return table_df.to_markdown(index=False)
        except ImportError:
            return table_df.to_csv(index=False)

    matrix_md_rows = []
    for benchmark_name, row in score_matrix.iterrows():
        valid = row.dropna()
        winner_score = valid.max() if not valid.empty else None
        row_data = {"Benchmark": benchmark_name}
        for model_name in score_matrix.columns:
            value = row[model_name]
            is_winner = pd.notna(value) and winner_score is not None and value == winner_score
            row_data[model_name] = _score_to_md(value, is_winner)
        matrix_md_rows.append(row_data)

    score_matrix_md_df = pd.DataFrame(matrix_md_rows)

    st.subheader("Head-to-Head Score Matrix")
    st.caption(
        "Rows are benchmarks, columns are models. Best score in each row is highlighted."
    )
    st.dataframe(
        score_matrix.style.format("{:.3f}", na_rep="-").apply(
            _highlight_row_winners, axis=1
        ),
        use_container_width=True,
    )

    with st.expander("View matrix in markdown format"):
        st.markdown(_safe_to_markdown(score_matrix_md_df))

    st.subheader("Head-to-head model comparison")
    st.caption(
        "Choose the primary model first, then add the comparison models you want to measure it against."
    )
    base_model = st.selectbox(
        "Primary model",
        options=selected_models,
        index=0,
        key="base_model_select",
    )
    compare_options = [m for m in selected_models if m != base_model]
    compare_models = st.multiselect(
        "Comparison models",
        options=compare_options,
        default=compare_options,
        key="compare_models_select",
    )

    comparison_summary_df = pd.DataFrame()
    benchmark_outcome_df = pd.DataFrame()
    key_takeaways = []

    if compare_models and base_model in score_matrix.columns:
        summary_rows = []
        for model_name in compare_models:
            if model_name not in score_matrix.columns:
                continue
            pair = score_matrix[[base_model, model_name]].dropna()
            if pair.empty:
                continue
            base_wins = int((pair[base_model] > pair[model_name]).sum())
            base_losses = int((pair[base_model] < pair[model_name]).sum())
            ties = int((pair[base_model] == pair[model_name]).sum())
            avg_delta = float((pair[base_model] - pair[model_name]).mean())
            summary_rows.append(
                {
                    "Comparison Model": model_name,
                    "Benchmarks Compared": int(len(pair)),
                    "Primary Model Wins": base_wins,
                    "Primary Model Losses": base_losses,
                    "Ties": ties,
                    "Avg Score Gap (Primary - Comparison)": avg_delta,
                }
            )

        if summary_rows:
            comparison_summary_df = pd.DataFrame(summary_rows).sort_values(
                "Avg Score Gap (Primary - Comparison)", ascending=False
            )

        outcome_rows = []
        for benchmark_name, row in score_matrix.iterrows():
            base_score = row.get(base_model)
            if pd.isna(base_score):
                continue
            competitor_scores = row[compare_models].dropna()
            if competitor_scores.empty:
                continue
            best_competitor = competitor_scores.idxmax()
            best_competitor_score = float(competitor_scores.max())
            delta = float(base_score - best_competitor_score)
            if delta > 0:
                outcome = "Primary model ahead"
            elif delta < 0:
                outcome = "Comparison models ahead"
            else:
                outcome = "Tie"
            outcome_rows.append(
                {
                    "Benchmark": benchmark_name,
                    "Primary Model Score": float(base_score),
                    "Best Comparison Model": best_competitor,
                    "Best Comparison Score": best_competitor_score,
                    "Score Gap (Primary - Best Comparison)": delta,
                    "Outcome": outcome,
                }
            )

        if outcome_rows:
            benchmark_outcome_df = pd.DataFrame(outcome_rows).sort_values("Benchmark")

        if not benchmark_outcome_df.empty:
            wins = int((benchmark_outcome_df["Outcome"] == "Primary model ahead").sum())
            losses = int((benchmark_outcome_df["Outcome"] == "Comparison models ahead").sum())
            ties = int((benchmark_outcome_df["Outcome"] == "Tie").sum())
            key_takeaways.append(
                f"Primary model ({base_model}): {wins} wins, {losses} losses, {ties} ties across selected benchmarks."
            )
        if not comparison_summary_df.empty:
            strongest = comparison_summary_df.iloc[-1]
            weakest = comparison_summary_df.iloc[0]
            key_takeaways.append(
                f"Hardest comparison model: {strongest['Comparison Model']} (Avg Δ={strongest['Avg Score Gap (Primary - Comparison)']:.3f})."
            )
            key_takeaways.append(
                f"Easiest comparison model: {weakest['Comparison Model']} (Avg Δ={weakest['Avg Score Gap (Primary - Comparison)']:.3f})."
            )

    if not benchmark_outcome_df.empty:
        st.subheader("Benchmark-by-benchmark comparison")
        st.dataframe(
            benchmark_outcome_df.style.format(
                {
                    "Primary Model Score": "{:.3f}",
                    "Best Comparison Score": "{:.3f}",
                    "Score Gap (Primary - Best Comparison)": "{:.3f}",
                }
            ).apply(_highlight_outcome_row, axis=1).apply(
                _highlight_delta_column,
                subset=["Score Gap (Primary - Best Comparison)"],
                axis=0,
            ),
            use_container_width=True,
        )

    if not comparison_summary_df.empty:
        st.subheader("Comparison summary")
        st.dataframe(
            comparison_summary_df.style.format(
                {"Avg Score Gap (Primary - Comparison)": "{:.3f}"}
            ).apply(
                _highlight_delta_column,
                subset=["Avg Score Gap (Primary - Comparison)"],
                axis=0,
            ),
            use_container_width=True,
        )

    if key_takeaways:
        st.subheader("Quick Read")
        for line in key_takeaways:
            st.markdown(f"- {line}")

    # Bar Chart
    fig = px.bar(
        filtered_df,
        x="Model",
        y="Score",
        color="Benchmark",
        barmode="group",
        title="Benchmark Results",
        text_auto=".2f",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1], fixedrange=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=120, l=60, r=60),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="results_bar_chart",
        config=PLOTLY_DOWNLOAD_CONFIG,
    )

    with st.expander("View long-form rows"):
        st.dataframe(
            long_form_df,
            use_container_width=True,
        )

    # Raw Data Expander (full, unfiltered data)
    with st.expander("View Raw Results"):
        st.json(results_data)

    # Export to Markdown / ZIP / PDF (filtered view)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    display_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = "# Benchmark Results Report\n\n"
    md_content += f"**Date:** {display_timestamp}\n\n"
    md_content += "## Scope\n"
    md_content += f"- **Displayed models:** {', '.join(selected_models)}\n"
    md_content += f"- **Displayed benchmarks:** {', '.join(selected_benchmarks)}\n"
    md_content += f"- **Primary comparison model:** {base_model}\n"
    if not diagnostics_df.empty:
        md_content += f"- **Comparable runs:** {comparable_runs}/{displayed_runs}\n"
        md_content += f"- **Experimental runs:** {experimental_runs}\n"
        md_content += f"- **Partial runs:** {partial_runs}\n"
    md_content += "\n"

    if not provenance_export_df.empty:
        md_content += "## Run Provenance\n\n"
        md_content += _safe_to_markdown(provenance_export_df)
        md_content += "\n\n"

    md_content += "## Head-to-Head Score Matrix\n\n"
    md_content += _safe_to_markdown(score_matrix_md_df)

    if not benchmark_outcome_df.empty:
        md_content += "\n\n## Benchmark-by-benchmark comparison\n\n"
        benchmark_outcome_export = benchmark_outcome_df.copy()
        md_content += _safe_to_markdown(benchmark_outcome_export)

    if not comparison_summary_df.empty:
        md_content += "\n\n## Comparison summary\n\n"
        md_content += _safe_to_markdown(comparison_summary_df)

    if key_takeaways:
        md_content += "\n\n## Quick Read\n\n"
        for line in key_takeaways:
            md_content += f"- {line}\n"

    md_content += "\n\n## Full Row Data\n\n"
    md_content += _safe_to_markdown(long_form_df)

    mmlu_filtered = None
    mmlu_subject_results = st.session_state.get("mmlu_subject_results")
    if mmlu_subject_results:
        mmlu_df = pd.DataFrame(mmlu_subject_results)
        mmlu_df["Subject"] = mmlu_df["Benchmark"].apply(
            lambda b: (
                b[len("mmlu_") :] if isinstance(b, str) and b.startswith("mmlu_") else b
            )
        )
        mmlu_filtered = mmlu_df[mmlu_df["Model"].isin(selected_models)]
        if not mmlu_filtered.empty:
            md_content += "\n\n## MMLU Subject Breakdown\n\n"
            md_content += (
                '![alt="MMLU Subject Breakdown"](mmlu_subject_breakdown.png)\n\n'
            )
            mmlu_export = mmlu_filtered[
                [
                    "Model",
                    "Subject",
                    "Benchmark",
                    "Score",
                    "Total Questions",
                    "Total Correct",
                ]
            ].copy()
            try:
                md_mmlu_table = mmlu_export.to_markdown(index=False)
            except ImportError:
                md_mmlu_table = mmlu_export.to_csv(index=False)
            md_content += md_mmlu_table

    # Generate chart images for ZIP/PDF exports
    results_image_bytes = None
    try:
        results_image_bytes = export_plotly_figure(fig, PLOTLY_DOWNLOAD_CONFIG)
    except Exception as e:
        results_image_bytes = None
        st.warning(
            f"Failed to generate main results chart image for PDF/ZIP export: {e}"
        )

    mmlu_image_bytes = None
    if mmlu_filtered is not None and not mmlu_filtered.empty:
        try:
            fig_mmlu_export = px.bar(
                mmlu_filtered,
                x="Subject",
                y="Score",
                color="Model",
                barmode="group",
                title="MMLU Subject Scores",
                text_auto=".2f",
            )
            fig_mmlu_export.update_layout(
                yaxis=dict(range=[0, 1], fixedrange=True),
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(t=80, b=150, l=60, r=60),
            )
            mmlu_image_bytes = export_plotly_figure(
                fig_mmlu_export,
                PLOTLY_MMLU_DOWNLOAD_CONFIG,
            )
        except Exception as e:
            mmlu_image_bytes = None
            st.warning(f"Failed to generate MMLU chart image for PDF/ZIP export: {e}")

    zip_bytes = None
    zip_error = None
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.md", md_content)
            if results_image_bytes is not None:
                zf.writestr("results_bar_chart.png", results_image_bytes)
            if mmlu_image_bytes is not None:
                zf.writestr("mmlu_subject_breakdown.png", mmlu_image_bytes)
        zip_bytes = zip_buffer.getvalue()
    except Exception as e:
        zip_bytes = None
        zip_error = str(e)

    pdf_bytes = None
    pdf_error = None
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Image as RLImage,
            PageBreak,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=landscape(letter),
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TeichTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=colors.HexColor("#2563EB"),
            alignment=1,
            spaceAfter=16,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
        )
        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=body_style,
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            wordWrap="CJK",
        )
        table_cell_style = ParagraphStyle(
            "TableCell",
            parent=body_style,
            fontSize=8,
            leading=10,
            wordWrap="CJK",
        )

        def _build_table(table_df, highlight_winners=False):
            cols = list(table_df.columns)
            data = [[Paragraph(escape(str(col)), table_header_style) for col in cols]]
            for _, row in table_df.iterrows():
                data.append(
                    [Paragraph(escape(str(row[c])), table_cell_style) for c in cols]
                )

            report_table = Table(
                data,
                repeatRows=1,
                splitByRow=1,
                colWidths=[doc.width / max(len(cols), 1)] * len(cols),
            )
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ]

            if highlight_winners and cols and cols[0] == "Benchmark":
                model_cols = cols[1:]
                for row_idx, (_, row) in enumerate(table_df.iterrows(), start=1):
                    numeric_scores = []
                    for model_name in model_cols:
                        try:
                            numeric_scores.append(float(row[model_name]))
                        except Exception:
                            numeric_scores.append(float("nan"))
                    clean_scores = [s for s in numeric_scores if pd.notna(s)]
                    if not clean_scores:
                        continue
                    winner_score = max(clean_scores)
                    for col_idx, score in enumerate(numeric_scores, start=1):
                        if pd.notna(score) and score == winner_score:
                            style_cmds.append(
                                (
                                    "BACKGROUND",
                                    (col_idx, row_idx),
                                    (col_idx, row_idx),
                                    colors.HexColor("#DCFCE7"),
                                )
                            )
                            style_cmds.append(
                                (
                                    "TEXTCOLOR",
                                    (col_idx, row_idx),
                                    (col_idx, row_idx),
                                    colors.HexColor("#166534"),
                                )
                            )

            report_table.setStyle(TableStyle(style_cmds))
            return report_table

        elements = []

        # Title
        elements.append(Paragraph("TeichAI Benchmark Results Report", title_style))

        # Configuration block
        elements.append(Paragraph(f"<b>Date:</b> {display_timestamp}", body_style))
        elements.append(Paragraph(f"<b>Displayed models:</b> {', '.join(selected_models)}", body_style))
        elements.append(Paragraph(f"<b>Displayed benchmarks:</b> {', '.join(selected_benchmarks)}", body_style))
        elements.append(Paragraph(f"<b>Primary comparison model:</b> {base_model}", body_style))
        if not diagnostics_df.empty:
            elements.append(Paragraph(f"<b>Comparable runs:</b> {comparable_runs}/{displayed_runs}", body_style))
            elements.append(Paragraph(f"<b>Experimental runs:</b> {experimental_runs}", body_style))
            elements.append(Paragraph(f"<b>Partial runs:</b> {partial_runs}", body_style))

        if key_takeaways:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Quick Read", heading_style))
            for line in key_takeaways:
                elements.append(Paragraph(f"- {line}", body_style))
        if not provenance_export_df.empty:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Run Provenance", heading_style))
            elements.append(_build_table(provenance_export_df[[
                "Model",
                "Benchmark",
                "Comparable",
                "Backend",
                "Quantization",
                "Sampling",
                "Few-shot",
                "Limit",
                "Chat Template",
                "Notes",
            ]]))

        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Head-to-Head Score Matrix", heading_style))
        matrix_pdf = score_matrix.reset_index().copy()
        for model_name in selected_models:
            if model_name in matrix_pdf.columns:
                matrix_pdf[model_name] = matrix_pdf[model_name].map(_score_to_text)
        elements.append(_build_table(matrix_pdf, highlight_winners=True))

        if not benchmark_outcome_df.empty:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Benchmark-by-benchmark comparison", heading_style))
            outcome_pdf = benchmark_outcome_df.copy()
            for col in [
                "Primary Model Score",
                "Best Comparison Score",
                "Score Gap (Primary - Best Comparison)",
            ]:
                outcome_pdf[col] = outcome_pdf[col].map(lambda v: f"{float(v):.3f}")
            elements.append(_build_table(outcome_pdf))

        if not comparison_summary_df.empty:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Comparison summary", heading_style))
            summary_pdf = comparison_summary_df.copy()
            summary_pdf["Avg Score Gap (Primary - Comparison)"] = summary_pdf[
                "Avg Score Gap (Primary - Comparison)"
            ].map(lambda v: f"{float(v):.3f}")
            elements.append(_build_table(summary_pdf))

        if results_image_bytes is not None:
            elements.append(PageBreak())
            elements.append(Paragraph("Benchmark Results", heading_style))
            elements.append(Spacer(1, 6))
            img_buffer = io.BytesIO(results_image_bytes)
            img = RLImage(img_buffer)
            img._restrictSize(doc.width, doc.height - 100)
            img.hAlign = "CENTER"
            elements.append(img)

        if mmlu_image_bytes is not None:
            elements.append(PageBreak())
            elements.append(Paragraph("MMLU Subject Breakdown", heading_style))
            elements.append(Spacer(1, 6))
            img_buffer = io.BytesIO(mmlu_image_bytes)
            img = RLImage(img_buffer)
            img._restrictSize(doc.width, doc.height - 100)
            img.hAlign = "CENTER"
            elements.append(img)

        doc.build(elements)
        pdf_bytes = pdf_buffer.getvalue()
    except Exception as e:
        pdf_bytes = None
        pdf_error = str(e)

    st.download_button(
        label="Download Clean Markdown Report",
        data=md_content,
        file_name=f"benchmark_report_{timestamp}.md",
        mime="text/markdown",
    )

    if zip_bytes is not None:
        st.download_button(
            label="Download Markdown ZIP (report + charts)",
            data=zip_bytes,
            file_name=f"benchmark_report_{timestamp}.zip",
            mime="application/zip",
        )
    else:
        st.info(
            "ZIP export failed. "
            f"Reason: {zip_error or 'unknown error'}."
        )

    if pdf_bytes is not None:
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"benchmark_report_{timestamp}.pdf",
            mime="application/pdf",
        )
    else:
        st.info(
            "PDF export failed. "
            f"Reason: {pdf_error or 'unknown error'}. "
            "Install/update `reportlab` (and `kaleido` if you want chart images)."
        )

    # --- DeepEval Qualitative Analysis ---
    if run_deepeval:
        st.divider()
        st.header("DeepEval Qualitative Analysis")

        deepeval_data = []
        for item in results_data:
            model = item["Model"]
            benchmark = item["Benchmark"]
            result_file = next(
                (
                    path
                    for path in get_deepeval_result_path_candidates(model, benchmark)
                    if os.path.exists(path)
                ),
                None,
            )

            if not result_file:
                continue

            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    try:
                        eval_results = json.load(f)
                        for res in eval_results:
                            deepeval_data.append(
                                {
                                    "Model": model,
                                    "Benchmark": benchmark,
                                    "Input": res.get("input", ""),
                                    "Score": res.get("score", 0),
                                    "Reason": res.get("reason", ""),
                                }
                            )
                    except json.JSONDecodeError:
                        st.warning(
                            f"Could not parse DeepEval results for {model} on {benchmark}"
                        )

        if deepeval_data:
            df_deep = pd.DataFrame(deepeval_data)

            # Average Score Chart
            avg_scores = (
                df_deep.groupby(["Model", "Benchmark"])["Score"].mean().reset_index()
            )
            fig_deep = px.bar(
                avg_scores,
                x="Model",
                y="Score",
                color="Benchmark",
                title="Average DeepEval Relevancy Score",
                text_auto=".2f",
            )
            fig_deep.update_layout(
                yaxis=dict(range=[0, 1], fixedrange=True),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(t=80, b=120, l=60, r=60),
            )
            st.plotly_chart(
                fig_deep,
                use_container_width=True,
                key="deepeval_bar_chart",
                config=PLOTLY_DOWNLOAD_CONFIG,
            )

            # Detailed Table
            st.subheader("Detailed Qualitative Feedback")
            st.dataframe(df_deep)

    st.subheader("Export for .eval_results")
    eval_results_model = st.selectbox(
        "Model to export as `.eval_results/benchmarks.yml`",
        options=selected_models,
        index=0,
        key="eval_results_model_select",
    )
    eval_results_entries = [
        item for item in filtered_results if item["Model"] == eval_results_model
    ]
    if eval_results_entries:
        st.download_button(
            label="Download `.eval_results/benchmarks.yml` bundle",
            data=build_eval_results_bundle(eval_results_model, eval_results_entries),
            file_name=f"{get_model_filename_keys(eval_results_model)[0]}_eval_results.zip",
            mime="application/zip",
        )
    else:
        st.info("No displayed results are available for the selected `.eval_results` export model.")

def get_task_config(data, benchmark):
    lm_data = data.get("lm_eval", {}) if isinstance(data, dict) else {}
    configs = lm_data.get("configs", {}) if isinstance(lm_data, dict) else {}
    task_config = configs.get(benchmark) if isinstance(configs, dict) else None
    return task_config if isinstance(task_config, dict) else {}

def get_primary_metric_name(data, benchmark):
    lm_data = data.get("lm_eval", {}) if isinstance(data, dict) else {}
    lm_results = lm_data.get("results", {}) if isinstance(lm_data, dict) else {}
    task_metrics = lm_results.get(benchmark, {}) if isinstance(lm_results, dict) else {}
    if not isinstance(task_metrics, dict):
        return None
    metric_keys = [
        "acc,none",
        "acc_norm,none",
        "exact_match,none",
        "prompt_level_strict_acc,none",
        "pass@1,create_test",
        "pass@1,none",
        "pass@1",
    ]
    for key in metric_keys:
        if task_metrics.get(key) is not None:
            return key.split(",", 1)[0]
    return None

def format_score_percent(score):
    return f"{float(score) * 100:.4f}".rstrip("0").rstrip(".")

def build_eval_results_note(result_entry):
    benchmark = result_entry.get("Benchmark")
    diagnostics = result_entry.get("Run Diagnostics") or {}
    fallback = EVAL_RESULTS_FALLBACKS.get(benchmark, {})
    parts = []

    metric_name = get_primary_metric_name(result_entry.get("Details", {}), benchmark) or fallback.get("metric_name")
    if metric_name:
        parts.append(metric_name)

    fewshot_value = diagnostics.get("Few-shot")
    if fewshot_value == "task default":
        parts.append("task-default few-shot")
    elif fewshot_value not in {None, "", "-"}:
        parts.append(f"{fewshot_value}-shot")

    if diagnostics.get("Sampling") == "On":
        temperature_value = diagnostics.get("Temperature")
        if temperature_value not in {None, "", "-"}:
            parts.append(f"temp={temperature_value}")
        parts.append("sampling")

    quantization_value = diagnostics.get("Quantization")
    if quantization_value not in {None, "", "-", "none"}:
        parts.append(f"{quantization_value} quantization")

    note_text = ", ".join(parts) if parts else "exported from Model Benchmark Suite"
    if diagnostics.get("Comparable") != "Yes":
        note_text += " (non-standard)"
    return note_text

def build_eval_results_yaml(model_name, result_entries):
    export_date = datetime.datetime.now().strftime("%Y-%m-%d")
    has_nonstandard = any(
        (entry.get("Run Diagnostics") or {}).get("Comparable") != "Yes"
        for entry in result_entries
    )
    lines = [
        f"# Evaluation results for {model_name}",
        "# Evaluated using lm-evaluation-harness via TeichAI/Model-Benchmark-Suite",
        f"# Date: {export_date}",
    ]
    if has_nonstandard:
        lines.extend(
            [
                "# NOTE: One or more exported runs use non-standard settings.",
                "# Check per-entry notes before comparing these values to leaderboard runs.",
            ]
        )
    lines.append("")

    for entry in sorted(result_entries, key=lambda item: str(item.get("Benchmark", ""))):
        benchmark = entry.get("Benchmark")
        task_config = get_task_config(entry.get("Details", {}), benchmark)
        fallback = EVAL_RESULTS_FALLBACKS.get(benchmark, {})
        dataset_id = task_config.get("dataset_path") or fallback.get("dataset_id") or str(benchmark)
        task_id = task_config.get("dataset_name") or fallback.get("task_id") or str(benchmark)
        lines.extend(
            [
                "- dataset:",
                f"    id: {dataset_id}",
                f"    task_id: {task_id}",
                f"  value: {format_score_percent(entry.get('Score', 0))}",
                f"  date: \"{export_date}\"",
                "  source:",
                "    url: https://github.com/TeichAI/Model-Benchmark-Suite",
                "    name: TeichAI Model Benchmark Suite",
                f"  notes: {json.dumps(build_eval_results_note(entry))}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"

def build_eval_results_bundle(model_name, result_entries):
    zip_buffer = io.BytesIO()
    yaml_content = build_eval_results_yaml(model_name, result_entries)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(".eval_results/benchmarks.yml", yaml_content)
    return zip_buffer.getvalue()

def export_plotly_figure(figure, download_config):
    options = download_config.get("toImageButtonOptions", {})
    width = int(options.get("width", 1400))
    height = int(options.get("height", 800))
    scale = int(options.get("scale", 2))
    export_figure = go.Figure(figure)
    export_figure.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        autosize=False,
        width=width,
        height=height,
    )
    export_figure.update_xaxes(automargin=True)
    export_figure.update_yaxes(automargin=True)
    export_figure.update_traces(textposition="none", cliponaxis=False)
    try:
        return pio.to_image(
            export_figure,
            format="png",
            width=width,
            height=height,
            scale=scale,
        )
    except TypeError:
        return export_figure.to_image(
            format="png",
            width=width,
            height=height,
            scale=scale,
            engine="kaleido",
        )

def render_mmlu_breakdown(mmlu_results):
    if not mmlu_results:
        return

    st.divider()
    st.header("MMLU Subject Breakdown")

    df = pd.DataFrame(mmlu_results)
    df["Subject"] = df["Benchmark"].apply(
        lambda b: (
            b[len("mmlu_") :] if isinstance(b, str) and b.startswith("mmlu_") else b
        )
    )

    global_selected_models = st.session_state.get("selected_models_for_mmlu")
    if isinstance(global_selected_models, list) and global_selected_models:
        df = df[df["Model"].isin(global_selected_models)]

    all_models = sorted(df["Model"].unique())
    all_subjects = sorted(df["Subject"].unique())

    selected_models = st.multiselect(
        "Models to display (MMLU)",
        options=all_models,
        default=all_models,
        key="mmlu_filter_models",
    )
    selected_subjects = st.multiselect(
        "MMLU subjects to display",
        options=all_subjects,
        default=all_subjects,
        key="mmlu_filter_subjects",
    )

    filtered_df = df[
        df["Model"].isin(selected_models) & df["Subject"].isin(selected_subjects)
    ]

    if filtered_df.empty:
        st.info("No MMLU subject data for current selection.")
        return

    fig = px.bar(
        filtered_df,
        x="Subject",
        y="Score",
        color="Model",
        barmode="group",
        title="MMLU Subject Scores",
        text_auto=".2f",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1], fixedrange=True),
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=150, l=60, r=60),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="mmlu_subject_bar_chart",
        config=PLOTLY_MMLU_DOWNLOAD_CONFIG,
    )

    st.dataframe(
        filtered_df[
            [
                "Model",
                "Subject",
                "Benchmark",
                "Score",
                "Total Questions",
                "Total Correct",
            ]
        ]
    )

def summarize_results(model, benchmark, data):
    # Extract score and details
    score = 0
    total_questions = 0
    total_correct = 0

    # lm_eval tasks: benchmark is the lm_eval task name
    lm_data = data.get("lm_eval", {})
    if not isinstance(lm_data, dict):
        return score, total_questions, total_correct

    lm_res = lm_data.get("results", {})
    task_metrics = lm_res.get(benchmark, {})

    if isinstance(task_metrics, dict):
        # Try common metric keys in priority order across different benchmark types:
        # - acc,none / acc_norm,none: standard multiple-choice benchmarks
        # - exact_match,none: gsm8k and similar
        # - prompt_level_strict_acc,none: ifeval
        # - pass@1,none: humaneval (code generation)
        metric_keys = [
            "acc,none",
            "acc_norm,none",
            "exact_match,none",
            "prompt_level_strict_acc,none",
            "pass@1,create_test",
            "pass@1,none",
            "pass@1",
        ]
        task_score = None
        for key in metric_keys:
            task_score = task_metrics.get(key)
            if task_score is not None:
                break
        if task_score is None:
            task_score = 0
        score = task_score

    n_samples_dict = lm_data.get("n-samples", {})
    group_subtasks = lm_data.get("group_subtasks", {})

    def resolve_n_samples(task: str, visited: set | None = None) -> int:
        """Return total number of samples for a task.

        For simple tasks we read lm_eval["n-samples"][task]. For grouped
        tasks like MMLU aggregates (e.g. "mmlu", "mmlu_humanities"), we
        recursively sum over their subtasks from group_subtasks.
        """

        if visited is None:
            visited = set()
        if task in visited:
            return 0
        visited.add(task)

        count_data = n_samples_dict.get(task)
        if isinstance(count_data, dict):
            eff = count_data.get("effective", count_data.get("original", 0))
            if isinstance(eff, (int, float)):
                return int(eff)
        elif isinstance(count_data, (int, float)):
            return int(count_data)

        # Fall back to summing over child tasks if this is a group key
        children = group_subtasks.get(task)
        if isinstance(children, list):
            return sum(resolve_n_samples(child, visited) for child in children)

        return 0

    total_questions = resolve_n_samples(benchmark)
    total_correct = int(score * total_questions) if total_questions else 0

    return score, total_questions, total_correct

def get_run_config(model, benchmark):
    effective_apply_chat_template, chat_template_disabled_reason = (
        get_effective_chat_template_settings(benchmark)
    )
    return {
        "model": model,
        "benchmark": benchmark,
        "backend": backend,
        "quantization": quantization,
        "max_model_len": int(vllm_max_model_len) if backend == "vllm" else None,
        "allow_code_eval": bool(allow_code_eval),
        "apply_chat_template_requested": bool(apply_chat_template),
        "apply_chat_template": effective_apply_chat_template,
        "chat_template_disabled_reason": chat_template_disabled_reason,
        "chat_template_kwargs": chat_template_kwargs if effective_apply_chat_template else None,
        "num_fewshot": None if num_fewshot is None else int(num_fewshot),
        "override_gen_kwargs": bool(override_gen_kwargs),
        "do_sample": bool(do_sample) if override_gen_kwargs else False,
        "temperature": float(temperature) if override_gen_kwargs else None,
        "top_p": float(top_p) if override_gen_kwargs else None,
        "top_k": int(top_k) if override_gen_kwargs else None,
        "repetition_penalty": float(repetition_penalty) if override_gen_kwargs else None,
        "batch_size": int(batch_size),
    }

def load_all_saved_results():
    """Load all lm_eval results from raw JSON files in saved_results/ without running benchmarks."""
    summary_results = []
    mmlu_subject_results = []
    saved_dir = "saved_results"

    if not os.path.isdir(saved_dir):
        return summary_results, mmlu_subject_results

    for fname in os.listdir(saved_dir):
        if not (fname.startswith("results_raw_") and fname.endswith(".json")):
            continue

        path = os.path.join(saved_dir, fname)
        try:
            _, data = unpack_result_payload(load_json_file(path))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        lm_data = data.get("lm_eval", {})
        if not isinstance(lm_data, dict):
            continue

        lm_results = lm_data.get("results", {})
        if not isinstance(lm_results, dict) or not lm_results:
            continue

        for benchmark in lm_results.keys():
            model = None

            configs = lm_data.get("configs", {})
            if isinstance(configs, dict):
                task_cfg = configs.get(benchmark)
                if isinstance(task_cfg, dict):
                    metadata = task_cfg.get("metadata") or {}
                    pretrained = metadata.get("pretrained")
                    if isinstance(pretrained, str) and pretrained:
                        model = pretrained

            if not model:
                base = os.path.splitext(fname)[0]
                prefix = "results_raw_"
                if base.startswith(prefix):
                    core = base[len(prefix) :]
                    suffix = f"_{benchmark}"
                    if core.endswith(suffix):
                        safe_model = core[: -len(suffix)]
                        if safe_model:
                            decoded_model = unquote(safe_model)
                            if decoded_model != safe_model or "%" in safe_model:
                                model = decoded_model
                            else:
                                model = safe_model.replace("_", "/")

            if not model:
                # Skip unresolved models instead of labeling them as "unknown_model".
                continue

            entry = build_result_entry(
                model,
                benchmark,
                data,
                saved_config=load_saved_run_config(model, benchmark),
            )

            if isinstance(benchmark, str) and benchmark.startswith("mmlu_"):
                mmlu_subject_results.append(entry)
            else:
                summary_results.append(entry)

    return summary_results, mmlu_subject_results


# --- Main Execution ---

if view_saved_only:
    summary_results, mmlu_subject_results = load_all_saved_results()
    if not summary_results and not mmlu_subject_results:
        st.info("No saved results found in 'saved_results' directory.")
    else:
        st.session_state.results = summary_results
        st.session_state.mmlu_subject_results = mmlu_subject_results
        with results_placeholder.container():
            if summary_results:
                render_results(summary_results)
            if mmlu_subject_results:
                render_mmlu_breakdown(mmlu_subject_results)

elif run_clicked:
    if not models:
        st.error("Please specify at least one model.")
    elif not benchmarks:
        st.error("Please select at least one benchmark.")
    elif chat_template_kwargs_error:
        st.error(chat_template_kwargs_error)
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_list = []

        total_steps = len(models) * len(benchmarks)
        current_step = 0

        os.makedirs("saved_results", exist_ok=True)

        for model in models:
            for benchmark in benchmarks:
                current_step += 1
                progress = current_step / total_steps
                progress_bar.progress(progress)

                cache_path = get_cache_path(model, benchmark)
                existing_cache_path = next(
                    (
                        path
                        for path in get_cache_path_candidates(model, benchmark)
                        if os.path.exists(path)
                    ),
                    None,
                )

                if existing_cache_path and not overwrite_saved:
                    try:
                        cached_config, data = unpack_result_payload(
                            load_json_file(existing_cache_path)
                        )

                        if not is_valid_lm_eval_payload(data):
                            raise ValueError(
                                "cached file does not contain a valid lm_eval results payload"
                            )

                        current_config = get_run_config(model, benchmark)
                        if not isinstance(cached_config, dict):
                            raise ValueError(
                                "cached file does not contain saved run configuration"
                            )
                        if cached_config != current_config:
                            raise ValueError(
                                "cached run configuration does not match the current request"
                            )

                        status_text.text(
                            f"Using cached results for {benchmark.upper()} on {model}"
                        )

                        results_list.append(
                            build_result_entry(
                                model,
                                benchmark,
                                data,
                                saved_config=cached_config,
                            )
                        )
                        st.session_state.results = results_list
                        continue
                    except Exception as e:
                        st.warning(
                            f"Failed to use cached results for {model} on {benchmark}: {e}. Rerunning."
                        )

                status_text.text(f"Running {benchmark.upper()} on {model}...")

                # Determine path to main.py
                script_path = "main.py" if os.path.exists("main.py") else "eval/main.py"
                if not os.path.exists(script_path):
                    st.error(f"Could not find main.py at {script_path}")
                    continue

                # Construct command: always use lm_eval as framework and this benchmark as the task
                output_file = get_raw_result_path(model, benchmark)
                cmd = [
                    sys.executable,
                    script_path,
                    "--model",
                    model,
                    "--benchmark",
                    "lm_eval",
                    "--tasks",
                    benchmark,
                    "--backend",
                    backend,
                    "--quantization",
                    quantization,
                ]

                if backend == "vllm":
                    cmd.extend(["--max_model_len", str(int(vllm_max_model_len))])

                cmd += [
                    "--batch_size",
                    str(int(batch_size)),
                    "--output",
                    output_file,
                ]

                if num_fewshot is not None:
                    cmd.extend(["--num_fewshot", str(int(num_fewshot))])

                if override_gen_kwargs:
                    cmd.append("--override_gen_kwargs")
                    if do_sample:
                        cmd.append("--do_sample")
                    cmd += [
                        "--temperature",
                        str(temperature),
                        "--top_p",
                        str(top_p),
                        "--top_k",
                        str(top_k),
                        "--repetition_penalty",
                        str(repetition_penalty),
                    ]

                if run_deepeval:
                    cmd.append("--deepeval")

                if hf_token:
                    cmd.extend(["--hf_token", hf_token])

                if allow_code_eval:
                    cmd.append("--allow_code_eval")

                effective_apply_chat_template, _ = get_effective_chat_template_settings(
                    benchmark
                )
                if effective_apply_chat_template:
                    cmd.append("--apply_chat_template")
                    if chat_template_kwargs is not None:
                        cmd.extend(
                            [
                                "--chat_template_kwargs",
                                json.dumps(chat_template_kwargs, sort_keys=True),
                            ]
                        )

                # Run subprocess with real-time logging
                process = None
                log_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w+", suffix=".log", delete=False
                    ) as log_file:
                        log_path = log_file.name

                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                            start_new_session=True,
                        )

                        recent_lines = deque(maxlen=200)
                        latest_line = ""
                        last_ui_update = 0.0
                        line_count = 0

                        with st.spinner(
                            f"Running **{benchmark}** for **{model}**:"):
                            log_placeholder = st.empty()

                            while True:
                                line = process.stdout.readline()
                                if not line and process.poll() is not None:
                                    break
                                if line:
                                    log_file.write(line)
                                    line_count += 1
                                    if line_count % 20 == 0:
                                        log_file.flush()
                                    candidate = line.rstrip()
                                    if candidate:
                                        recent_lines.append(candidate)
                                        latest_line = candidate
                                        now = time.time()
                                        if now - last_ui_update > 0.25:
                                            log_placeholder.code(
                                                latest_line, language="bash"
                                            )
                                            last_ui_update = now

                        # Clear live log line once the run is finished
                        log_placeholder.empty()

                        # Ensure log is flushed
                        log_file.flush()

                    # Join all logs for download (read from temp file)
                    final_logs = ""
                    if log_path and os.path.exists(log_path):
                        with open(log_path, "r", encoding="utf-8") as f:
                            final_logs = f.read()

                    # Clean up temp log file
                    if log_path and os.path.exists(log_path):
                        try:
                            os.remove(log_path)
                        except OSError:
                            pass

                    if process.returncode != 0:
                        st.error(f"Error running {benchmark} on {model}")
                        if recent_lines:
                            st.code("\n".join(recent_lines), language="bash")
                        st.download_button(
                            label="Download Full Logs (Error)",
                            data=final_logs,
                            file_name=f"logs_{get_model_filename_keys(model)[0]}_{benchmark}_error.txt",
                            mime="text/plain",
                        )
                        continue

                    st.success(f"Finished **{benchmark}** for **{model}**:")
                    st.download_button(
                        label="Download Full Logs",
                        data=final_logs,
                        file_name=f"logs_{get_model_filename_keys(model)[0]}_{benchmark}.txt",
                        mime="text/plain",
                    )

                    result_file = output_file
                    if not os.path.exists(result_file):
                        st.error(
                            f"Expected result file {result_file} not found for {model} on {benchmark}"
                        )
                        continue

                    _, data = unpack_result_payload(load_json_file(result_file))

                    if not is_valid_lm_eval_payload(data):
                        st.error(
                            f"Result file {result_file} for {model} on {benchmark} is not a valid lm_eval results payload"
                        )
                        continue

                    # Save a copy into cache with configuration
                    try:
                        current_config = get_run_config(model, benchmark)
                        cache_payload = {
                            "config": current_config,
                            "data": data,
                        }
                        with open(cache_path, "w") as f:
                            json.dump(cache_payload, f)
                    except Exception as e:
                        st.warning(
                            f"Failed to write cached results for {model} on {benchmark}: {e}"
                        )

                    results_list.append(
                        build_result_entry(
                            model,
                            benchmark,
                            data,
                            saved_config=current_config,
                        )
                    )

                    st.session_state.results = results_list

                except Exception as e:
                    st.error(f"Execution failed: {e}")
                    import traceback

                    st.code(traceback.format_exc())
                finally:
                    # Ensure subprocess is terminated and pipes are closed
                    try:
                        if process and process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except Exception:
                                process.terminate()
                            try:
                                process.wait(timeout=10)
                            except Exception:
                                try:
                                    os.killpg(process.pid, signal.SIGKILL)
                                except Exception:
                                    process.kill()
                        if process and process.stdout:
                            process.stdout.close()
                    except Exception:
                        pass

        status_text.text("All benchmarks completed!")
        st.session_state.results = results_list
        with results_placeholder.container():
            render_results(results_list)


# --- Visualization ---
elif "results" in st.session_state and st.session_state.results:
    with results_placeholder.container():
        render_results(st.session_state.results)
        if (
            "mmlu_subject_results" in st.session_state
            and st.session_state.mmlu_subject_results
        ):
            render_mmlu_breakdown(st.session_state.mmlu_subject_results)
