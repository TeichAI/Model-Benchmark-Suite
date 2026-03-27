import argparse
import os
import gc
import sys
import torch
from lm_eval import tasks, evaluator, utils
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import clear_torch_cache


CHAT_TEMPLATE_UNSAFE_TASKS = {
    "arc_challenge",
    "gpqa_diamond_zeroshot",
    "hellaswag",
    "mmlu",
    "truthfulqa_mc2",
    "winogrande",
}


def task_supports_chat_template(task_name):
    return task_name not in CHAT_TEMPLATE_UNSAFE_TASKS


def run_lm_eval(
    model_name,
    tasks_list=None,
    limit=None,
    batch_size=1,
    max_model_len=None,
    quantization="none",
    num_fewshot=None,
    override_gen_kwargs=False,
    do_sample=False,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.1,
    hf_token=None,
    allow_code_eval=False,
    apply_chat_template=False,
    chat_template_kwargs=None,
    backend="hf",
):
    if tasks_list is None:
        tasks_list = ["gpqa_diamond_zeroshot"]
    tasks_list = list(tasks_list)
    effective_apply_chat_template = bool(apply_chat_template)
    incompatible_chat_template_tasks = [
        task_name for task_name in tasks_list if not task_supports_chat_template(task_name)
    ]
    if effective_apply_chat_template and incompatible_chat_template_tasks:
        effective_apply_chat_template = False
        print(
            "apply_chat_template=True requested, but it was auto-disabled for "
            f"{incompatible_chat_template_tasks} because assistant-prefill/reasoning prefixes can corrupt multiple-choice likelihood benchmarking."
        )
    chat_template_kwargs = (
        dict(chat_template_kwargs)
        if effective_apply_chat_template and chat_template_kwargs
        else None
    )

    if backend == "vllm" and sys.platform.startswith("win"):
        raise RuntimeError(
            "vLLM backend is not supported on native Windows. Use the hf backend or run under Linux/WSL."
        )

    if effective_apply_chat_template and "humaneval" in tasks_list:
        tasks_list = [
            "humaneval_instruct" if task == "humaneval" else task
            for task in tasks_list
        ]
        print(
            "apply_chat_template=True: using humaneval_instruct for chat/instruct models."
        )
    print(
        f"Running LM Eval on {model_name} with tasks {tasks_list}, "
        f"{quantization} quantization, backend={backend}..."
    )

    # Authenticate with HuggingFace if a token is provided (for private/gated models)
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("Authenticated with HuggingFace token.")
    elif not os.environ.get("HF_TOKEN"):
        # No explicit token provided and HF_TOKEN not set — try to use cached token from `hf auth login`
        from huggingface_hub import get_token

        cached_token = get_token()
        if cached_token:
            os.environ["HF_TOKEN"] = cached_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = cached_token
            print("Using cached HuggingFace token.")

    if allow_code_eval:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    results = None
    try:
        gen_kwargs = None
        if override_gen_kwargs:
            gen_kwargs = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": bool(do_sample),
            }

        if backend == "vllm":
            # vLLM backend: faster inference, Linux/WSL only
            # Auto-detect safe gpu_memory_utilization based on free memory
            gpu_mem_util = 0.8
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                # Use 90% of currently free memory as a fraction of total
                safe_util = round((free_mem * 0.9) / total_mem, 2)
                gpu_mem_util = min(safe_util, 0.9)
                print(
                    f"GPU memory: {free_mem // (1024**2)}MB free / {total_mem // (1024**2)}MB total "
                    f"-> gpu_memory_utilization={gpu_mem_util}"
                )
            if max_model_len is None:
                max_model_len = os.getenv("VLLM_MAX_MODEL_LEN", "8192")
            max_model_len = int(max_model_len)
            model_args = {
                "pretrained": model_name,
                "dtype": "auto",
                "gpu_memory_utilization": gpu_mem_util,
                "trust_remote_code": True,
                "max_model_len": max_model_len,
            }

            # Apply quantization for vLLM (bitsandbytes 4bit/8bit)
            if quantization in ("4bit", "8bit"):
                model_args["quantization"] = "bitsandbytes"
                model_args["load_format"] = "bitsandbytes"

            if chat_template_kwargs:
                model_args["chat_template_args"] = dict(chat_template_kwargs)

            # Avoid embedding tokens in model_args (results JSON stores model_args)
            # Authentication is handled via HF_TOKEN env var above.

            results = evaluator.simple_evaluate(
                model="vllm",
                model_args=model_args,
                tasks=tasks_list,
                num_fewshot=num_fewshot,
                batch_size="auto",
                limit=limit,
                confirm_run_unsafe_code=allow_code_eval,
                apply_chat_template=effective_apply_chat_template,
                gen_kwargs=gen_kwargs,
                log_samples=True,
            )
        else:
            # HuggingFace Transformers backend (default)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if quantization == "4bit":
                model_args = {
                    "pretrained": model_name,
                    "load_in_4bit": True,
                    "trust_remote_code": True,
                }
            elif quantization == "8bit":
                model_args = {
                    "pretrained": model_name,
                    "load_in_8bit": True,
                    "trust_remote_code": True,
                }
            else:
                dtype = "bfloat16" if device == "cuda:0" else "float32"
                model_args = {
                    "pretrained": model_name,
                    "dtype": dtype,
                    "trust_remote_code": True,
                }

            if chat_template_kwargs:
                model_args["chat_template_args"] = dict(chat_template_kwargs)
                model_args["enable_thinking"] = chat_template_kwargs.get(
                    "enable_thinking", False
                )

            # Avoid embedding tokens in model_args (results JSON stores model_args)
            # Authentication is handled via HF_TOKEN env var above.

            if device == "cpu":
                device = "cpu"
                print("CUDA not available, using CPU. This may be slow for large models.")

            results = evaluator.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=tasks_list,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
                device=device,
                confirm_run_unsafe_code=allow_code_eval,
                apply_chat_template=effective_apply_chat_template,
                gen_kwargs=gen_kwargs,
                log_samples=True,
            )

        return results
    finally:
        try:
            clear_torch_cache()
        except Exception:
            pass
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        default="gpqa_diamond_zeroshot",
        help="Comma separated list of tasks",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",")]
    results = run_lm_eval(args.model, task_list, args.limit)
    print(results)
