import argparse
import sys
import json
import os
import logging
from benchmarks.run_lm_eval import run_lm_eval
from benchmarks.run_deepeval import run_deepeval

# Setup logging
logging.basicConfig(
    filename="eval.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark models with lm_eval")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["lm_eval"],
        default="lm_eval",
        help="Benchmark framework (currently only lm_eval)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gpqa_diamond_zeroshot",
        help="Comma separated list of tasks for lm_eval (e.g. arc_challenge,gsm8k)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of examples for testing (None = full dataset)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for lm_eval inference (parallel examples per step)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Max context length for vLLM (limits KV cache memory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("saved_results", "results.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization level",
    )
    parser.add_argument(
        "--deepeval", action="store_true", help="Run DeepEval on the results"
    )

    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Override num_fewshot for all tasks (default: use task YAML/default)",
    )
    parser.add_argument(
        "--override_gen_kwargs",
        action="store_true",
        help="Override task generation kwargs with the parameters below",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="When overriding generation kwargs, enable sampling (default: greedy)",
    )

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' for HuggingFace Transformers, 'vllm' for vLLM (Linux only, faster)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for accessing private/gated models",
    )
    parser.add_argument(
        "--allow_code_eval",
        action="store_true",
        help="Allow code execution for code_eval/Humaneval tasks",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template for instruct/chat models",
    )

    args = parser.parse_args()

    results = {}

    # Run lm_eval on the requested tasks
    try:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        results["lm_eval"] = run_lm_eval(
            args.model,
            tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            quantization=args.quantization,
            num_fewshot=args.num_fewshot,
            override_gen_kwargs=args.override_gen_kwargs,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            hf_token=args.hf_token,
            allow_code_eval=args.allow_code_eval,
            apply_chat_template=args.apply_chat_template,
            backend=args.backend,
        )
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"LM Eval Failed: {e}\n{tb}")
        logger.error(f"LM Eval Failed: {e}\n{tb}")
        results["lm_eval"] = str(e)

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                return str(obj)
            except:
                return str(type(obj))

    print("\n=== Final Results ===")
    print(f"Results saved to {args.output}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=CustomEncoder)

    # Run DeepEval if requested
    if args.deepeval:
        try:
            run_deepeval(args.output, limit=args.limit if args.limit else 10)
        except Exception as e:
            print(f"DeepEval Failed: {e}")
            logger.error(f"DeepEval Failed: {e}")


if __name__ == "__main__":
    main()
