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

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

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
            quantization=args.quantization,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
    except Exception as e:
        print(f"LM Eval Failed: {e}")
        logger.error(f"LM Eval Failed: {e}")
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
