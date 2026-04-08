"""
run_baseline.py - Baseline inference script for CampusAssistantEnv

Usage:
    python run_baseline.py                    # Benchmark all 3 tasks
    python run_baseline.py --task easy        # Run single task
    python run_baseline.py --task hard --llm  # Use LLM (requires OPENAI_API_KEY)
    python run_baseline.py --output results.json

Produces reproducible scores with seed=42 (rule-based mode).
"""

import argparse
import json
import os
import sys

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from agent.baseline_agent import BaselineAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CampusAssistantEnv baseline agent.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help=(
            "Which task to run.\n"
            "  easy   - Summarize DBMS Notes\n"
            "  medium - Prepare for SQL Viva Tomorrow\n"
            "  hard   - Full 2-Hour Study Sprint\n"
            "  all    - Run benchmark on all three tasks (default)"
        ),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Use LLM policy (requires OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model to use (default: gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results as JSON (e.g. results.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress verbose per-step output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    agent = BaselineAgent(
        model=args.model,
        use_llm=args.llm if args.llm else None,
        verbose=not args.quiet,
        seed=args.seed,
    )

    if args.task == "all":
        results = agent.benchmark()
    else:
        results = agent.run(task_difficulty=args.task)
        # Pretty summary for single task
        print(f"\nTask      : {results['task']}")
        print(f"Difficulty: {results['difficulty'].upper()}")
        print(f"Steps     : {results['steps_taken']}")
        print(f"Reward    : {results['final_reward']:.4f}")

    if args.output:
        # Remove non-serialisable keys if any
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
