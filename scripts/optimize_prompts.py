#!/usr/bin/env python3
"""CLI entry point for DSPy prompt optimization.

Usage:
    python scripts/optimize_prompts.py                          # Evaluate baseline
    python scripts/optimize_prompts.py --optimize               # Optimize all ready agents
    python scripts/optimize_prompts.py --optimize --agent macro # Single agent
    python scripts/optimize_prompts.py --ticker CC              # Different commodity
    python scripts/optimize_prompts.py --data-dir /path/to/data # Custom data path
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_bot.dspy_optimizer import (
    CouncilDataset,
    check_readiness,
    evaluate_baseline,
    optimize_agent,
    should_suggest_enable,
    _ensure_signature,
    DEFAULT_MIN_EXAMPLES_PER_AGENT,
    DEFAULT_MIN_EXAMPLES_FOR_SUGGEST,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}")
        sys.exit(1)


def print_eval_table(stats: dict, readiness: dict, min_examples: int):
    """Print formatted evaluation table."""
    agents_info = stats.get("agents", {})
    date_range = stats.get("date_range", ("N/A", "N/A"))
    total = stats.get("total_resolved", 0)

    print(f"\n{'='*56}")
    print(f"  DSPy Prompt Evaluation")
    print(f"{'='*56}")
    print(f"Data: {total} resolved predictions")
    print(f"Date range: {date_range[0]} to {date_range[1]}")

    # Overall class balance
    total_bullish = sum(v.get("bullish", 0) for v in agents_info.values())
    total_bearish = sum(v.get("bearish", 0) for v in agents_info.values())
    total_neutral = sum(v.get("neutral", 0) for v in agents_info.values())
    total_all = total_bullish + total_bearish + total_neutral
    if total_all > 0:
        print(f"Class balance: {total_bearish/total_all:.0%} BEARISH, "
              f"{total_bullish/total_all:.0%} BULLISH, "
              f"{total_neutral/total_all:.0%} NEUTRAL")
    print()

    # Header
    fmt = "{:<20} {:>8} {:>9} {:>7}  {}"
    print(fmt.format("Agent", "Examples", "Accuracy", "Brier", "Ready?"))
    print("-" * 70)

    # Sort by accuracy ascending (worst first)
    sorted_agents = sorted(
        agents_info.items(),
        key=lambda x: x[1].get("accuracy", 0),
    )

    for agent, info in sorted_agents:
        ready_info = readiness.get(agent, {})
        ready_str = "YES" if ready_info.get("ready") else "NO"
        reason = ready_info.get("reason", "")
        flag = ""
        if info["accuracy"] < 0.30:
            flag = " <-- worst"

        print(fmt.format(
            agent,
            info["total"],
            f"{info['accuracy']:.1%}",
            f"{info['brier_score']:.2f}",
            f"{ready_str} ({reason}){flag}",
        ))

    print()
    print("Run with --optimize to generate improved prompts.")


def print_optimization_results(
    baseline: dict,
    optimized_results: dict,
    ticker: str,
    output_dir: str,
    stats: dict,
    min_for_suggest: int,
):
    """Print before/after comparison and recommendation."""
    print(f"\n{'='*56}")
    print(f"  Optimization Results ({ticker})")
    print(f"{'='*56}\n")

    fmt = "{:<20} {:>8} {:>8} {:>10}  {:>5}"
    print(fmt.format("Agent", "Before", "After", "Delta", "Demos"))
    print("-" * 60)

    for agent in sorted(optimized_results.keys()):
        opt = optimized_results[agent]
        if opt.get("skipped"):
            print(f"{agent:<20} {'skipped':>8}")
            continue

        base_acc = baseline.get(agent, {}).get("accuracy", 0)
        opt_acc = opt.get("accuracy", 0)
        delta = opt_acc - base_acc
        delta_str = f"{delta:+.1%}"

        print(fmt.format(
            agent,
            f"{base_acc:.1%}",
            f"{opt_acc:.1%}",
            delta_str,
            opt.get("n_demos", 0),
        ))

    print(f"\nOptimized prompts saved to {output_dir}/{ticker}/")

    # Suggestion
    suggest, explanation = should_suggest_enable(
        baseline, optimized_results, stats, min_for_suggest
    )
    print()
    if suggest:
        print(f"  RECOMMEND enabling optimized prompts:")
        print(f"  {explanation}")
        print(f'  -> Set dspy.use_optimized_prompts.{ticker}: true in config.json')
    else:
        print(f"  {explanation}")


def main():
    parser = argparse.ArgumentParser(description="DSPy prompt optimization for Trading Council")
    parser.add_argument("--optimize", action="store_true", help="Run BootstrapFewShot optimization")
    parser.add_argument("--agent", type=str, help="Optimize a single agent (e.g., macro)")
    parser.add_argument("--ticker", type=str, help="Commodity ticker (default: from config)")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.json"),
                        help="Path to config.json")
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))
    dspy_config = config.get("dspy", {})

    # Resolve paths
    ticker = args.ticker or config.get("symbol", "KC")
    data_dir = args.data_dir or config.get("data_directory", str(PROJECT_ROOT / "data"))
    output_dir = dspy_config.get("optimized_prompts_dir", "data/dspy_optimized")
    if not Path(output_dir).is_absolute():
        output_dir = str(PROJECT_ROOT / output_dir)
    min_examples = dspy_config.get("min_examples_per_agent", DEFAULT_MIN_EXAMPLES_PER_AGENT)
    min_for_suggest = dspy_config.get("min_examples_for_suggest", DEFAULT_MIN_EXAMPLES_FOR_SUGGEST)

    # Load data
    dataset = CouncilDataset(data_dir)
    try:
        predictions = dataset.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure data files exist in: {data_dir}")
        sys.exit(1)

    stats = dataset.stats()
    readiness = check_readiness(stats, min_examples)
    baseline = evaluate_baseline(stats)

    if not args.optimize:
        print_eval_table(stats, readiness, min_examples)
        return

    # Optimization mode — pre-flight checks
    # Check for required API key before starting (avoids N identical failures)
    bootstrap_model = dspy_config.get("bootstrap_model", "openai/gpt-4o-mini")
    if bootstrap_model.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required for --optimize mode.")
        print(f"  Bootstrap model: {bootstrap_model}")
        print("  Set the key or configure dspy.bootstrap_model in config.json.")
        sys.exit(1)

    _ensure_signature()

    agents_to_optimize = []
    if args.agent:
        if args.agent not in predictions:
            print(f"Error: Agent '{args.agent}' not found in data. "
                  f"Available: {', '.join(sorted(predictions.keys()))}")
            sys.exit(1)
        agents_to_optimize = [args.agent]
    else:
        agents_to_optimize = [
            agent for agent, info in readiness.items()
            if info["ready"]
        ]

    if not agents_to_optimize:
        print("No agents ready for optimization. Run without --optimize to see status.")
        sys.exit(0)

    print(f"\nOptimizing {len(agents_to_optimize)} agents for {ticker}...")
    print(f"Output: {output_dir}/{ticker}/\n")

    optimized_results = {}
    for agent in agents_to_optimize:
        print(f"  Optimizing {agent} ({len(predictions[agent])} examples)...")
        try:
            result = optimize_agent(
                agent_name=agent,
                examples=predictions[agent],
                config=config,
                output_dir=output_dir,
                ticker=ticker,
            )
            optimized_results[agent] = result
        except Exception as e:
            logger.error(f"  Failed to optimize {agent}: {e}")
            optimized_results[agent] = {"accuracy": 0.0, "skipped": True}

    print_optimization_results(baseline, optimized_results, ticker, output_dir, stats, min_for_suggest)

    # Exit with error if all agents failed
    if all(r.get("skipped") for r in optimized_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
