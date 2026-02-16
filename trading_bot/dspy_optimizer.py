"""DSPy prompt optimization pipeline for the Trading Council.

Offline-only module: loads historical labeled data, runs BootstrapFewShot
optimization, and exports optimized prompts as plain JSON. No DSPy import
is required at runtime in the trading hot path.

Usage:
    python scripts/optimize_prompts.py               # Evaluate baseline
    python scripts/optimize_prompts.py --optimize     # Run optimization
"""

import csv
import json
import logging
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Readiness thresholds (overridable via config)
DEFAULT_MIN_EXAMPLES_PER_AGENT = 30
DEFAULT_MIN_EXAMPLES_FOR_SUGGEST = 100
MIN_IMPROVEMENT_PCT = 5.0
MIN_CLASS_RATIO = 0.15
MIN_AGENTS_IMPROVED_RATIO = 0.60

# Valid characters for path components (ticker, agent_name)
_SAFE_PATH_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_path_component(value: str, label: str) -> str:
    """Validate that a value is safe to use as a path component."""
    if not _SAFE_PATH_RE.match(value):
        raise ValueError(
            f"Invalid {label} '{value}': must contain only alphanumeric, "
            f"underscore, or hyphen characters"
        )
    return value


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class CouncilDataset:
    """Load labeled examples from council_history.csv + agent_accuracy_structured.csv."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._predictions_cache: dict[str, list[dict]] | None = None

    def load(self) -> dict[str, list[dict]]:
        """Return {agent_name: [example_dict]} with resolved predictions only.

        Agent names are discovered from the data, not hardcoded.
        Each example_dict has keys: cycle_id, timestamp, direction, confidence,
        prob_bullish, actual, market_context (from council_history join).
        """
        if self._predictions_cache is not None:
            return self._predictions_cache

        accuracy_path = self.data_dir / "agent_accuracy_structured.csv"
        council_path = self.data_dir / "council_history.csv"

        if not accuracy_path.exists():
            raise FileNotFoundError(f"Agent accuracy data not found: {accuracy_path}")

        # Load agent predictions, filter resolved
        predictions: dict[str, list[dict]] = {}
        skipped_rows = 0
        with open(accuracy_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    actual = row.get("actual", "PENDING").strip()
                    if actual == "PENDING" or not actual:
                        continue
                    agent = row["agent"].strip()
                    if not agent:
                        continue

                    # Safe float conversion with fallback
                    confidence = _safe_float(row.get("confidence"), 0.5)
                    prob_bullish = _safe_float(row.get("prob_bullish"), 0.5)

                    predictions.setdefault(agent, []).append({
                        "cycle_id": row.get("cycle_id", "").strip(),
                        "timestamp": row.get("timestamp", "").strip(),
                        "direction": row.get("direction", "").strip(),
                        "confidence": confidence,
                        "prob_bullish": prob_bullish,
                        "actual": actual,
                    })
                except (KeyError, ValueError) as e:
                    skipped_rows += 1
                    logger.debug(f"Skipping malformed row {row_num}: {e}")
                    continue

        if skipped_rows > 0:
            logger.warning(f"Skipped {skipped_rows} malformed rows in {accuracy_path}")

        # Load council history for market context (optional enrichment)
        council_context: dict[str, dict] = {}
        if council_path.exists():
            with open(council_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("cycle_id", "").strip()
                    if cid:
                        council_context[cid] = {
                            "contract": row.get("contract", ""),
                            "entry_price": row.get("entry_price", ""),
                            "trigger_type": row.get("trigger_type", ""),
                            "thesis_strength": row.get("thesis_strength", ""),
                            "master_decision": row.get("master_decision", ""),
                        }

        # Enrich predictions with council context
        for agent, examples in predictions.items():
            for ex in examples:
                ctx = council_context.get(ex["cycle_id"], {})
                ex["market_context"] = (
                    f"Contract: {ctx.get('contract', 'N/A')}, "
                    f"Price: {ctx.get('entry_price', 'N/A')}, "
                    f"Trigger: {ctx.get('trigger_type', 'N/A')}"
                )

        self._predictions_cache = predictions
        return predictions

    def stats(self) -> dict:
        """Return per-agent counts, class balance, date range."""
        predictions = self.load()
        result = {}
        all_dates = []

        for agent, examples in predictions.items():
            directions = [ex["actual"] for ex in examples]
            bullish = directions.count("BULLISH")
            bearish = directions.count("BEARISH")
            neutral = directions.count("NEUTRAL")
            total = len(examples)

            correct = sum(
                1 for ex in examples if ex["direction"] == ex["actual"]
            )
            accuracy = correct / total if total > 0 else 0.0

            # Brier score: mean squared error of probability vs outcome
            brier = _compute_brier(examples)

            dates = [ex["timestamp"][:10] for ex in examples if ex["timestamp"]]
            all_dates.extend(dates)

            result[agent] = {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "brier_score": brier,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "class_balance": _class_balance(directions),
            }

        sorted_dates = sorted(set(all_dates))
        return {
            "agents": result,
            "date_range": (sorted_dates[0], sorted_dates[-1]) if sorted_dates else ("N/A", "N/A"),
            "total_resolved": sum(v["total"] for v in result.values()),
        }


def _safe_float(value, default: float) -> float:
    """Convert to float, returning default for None/empty/NaN/non-numeric."""
    if value is None:
        return default
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _compute_brier(examples: list[dict]) -> float:
    """Compute Brier score from examples with prob_bullish vs actual outcome."""
    if not examples:
        return 1.0
    total = 0.0
    valid_count = 0
    for ex in examples:
        prob = ex.get("prob_bullish", 0.5)
        if not isinstance(prob, (int, float)) or math.isnan(prob):
            prob = 0.5
        outcome = 1.0 if ex["actual"] == "BULLISH" else 0.0
        total += (prob - outcome) ** 2
        valid_count += 1
    return total / valid_count if valid_count > 0 else 1.0


def _class_balance(directions: list[str]) -> float:
    """Return minority class ratio (0.0-0.5).

    Returns 0.0 for empty input or single-class data (maximum imbalance).
    """
    if not directions:
        return 0.0
    from collections import Counter
    counts = Counter(directions)
    if len(counts) < 2:
        return 0.0
    total = len(directions)
    min_count = min(counts.values())
    return min_count / total


# ---------------------------------------------------------------------------
# Readiness Check
# ---------------------------------------------------------------------------

def check_readiness(stats: dict, min_examples: int = DEFAULT_MIN_EXAMPLES_PER_AGENT) -> dict:
    """Check if each agent has enough data for optimization.

    Returns: {agent: {ready: bool, reason: str, examples: int}}
    """
    result = {}
    for agent, info in stats.get("agents", {}).items():
        total = info["total"]
        balance = info["class_balance"]

        if total < min_examples:
            result[agent] = {
                "ready": False,
                "reason": f"Need {min_examples} examples, have {total}",
                "examples": total,
            }
        elif balance < MIN_CLASS_RATIO:
            result[agent] = {
                "ready": False,
                "reason": f"Class imbalance too severe ({balance:.0%} minority)",
                "examples": total,
            }
        else:
            result[agent] = {
                "ready": True,
                "reason": f"{total} >= {min_examples}",
                "examples": total,
            }
    return result


def should_suggest_enable(
    baseline: dict[str, dict],
    optimized: dict[str, dict],
    stats: dict,
    min_for_suggest: int = DEFAULT_MIN_EXAMPLES_FOR_SUGGEST,
) -> tuple[bool, str]:
    """After optimization, decide whether to recommend enabling.

    Conditions (ALL must be true):
    1. Every agent has >= min_for_suggest resolved predictions
    2. Optimized accuracy beats baseline by >= MIN_IMPROVEMENT_PCT on average
    3. Class balance: minority class >= MIN_CLASS_RATIO for all agents
    4. Improvement is consistent (>= 60% of agents improved)

    Returns: (should_enable, explanation_string)
    """
    agents_info = stats.get("agents", {})
    reasons = []

    # 1. Data sufficiency
    insufficient = []
    for agent, info in agents_info.items():
        if info["total"] < min_for_suggest:
            insufficient.append(f"{agent}: {info['total']}/{min_for_suggest}")
    if insufficient:
        reasons.append(
            f"Need >= {min_for_suggest} examples/agent. Short: {', '.join(insufficient)}"
        )

    # 2 & 4. Improvement checks
    avg_improvement = 0.0
    pct_improved = 0.0
    improvements = []
    common_agents = set(baseline.keys()) & set(optimized.keys())
    for agent in common_agents:
        base_acc = baseline[agent].get("accuracy", 0)
        opt_acc = optimized[agent].get("accuracy", 0)
        improvements.append(opt_acc - base_acc)

    if improvements:
        avg_improvement = sum(improvements) / len(improvements) * 100
        pct_improved = sum(1 for d in improvements if d > 0) / len(improvements)

        if avg_improvement < MIN_IMPROVEMENT_PCT:
            reasons.append(
                f"Average improvement {avg_improvement:.1f}% < {MIN_IMPROVEMENT_PCT}% threshold"
            )
        if pct_improved < MIN_AGENTS_IMPROVED_RATIO:
            reasons.append(
                f"Only {pct_improved:.0%} of agents improved (need >= {MIN_AGENTS_IMPROVED_RATIO:.0%})"
            )
    else:
        reasons.append("No optimization results to compare")

    # 3. Class balance
    imbalanced = []
    for agent, info in agents_info.items():
        if info["class_balance"] < MIN_CLASS_RATIO:
            imbalanced.append(agent)
    if imbalanced:
        reasons.append(f"Class imbalance too severe for: {', '.join(imbalanced)}")

    if reasons:
        return False, "NOT ready to enable: " + "; ".join(reasons)

    return True, (
        f"RECOMMEND enabling optimized prompts: "
        f"avg improvement {avg_improvement:.1f}%, "
        f"{pct_improved:.0%} of agents improved"
    )


# ---------------------------------------------------------------------------
# DSPy Optimization (only imported when --optimize is used)
# ---------------------------------------------------------------------------

# Lazily-initialized DSPy signature class. Replaced by _ensure_signature()
# before any optimization runs. Do NOT use directly — call
# _ensure_signature() first.
AgentAnalysis = None


def _define_signature():
    """Lazily define the DSPy signature (avoids top-level dspy import)."""
    import dspy

    class _AgentAnalysis(dspy.Signature):
        """Analyze market data for a commodity futures contract and predict direction."""
        persona: str = dspy.InputField(desc="Agent role and domain expertise")
        market_context: str = dspy.InputField(desc="Price, regime, trigger type")
        task: str = dspy.InputField(desc="Specific analysis instruction")
        analysis: str = dspy.OutputField(desc="Evidence-based market analysis")
        direction: str = dspy.OutputField(desc="BULLISH, BEARISH, or NEUTRAL")
        confidence: float = dspy.OutputField(desc="0.0-1.0")

    return _AgentAnalysis


def _ensure_signature():
    """Replace the AgentAnalysis placeholder with a real dspy.Signature."""
    global AgentAnalysis
    AgentAnalysis = _define_signature()


def optimize_agent(
    agent_name: str,
    examples: list[dict],
    config: dict,
    output_dir: str,
    ticker: str = "KC",
) -> dict:
    """Run BootstrapFewShot to find best demonstrations for an agent.

    Uses the agent's production model mapping from config so prompts are
    optimized for the model they'll actually run on.

    Returns: {accuracy, n_demos, instruction, demos}
    """
    import dspy

    # Ensure signature is initialized (idempotent, safe to call multiple times)
    global AgentAnalysis
    if AgentAnalysis is None or not (
        isinstance(AgentAnalysis, type) and issubclass(AgentAnalysis, dspy.Signature)
    ):
        _ensure_signature()

    # Build dspy.Example list
    trainset, valset = _build_splits(examples)

    if len(trainset) < 5:
        logger.warning(f"[{agent_name}] Too few training examples ({len(trainset)}), skipping")
        return {"accuracy": 0.0, "n_demos": 0, "skipped": True}

    # Look up agent's persona prompt from config
    personas = config.get("gemini", {}).get("personas", {})
    persona_prompt = personas.get(agent_name, "You are a helpful market analyst.")

    # Configure DSPy with bootstrap model (configurable, defaults to gpt-4o-mini)
    bootstrap_model = config.get("dspy", {}).get("bootstrap_model", "openai/gpt-4o-mini")
    lm = dspy.LM(bootstrap_model, max_tokens=500)
    dspy.configure(lm=lm)

    # Define the module
    class AgentPredictor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(AgentAnalysis)

        def forward(self, market_context, persona=persona_prompt):
            return self.predict(
                persona=persona,
                market_context=market_context,
                task=f"Analyze as {agent_name}",
            )

    # Metric: direction accuracy (0.7) + calibration (0.3)
    def metric(example, prediction, trace=None):
        direction_match = 1.0 if prediction.direction == example.direction else 0.0
        try:
            pred_conf = float(prediction.confidence)
        except (ValueError, TypeError, AttributeError):
            pred_conf = 0.5
        actual_hit = 1.0 if example.direction == example.actual else 0.0
        calibration = 1.0 - abs(pred_conf - actual_hit)
        return direction_match * 0.7 + calibration * 0.3

    # Run BootstrapFewShot
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )

    module = AgentPredictor()
    compiled = optimizer.compile(module, trainset=trainset)

    # Evaluate on validation set
    val_correct = 0
    for ex in valset:
        try:
            pred = compiled(market_context=ex.market_context)
            if pred.direction == ex.actual:
                val_correct += 1
        except Exception as e:
            logger.debug(f"[{agent_name}] Eval error: {e}")

    val_accuracy = val_correct / len(valset) if valset else 0.0

    # Extract optimized instruction and demos
    instruction = persona_prompt  # BootstrapFewShot selects demos, keeps instruction
    demos = []
    if hasattr(compiled, "predict") and hasattr(compiled.predict, "demos"):
        for demo in compiled.predict.demos:
            demos.append({
                "input": {
                    "market_context": getattr(demo, "market_context", ""),
                },
                "output": {
                    "direction": getattr(demo, "direction", ""),
                    "confidence": str(getattr(demo, "confidence", "0.5")),
                    "analysis": getattr(demo, "analysis", ""),
                },
            })

    # Save
    result = {
        "accuracy": val_accuracy,
        "n_demos": len(demos),
        "instruction": instruction,
        "demos": demos,
    }
    export_optimized_prompt(agent_name, result, output_dir, ticker)
    return result


def _build_splits(examples: list[dict], train_ratio: float = 0.8) -> tuple:
    """Convert raw example dicts into dspy.Example train/val splits."""
    import dspy

    dspy_examples = []
    for ex in examples:
        dspy_examples.append(dspy.Example(
            market_context=ex.get("market_context", ""),
            direction=ex["direction"],
            confidence=str(ex.get("confidence", 0.5)),
            actual=ex["actual"],
            analysis="",  # Historical analysis text not stored in accuracy CSV
        ).with_inputs("market_context"))

    split = int(len(dspy_examples) * train_ratio)
    return dspy_examples[:split], dspy_examples[split:]


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

def export_optimized_prompt(
    agent_name: str,
    result: dict,
    output_dir: str,
    ticker: str = "KC",
):
    """Save optimized instruction + few-shot demos to JSON (atomic write)."""
    _validate_path_component(agent_name, "agent_name")
    _validate_path_component(ticker, "ticker")

    out_path = Path(output_dir) / ticker / agent_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Safety: reject instructions that contain output format directives
    instruction = result.get("instruction", "")
    format_keywords = ["OUTPUT FORMAT", "output as JSON", "FORMAT:", "```json"]
    for kw in format_keywords:
        if kw.lower() in instruction.lower():
            logger.warning(
                f"[{agent_name}] Optimized instruction contains format directive "
                f"'{kw}' — stripping. Format directives belong to the template."
            )
            # Remove the offending line
            lines = instruction.split("\n")
            instruction = "\n".join(
                line for line in lines
                if kw.lower() not in line.lower()
            )

    payload = {
        "agent": agent_name,
        "ticker": ticker,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "n_training_examples": result.get("n_demos", 0),
        "baseline_accuracy": result.get("baseline_accuracy", None),
        "optimized_accuracy": result.get("accuracy", None),
        "instruction": instruction,
        "demos": result.get("demos", []),
    }

    # Atomic write: temp file + os.replace()
    prompt_path = out_path / "prompt.json"
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(out_path), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, str(prompt_path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info(f"[{agent_name}] Saved optimized prompt to {prompt_path}")


def load_optimized_prompt(ticker: str, agent_name: str, prompts_dir: str) -> dict | None:
    """Load an optimized prompt JSON for a given ticker/agent.

    Returns the parsed dict or None if not found / invalid.
    """
    try:
        _validate_path_component(ticker, "ticker")
        _validate_path_component(agent_name, "agent_name")
    except ValueError as e:
        logger.warning(f"Invalid path component: {e}")
        return None

    prompt_path = Path(prompts_dir) / ticker / agent_name / "prompt.json"
    if not prompt_path.exists():
        return None
    try:
        with open(prompt_path, "r") as f:
            data = json.load(f)
        # Validate required key
        if not isinstance(data, dict) or "instruction" not in data:
            logger.warning(f"Optimized prompt {prompt_path} missing 'instruction' key")
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load optimized prompt {prompt_path}: {e}")
        return None


def evaluate_baseline(stats: dict) -> dict[str, dict]:
    """Score current prompts against historical outcomes. No LLM calls.

    Returns: {agent: {accuracy, brier_score, n_examples, class_balance}}
    """
    result = {}
    for agent, info in stats.get("agents", {}).items():
        result[agent] = {
            "accuracy": info["accuracy"],
            "brier_score": info["brier_score"],
            "n_examples": info["total"],
            "class_balance": info["class_balance"],
        }
    return result
