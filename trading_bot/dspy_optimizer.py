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

# Agents excluded from DSPy optimization (meta-agents whose output is derived
# from other agents, not standalone forecasts)
DSPY_EXCLUDED_AGENTS = {"master_decision"}

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
    """Load labeled examples from enhanced_brier.json + council_history.csv."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._predictions_cache: dict[str, list[dict]] | None = None

    def load(self) -> dict[str, list[dict]]:
        """Return {agent_name: [example_dict]} with resolved predictions only.

        Reads from enhanced_brier.json (source of truth) and enriches with
        market context from council_history.csv and contribution scores.
        Agent names are discovered from the data, not hardcoded.
        Each example_dict has keys: cycle_id, timestamp, direction, confidence,
        prob_bullish, actual, market_context, contribution_score (from
        contribution_scores.json via council_history join).
        """
        if self._predictions_cache is not None:
            return self._predictions_cache

        brier_path = self.data_dir / "enhanced_brier.json"
        council_path = self.data_dir / "council_history.csv"

        if not brier_path.exists():
            raise FileNotFoundError(f"Enhanced Brier data not found: {brier_path}")

        # Load enhanced Brier predictions
        with open(brier_path, "r") as f:
            brier_data = json.load(f)

        # Filter to resolved, non-ORPHANED predictions
        predictions: dict[str, list[dict]] = {}
        skipped_rows = 0
        for pred in brier_data.get("predictions", []):
            try:
                actual = pred.get("actual_outcome")
                if actual is None or actual == "ORPHANED":
                    continue
                agent = pred.get("agent", "").strip()
                if not agent or agent in DSPY_EXCLUDED_AGENTS:
                    continue

                # Derive direction and confidence from probability triple
                prob_bullish = _safe_float(pred.get("prob_bullish"), 1 / 3)
                prob_neutral = _safe_float(pred.get("prob_neutral"), 1 / 3)
                prob_bearish = _safe_float(pred.get("prob_bearish"), 1 / 3)

                probs = {
                    "BULLISH": prob_bullish,
                    "NEUTRAL": prob_neutral,
                    "BEARISH": prob_bearish,
                }
                direction = max(probs, key=probs.get)
                confidence = max(prob_bullish, prob_neutral, prob_bearish)

                predictions.setdefault(agent, []).append({
                    "cycle_id": pred.get("cycle_id", ""),
                    "timestamp": pred.get("timestamp", ""),
                    "direction": direction,
                    "confidence": confidence,
                    "prob_bullish": prob_bullish,
                    "actual": actual,
                    "contribution_score": None,  # Enriched below if available
                })
            except (KeyError, ValueError) as e:
                skipped_rows += 1
                logger.debug(f"Skipping malformed prediction: {e}")
                continue

        if skipped_rows > 0:
            logger.warning(f"Skipped {skipped_rows} malformed predictions in {brier_path}")

        # Load council history for market context and contribution enrichment
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
                            "exit_price": row.get("exit_price", ""),
                            "trigger_type": row.get("trigger_type", ""),
                            "thesis_strength": row.get("thesis_strength", ""),
                            "master_decision": row.get("master_decision", ""),
                            "prediction_type": row.get("prediction_type", "DIRECTIONAL"),
                            "strategy_type": row.get("strategy_type", ""),
                            "volatility_outcome": row.get("volatility_outcome", ""),
                            "actual_trend_direction": row.get("actual_trend_direction", ""),
                            "vote_breakdown": row.get("vote_breakdown", ""),
                        }

        # Build per-agent, per-cycle contribution score lookup from
        # vote_breakdown in council_history (avoids dependency on
        # contribution_scores.json file format).
        contrib_lookup = self._build_contribution_lookup(council_context)

        # Enrich predictions with council context and contribution scores
        for agent, examples in predictions.items():
            for ex in examples:
                ctx = council_context.get(ex["cycle_id"], {})
                ex["market_context"] = (
                    f"Contract: {ctx.get('contract', 'N/A')}, "
                    f"Price: {ctx.get('entry_price', 'N/A')}, "
                    f"Trigger: {ctx.get('trigger_type', 'N/A')}"
                )
                # Attach contribution score if available
                score = contrib_lookup.get((agent, ex["cycle_id"]))
                if score is not None:
                    ex["contribution_score"] = score

        self._predictions_cache = predictions
        return predictions

    def _build_contribution_lookup(
        self, council_context: dict[str, dict]
    ) -> dict[tuple[str, str], float]:
        """Build (agent, cycle_id) → contribution_score lookup.

        Recomputes scores from vote_breakdown + actual outcomes using
        the contribution scorer, so DSPy doesn't depend on whether
        the live hook has run.
        """
        lookup: dict[tuple[str, str], float] = {}
        try:
            from trading_bot.contribution_scorer import ContributionTracker
            from trading_bot.agent_names import normalize_agent_name

            # Load materiality threshold
            try:
                from config.commodity_profiles import get_commodity_profile
                ticker = self.data_dir.name  # e.g. "KC" from data/KC/
                profile = get_commodity_profile(ticker)
                threshold = profile.neutral_move_threshold_pct
            except Exception:
                threshold = 0.008

            # Temporary tracker just for _compute_score
            tracker = ContributionTracker.create_empty("/dev/null")

            for cycle_id, ctx in council_context.items():
                vb_raw = ctx.get("vote_breakdown", "")
                if not vb_raw or vb_raw == "":
                    continue
                try:
                    vote_data = json.loads(vb_raw) if isinstance(vb_raw, str) else vb_raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if not vote_data or not isinstance(vote_data, list):
                    continue

                master_dir = (ctx.get("master_decision") or "NEUTRAL").upper().strip()
                pred_type = (ctx.get("prediction_type") or "DIRECTIONAL").upper().strip()
                strat_type = (ctx.get("strategy_type") or "").upper().strip()
                vol_outcome = (ctx.get("volatility_outcome") or "").upper().strip()
                if vol_outcome in ("NAN", "NONE"):
                    vol_outcome = ""
                if pred_type in ("NAN", "NONE"):
                    pred_type = "DIRECTIONAL"
                if strat_type in ("NAN", "NONE"):
                    strat_type = ""

                # Determine actual outcome with materiality threshold
                raw_trend = (ctx.get("actual_trend_direction") or "").upper().strip()
                if raw_trend in ("NAN", "NONE", ""):
                    continue  # Unresolved cycle
                entry_p = _safe_float(ctx.get("entry_price"), 0)
                exit_p = _safe_float(ctx.get("exit_price"), 0)
                if entry_p > 0 and exit_p > 0:
                    pct_move = abs((exit_p - entry_p) / entry_p)
                    if pct_move < threshold:
                        actual_outcome = "NEUTRAL"
                    else:
                        actual_outcome = raw_trend if raw_trend in ("BULLISH", "BEARISH") else "NEUTRAL"
                else:
                    actual_outcome = raw_trend if raw_trend in ("BULLISH", "BEARISH", "NEUTRAL") else "NEUTRAL"

                for vote in vote_data:
                    agent = vote.get("agent", "")
                    if not agent:
                        continue
                    agent = normalize_agent_name(agent)
                    direction = vote.get("direction", "NEUTRAL")
                    confidence = float(vote.get("confidence", 0.5))
                    weight = float(vote.get("final_weight", 1.0))

                    score = tracker._compute_score(
                        agent_name=agent,
                        agent_direction=direction,
                        agent_confidence=confidence,
                        master_direction=master_dir,
                        actual_outcome=actual_outcome,
                        prediction_type=pred_type,
                        strategy_type=strat_type,
                        volatility_outcome=vol_outcome,
                        influence_weight=weight,
                    )
                    lookup[(agent, cycle_id)] = score

        except ImportError:
            logger.info("Contribution scorer not available, skipping enrichment")
        except Exception as e:
            logger.warning(f"Failed to build contribution lookup: {e}")

        if lookup:
            logger.info(f"Built contribution lookup: {len(lookup)} (agent, cycle) pairs")
        return lookup

    def stats(self) -> dict:
        """Return per-agent counts, class balance, date range."""
        predictions = self.load()
        result = {}
        all_dates = []

        for agent, examples in predictions.items():
            actuals = [ex["actual"] for ex in examples]
            bullish = actuals.count("BULLISH")
            bearish = actuals.count("BEARISH")
            neutral = actuals.count("NEUTRAL")
            total = len(examples)

            # Directional accuracy: only score BULLISH/BEARISH predictions
            directional = [ex for ex in examples if ex["direction"] != "NEUTRAL"]
            directional_correct = sum(
                1 for ex in directional if ex["direction"] == ex["actual"]
            )
            directional_accuracy = (
                directional_correct / len(directional) if directional else 0.0
            )
            neutral_predictions = sum(
                1 for ex in examples if ex["direction"] == "NEUTRAL"
            )
            abstention_rate = neutral_predictions / total if total > 0 else 0.0

            # Brier score: mean squared error of probability vs outcome
            brier = _compute_brier(examples)

            dates = [ex["timestamp"][:10] for ex in examples if ex["timestamp"]]
            all_dates.extend(dates)

            # Contribution-aware metric score (baseline = historical predictions)
            metric_scores = [
                _compute_metric_score(
                    pred_direction=ex["direction"],
                    pred_confidence=ex.get("confidence", 0.5),
                    actual=ex["actual"],
                    contribution_score=ex.get("contribution_score"),
                    example_direction=ex["direction"],
                )
                for ex in examples
            ]
            avg_metric = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
            contrib_count = sum(1 for ex in examples if ex.get("contribution_score") is not None)

            result[agent] = {
                "total": total,
                "directional_total": len(directional),
                "directional_correct": directional_correct,
                "directional_accuracy": directional_accuracy,
                "abstention_rate": abstention_rate,
                "brier_score": brier,
                "avg_metric_score": avg_metric,
                "contribution_coverage": contrib_count / total if total > 0 else 0.0,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "class_balance": _class_balance(actuals),
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
    """Return minority class ratio (0.0-0.5) for the two primary classes.

    Only considers BULLISH and BEARISH for balance calculation since
    NEUTRAL outcomes are rare and shouldn't skew the imbalance metric.
    Returns 0.0 for empty input or single-class data (maximum imbalance).
    """
    if not directions:
        return 0.0
    from collections import Counter
    counts = Counter(directions)
    # Focus on the two primary directional classes
    primary = {k: v for k, v in counts.items() if k in ("BULLISH", "BEARISH")}
    if len(primary) < 2:
        return 0.0
    total = sum(primary.values())
    min_count = min(primary.values())
    return min_count / total


def _compute_metric_score(
    pred_direction: str,
    pred_confidence: float,
    actual: str,
    contribution_score: float | None = None,
    example_direction: str = "",
) -> float:
    """Contribution-aware metric score for a single prediction.

    Used by the DSPy optimizer metric, baseline evaluation, and validation.

    Args:
        pred_direction: Predicted direction (BULLISH/BEARISH/NEUTRAL).
        pred_confidence: Predicted confidence (0.0-1.0).
        actual: Actual outcome (BULLISH/BEARISH/NEUTRAL).
        contribution_score: From contribution scorer (None if unavailable).
        example_direction: Historical direction from training example
            (used to check if prediction matches the known-good/bad call).
    """
    if contribution_score is not None:
        if contribution_score > 0:
            if pred_direction == example_direction:
                return 0.7 + 0.3 * min(1.0, abs(contribution_score))
            return 0.2
        elif contribution_score < 0:
            if pred_direction == example_direction:
                return 0.0
            return 0.5
        else:
            # Zero contribution: actual was NEUTRAL, agent was NEUTRAL, or
            # Master was NEUTRAL. Distinguish "correct abstention on noise"
            # from "missed a real market move."
            if pred_direction == "NEUTRAL":
                if actual == "NEUTRAL":
                    # Market noise — correctly identified nothing was happening
                    return 0.4 + 0.2 * pred_confidence
                else:
                    # Market moved significantly but agent had no opinion.
                    # Better than being actively wrong (0.0) but worse than
                    # committing correctly (0.7+) or identifying noise (0.4+).
                    return 0.15
            else:
                # Agent forced a direction but contribution was still 0
                # (e.g., aligned with Master but outcome was NEUTRAL)
                return 0.25

    # Fallback: directional accuracy + calibration
    if pred_direction == "NEUTRAL":
        return 0.15
    direction_match = 1.0 if pred_direction == actual else 0.0
    actual_hit = 1.0 if pred_direction == actual else 0.0
    calibration = 1.0 - abs(pred_confidence - actual_hit)
    return direction_match * 0.7 + calibration * 0.3


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

    # 2 & 4. Improvement checks (prefer contribution metric, fall back to directional)
    avg_improvement = 0.0
    pct_improved = 0.0
    improvements = []
    using_metric = False
    common_agents = set(baseline.keys()) & set(optimized.keys())
    for agent in common_agents:
        # Use contribution-aware metric when available on both sides
        base_metric = baseline[agent].get("avg_metric_score")
        opt_metric = optimized[agent].get("avg_metric_score")
        if base_metric is not None and opt_metric is not None:
            improvements.append(opt_metric - base_metric)
            using_metric = True
        else:
            base_acc = baseline[agent].get("directional_accuracy", 0)
            opt_acc = optimized[agent].get("directional_accuracy", 0)
            improvements.append(opt_acc - base_acc)

    if improvements:
        avg_improvement = sum(improvements) / len(improvements) * 100
        pct_improved = sum(1 for d in improvements if d > 0) / len(improvements)
        metric_label = "metric" if using_metric else "directional"

        if avg_improvement < MIN_IMPROVEMENT_PCT:
            reasons.append(
                f"Average {metric_label} improvement {avg_improvement:.1f}% < {MIN_IMPROVEMENT_PCT}% threshold"
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

    # Compute baseline directional accuracy from raw examples
    directional_examples = [ex for ex in examples if ex["direction"] != "NEUTRAL"]
    baseline_correct = sum(1 for ex in directional_examples if ex["direction"] == ex["actual"])
    baseline_accuracy = baseline_correct / len(directional_examples) if directional_examples else 0.0

    # Compute baseline contribution metric from raw examples
    baseline_metric_scores = [
        _compute_metric_score(
            pred_direction=ex["direction"],
            pred_confidence=ex.get("confidence", 0.5),
            actual=ex["actual"],
            contribution_score=ex.get("contribution_score"),
            example_direction=ex["direction"],
        )
        for ex in examples
    ]
    baseline_metric = sum(baseline_metric_scores) / len(baseline_metric_scores) if baseline_metric_scores else 0.0

    # Build dspy.Example list
    trainset, valset = _build_splits(examples)

    if len(trainset) < 5:
        logger.warning(f"[{agent_name}] Too few training examples ({len(trainset)}), skipping")
        return {"directional_accuracy": 0.0, "abstention_rate": 0.0, "n_demos": 0, "skipped": True}

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

    # Metric: contribution-aware scoring with directional fallback.
    # Delegates to _compute_metric_score (shared with baseline eval).
    def metric(example, prediction, trace=None):
        try:
            pred_conf = float(prediction.confidence)
        except (ValueError, TypeError, AttributeError):
            pred_conf = 0.5
        return _compute_metric_score(
            pred_direction=prediction.direction,
            pred_confidence=pred_conf,
            actual=example.actual,
            contribution_score=getattr(example, "contribution_score", None),
            example_direction=example.direction,
        )

    # Run BootstrapFewShot (demo counts configurable via config.dspy)
    dspy_cfg = config.get("dspy", {})
    max_bootstrapped = dspy_cfg.get("max_bootstrapped_demos", 4)
    max_labeled = dspy_cfg.get("max_labeled_demos", 4)
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
    )

    module = AgentPredictor()
    compiled = optimizer.compile(module, trainset=trainset)

    # Evaluate on validation set (contribution metric + directional accuracy)
    val_directional = 0
    val_correct = 0
    val_metric_scores = []
    for ex in valset:
        try:
            pred = compiled(market_context=ex.market_context)
            try:
                pred_conf = float(pred.confidence)
            except (ValueError, TypeError, AttributeError):
                pred_conf = 0.5
            val_metric_scores.append(_compute_metric_score(
                pred_direction=pred.direction,
                pred_confidence=pred_conf,
                actual=ex.actual,
                contribution_score=getattr(ex, "contribution_score", None),
                example_direction=ex.direction,
            ))
            if pred.direction != "NEUTRAL":
                val_directional += 1
                if pred.direction == ex.actual:
                    val_correct += 1
        except Exception as e:
            logger.debug(f"[{agent_name}] Eval error: {e}")

    val_accuracy = val_correct / val_directional if val_directional else 0.0
    val_abstention = 1.0 - (val_directional / len(valset)) if valset else 0.0
    val_metric = sum(val_metric_scores) / len(val_metric_scores) if val_metric_scores else 0.0

    # Abstention ceiling: reject optimization if NEUTRAL rate exceeds
    # per-agent ceiling. Different agents have different domain frequencies.
    max_neutral_rates = dspy_cfg.get("max_neutral_rate", {})
    default_ceiling = max_neutral_rates.get("default", 0.60)
    agent_ceiling = max_neutral_rates.get(agent_name, default_ceiling)
    if val_abstention > agent_ceiling:
        logger.warning(
            f"[{agent_name}] Abstention rate {val_abstention:.0%} exceeds "
            f"ceiling {agent_ceiling:.0%} — rejecting optimization"
        )
        return {
            "directional_accuracy": 0.0,
            "avg_metric_score": 0.0,
            "abstention_rate": val_abstention,
            "n_demos": 0,
            "skipped": True,
            "reason": f"abstention {val_abstention:.0%} > ceiling {agent_ceiling:.0%}",
        }

    # Extract optimized instruction and demos.
    # NEUTRAL demos are kept — they're valuable examples of correctly
    # identifying that the agent's domain isn't the market driver.
    instruction = persona_prompt  # BootstrapFewShot selects demos, keeps instruction
    demos = []
    if hasattr(compiled, "predict") and hasattr(compiled.predict, "demos"):
        for demo in compiled.predict.demos:
            direction = getattr(demo, "direction", "")
            demos.append({
                "input": {
                    "market_context": getattr(demo, "market_context", ""),
                },
                "output": {
                    "direction": direction,
                    "confidence": str(getattr(demo, "confidence", "0.5")),
                    "analysis": getattr(demo, "analysis", ""),
                },
            })

    # Save
    result = {
        "baseline_directional_accuracy": baseline_accuracy,
        "baseline_metric": baseline_metric,
        "directional_accuracy": val_accuracy,
        "avg_metric_score": val_metric,
        "abstention_rate": val_abstention,
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
        example_kwargs = {
            "market_context": ex.get("market_context", ""),
            "direction": ex["direction"],
            "confidence": str(ex.get("confidence", 0.5)),
            "actual": ex["actual"],
            "analysis": "",  # Historical analysis text not stored in accuracy CSV
        }
        # Attach contribution score if available (used by metric)
        if ex.get("contribution_score") is not None:
            example_kwargs["contribution_score"] = ex["contribution_score"]
        dspy_examples.append(dspy.Example(**example_kwargs).with_inputs("market_context"))

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
        "baseline_directional_accuracy": result.get("baseline_directional_accuracy", None),
        "baseline_metric": result.get("baseline_metric", None),
        "optimized_directional_accuracy": result.get("directional_accuracy", None),
        "optimized_metric": result.get("avg_metric_score", None),
        "optimized_abstention_rate": result.get("abstention_rate", None),
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
            "directional_accuracy": info["directional_accuracy"],
            "avg_metric_score": info.get("avg_metric_score"),
            "abstention_rate": info["abstention_rate"],
            "brier_score": info["brier_score"],
            "n_examples": info["total"],
            "class_balance": info["class_balance"],
        }
    return result
