# Real Options — Engineering Roadmap

**Last updated:** 2026-02-21
**Reviewed by:** Jules

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| Done | Fully implemented and integrated |
| Partial | Module exists but incomplete or not wired in |
| Not started | No implementation exists |

## Items Removed (already done or low value)

| Item | Reason |
|------|--------|
| A.3 Reflexion Expansion | **Done.** All Tier 2 agents use `research_topic_with_reflexion()` with TMS episodic memory retrieval. |
| A.4 Dynamic Agent Weight Self-Tuning | **Done.** `weighted_voting.py` integrates Brier scores via `brier_bridge.py` + regime multipliers. |
| A.5 Automated Trade Journal | **Done.** `trade_journal.py` generates LLM post-mortems, called from `reconciliation.py`. |
| B.2 TMS Temporal Filtering | **Done.** Exponential decay, `valid_from` timestamps, simulation clock for backtests. |
| C.1 Semantic Cache Completion | **Done.** Wired into orchestrator-level council decision flow, sentinel-alert invalidation. |
| C.5 Budget Guard Wiring | **Done.** `record_cost()` called from router on every LLM call. |
| F.4 Neuro-Symbolic Compliance | **Done.** Deterministic pre-checks for confidence thresholds + max positions. |
| E.2 Dynamic Exit Enhancements | **Done.** Per-position P&L exits, DTE-aware acceleration, regime-aware directional exits. |
| A.1 DSPy Prompt Optimization | **Done.** `trading_bot/dspy_optimizer.py` + `scripts/optimize_prompts.py`. |
| G.5 Multi-Commodity | **Done.** MasterOrchestrator manages isolated CommodityEngines. Shared services for Equity/Macro. |
| E.1 Portfolio-Level VaR | **Done.** `trading_bot/var_calculator.py`: Full Revaluation HS VaR (95%/99%), AI Risk Agent (L1/L2), shared state. |
| F.5 Prediction Market Integration | **Done.** `PredictionMarketSentinel` via Gamma API. `TopicDiscoveryAgent` automates topic finding using Claude Haiku. |
| E.3 Liquidity-Aware Execution | **Done.** `order_manager.py:check_liquidity_conditions()` — pre-execution bid/ask depth analysis. |

---

## Prioritized Backlog (optimized for return on capital)

### #1 — C.4 Surrogate Model Activation → C.2 Regime-Switching Inference
**Type:** Cost → P&L | **Effort:** 3-4 weeks (both) | **Status:** Partial (surrogate exists, not wired)

**C.4 (2 weeks):** `backtesting/surrogate_models.py` has a complete GradientBoosting pipeline. Need: wire to live `council_history.csv` for training data, automated weekly retraining.
**C.2 (2 weeks, after C.4):** Route inference by market regime: quiet/range-bound days use surrogate model, crises use full Council.

**Why #1:** Direct bottom-line impact. Regime routing saves 60-80% of LLM costs.

---

### #2 — E.4 Options Greeks Monitor
**Type:** Survival | **Effort:** 2-3 weeks | **Status:** Partial (IV in VaR only)

Real-time portfolio Greeks (Delta, Gamma, Theta, Vega). Need: per-position and portfolio-level Greeks via IB's model, threshold alerts.

**Why #2:** Prevents invisible loss accumulation. Critical with two commodities.

---

### #3 — G.6 3rd Commodity (Sugar SB or Cotton CT)
**Type:** Growth | **Effort:** 1 week | **Status:** Not started (infra ready)

With G.5 (Multi-Commodity) and E.1 (VaR) complete, adding a 3rd commodity is trivial.

**Why #3:** Marginal cost near zero — all infra exists.

---

### #4 — A.2 TextGrad
**Type:** Optimization | **Effort:** 3-4 weeks | **Status:** Not started

Textual backpropagation — Judge LLM generates specific critique for losing trades.

---

### #5 — G.2 A/B Testing Framework
**Type:** Safety | **Effort:** 2-3 weeks | **Status:** Not started

Run DSPy/TextGrad prompt changes as A/B tests on live markets.

---

### #6 — D.5 Reasoning Quality Metrics
**Type:** Insight | **Effort:** 3-4 weeks | **Status:** Not started

Text-Based Information Coefficient, Faithfulness via NLI, Debate Divergence.

---

### #8 — D.1 Event-Driven Backtest Engine
**Type:** Validation | **Effort:** 6-8 weeks | **Status:** Not started

Discrete-event simulation replaying historical data through the full system.

---

## Dependency Graph

```
✅ A.3 Reflexion ──→ ✅ A.1 DSPy ──→ A.2 TextGrad ──→ G.2 A/B Testing
                                    │
✅ A.5 Trade Journal ───────────────┘

✅ C.1 Semantic Cache
✅ C.5 Budget Guard

C.4 Surrogate ──→ C.2 Regime Switching
       │
       └──→ D.1 Backtest Engine

✅ F.4 Neuro-Symbolic ──→ ✅ E.1 Portfolio VaR ──→ E.4 Greeks Monitor

✅ G.5 Multi-Commodity ──→ G.6 3rd Commodity
```
