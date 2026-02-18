# Real Options — Engineering Roadmap

**Last updated:** 2026-02-18
**Reviewed by:** Rodrigo + Claude

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
| A.3 Reflexion Expansion | **Done.** All Tier 2 agents use `research_topic_with_reflexion()` with TMS episodic memory retrieval. PRs #816, #818 |
| A.4 Dynamic Agent Weight Self-Tuning | **Done.** `weighted_voting.py` integrates Brier scores via `brier_bridge.py` + regime multipliers + staleness decay |
| A.5 Automated Trade Journal | **Done.** `trade_journal.py` generates LLM post-mortems, called from `reconciliation.py`, stored in TMS |
| B.2 TMS Temporal Filtering | **Done.** Exponential decay, `valid_from` timestamps, simulation clock for backtests in `tms.py` |
| C.1 Semantic Cache Completion | **Done.** Wired into orchestrator-level council decision flow, sentinel-alert invalidation, config-enabled. PRs #812, #816 |
| C.5 Budget Guard Wiring | **Done.** `record_cost()` called from router on every LLM call with actual token counts. Graduated priority throttling active. PR #815 |
| F.4 Neuro-Symbolic Compliance | **Done.** PR #841. Deterministic pre-checks for confidence thresholds + max positions. 63 gates audited, 2 dead checks enforced |
| E.2 Dynamic Exit Enhancements | **Done.** PR #841. Per-position P&L exits, DTE-aware acceleration, regime-aware directional exits. All config-driven, fail-closed |
| A.1 DSPy Prompt Optimization | **Done.** PRs #873, #878. `trading_bot/dspy_optimizer.py` (595 lines) + `scripts/optimize_prompts.py`. BootstrapFewShot pipeline, readiness checks, baseline evaluation. Config-enabled via `dspy.use_optimized_prompts` |
| F.2 Generative Simulacra | Research-grade (6-8 wk), unproven alpha source. Revisit after core profitability validated |
| F.3 Liquid Neural Networks | Research-grade (8-12 wk), needs tick infrastructure that doesn't exist |
| C.3 Knowledge Distillation (SLM) | Model API costs dropping fast. Training custom SLM for 4-6 weeks unlikely to break even |
| B.1 Temporal GraphRAG | Heavy infra (Neo4j, 6-8 wk). TMS with temporal filtering covers 80% of use case |
| D.2 Time-Travel Data Layer | Massive data engineering, depends on D.1 which doesn't exist |
| D.3 Walk-Forward Optimization | Depends on D.1 + D.2 |
| G.8 Notification Expansion | Pushover works for single operator |
| G.3 Advanced Dashboard | Incremental — build pages as modules ship |
| E.5 Market Impact Modeling | Position sizes 1-5 contracts on $50K, impact negligible. Revisit at $500K+ |
| G.7 Automated Incident Post-Mortems | Trade journal (A.5) + self-healing (G.4) cover 80% |
| B.4 Cross-Agent Knowledge Graph | TMS handles knowledge sharing; active cross-cueing unproven need |
| G.5 Multi-Commodity | **Done.** Cocoa (CC) launched Feb 17. Full path isolation across 34 modules, per-commodity systemd services, staggered boot, auto-detect deploy pipeline. PRs #879-#949. KC + CC running simultaneously on DEV. Key infra: `set_data_dir()` pattern across all stateful modules, `COMMODITY_ID_OFFSET` for IB client IDs, `_resolve_data_path()` for dashboard, `migrate_data_dirs.py` for migration, per-commodity deploy verification |

---

## Prioritized Backlog (optimized for return on capital)

### #1 — C.4 Surrogate Model Activation → C.2 Regime-Switching Inference
**Type:** Cost → P&L | **Effort:** 3-4 weeks (both) | **Status:** Partial (surrogate exists, not wired)

**C.4 (2 weeks):** `backtesting/surrogate_models.py` has a complete GradientBoosting pipeline. Need: wire to live `council_history.csv` for training data, automated weekly retraining, model versioning in `data/{ticker}/surrogate_models/`, accuracy tracking via Brier-like metric. Now multi-commodity: train separate models per commodity.

**C.2 (2 weeks, after C.4):** Route inference by market regime: quiet/range-bound days use surrogate model (~$0 per decision), news-driven days use lightweight 2-agent panel, crises use full Council. ~70% of trading days are quiet.

**Why #1:** Direct bottom-line impact. With two commodities running, LLM costs doubled. If LLM costs are $500-1000/month per commodity, regime routing saves 60-80% ($600-1600/month across KC+CC). The surrogate also provides a fast "second opinion" that can flag when the Council is hallucinating.

**Revenue impact:** High — cost savings drop straight to profit. Impact multiplied by number of commodities.

**Prerequisites:** C.4 before C.2.

---

### #2 — E.1 Portfolio-Level VaR
**Type:** Survival | **Effort:** 3-4 weeks | **Status:** Not started

Historical simulation VaR at 95%/99% confidence. Currently only per-trade risk exists in `risk_management.py`. Need portfolio-level aggregation with correlation between positions (including cross-commodity), stress testing against known scenarios (2014 frost, 2020 COVID crash, 2022 Brazil drought), integration with compliance to block new trades on VaR breach.

**Why #2:** Now urgent — G.5 shipped and KC+CC run simultaneously. Coffee and cocoa are correlated (same growing regions, overlapping weather risk, shared logistics). Without portfolio-level VaR, the system can't detect that "long coffee calls + long cocoa calls" doubles exposure to a Brazil weather event. Per-trade checks pass individually but the portfolio is concentrated.

**Revenue impact:** Low direct (prevents losses, not generates returns), but unlocks confidence to size up positions and add more commodities (Sugar, Cotton).

**Prerequisites:** None. Urgency increased now that G.5 is live.

---

### #3 — E.4 Options Greeks Monitor
**Type:** Survival | **Effort:** 2-3 weeks | **Status:** Not started

Real-time portfolio Greeks (Delta, Gamma, Theta, Vega). Options spreads have non-linear risk — a "delta-neutral" straddle becomes directional after a price move. Currently the system only uses delta for strike selection (`strategy.py:find_strike_by_delta`), with no monitoring after entry.

Need: per-position and portfolio-level Greeks via IB's model, threshold alerts (e.g., portfolio delta exceeds ±50), theta-burn warnings, integration with position audit exit cycle. Now multi-commodity: aggregate Greeks across KC+CC positions.

**Why #3:** Prevents invisible loss accumulation. Without Greeks monitoring, the exit cycle can't distinguish between "position is slightly down but structurally fine" and "position has become a naked directional bet due to gamma." Critical with two commodities — more concurrent positions mean more gamma risk.

**Revenue impact:** Medium-high — loss prevention on existing positions.

---

### #4 — A.2 TextGrad
**Type:** Optimization | **Effort:** 3-4 weeks | **Status:** Not started

Textual backpropagation — when a trade loses, a Judge LLM generates specific critique ("the analysis failed to account for X") and suggests prompt edits. Completes the self-learning loop: DSPy (done) optimizes structure, TextGrad optimizes reasoning.

**Why #4:** Now that DSPy is live, TextGrad is the next compounding investment. Every losing trade becomes a training signal. The trade journal (A.5, done) provides the raw material — TextGrad adds the "what should we do differently" layer. With two commodities, the training signal doubles.

**Revenue impact:** Medium — improves win rate over time. Effect compounds.

**Prerequisites:** A.1 (done)

---

### #5 — G.6 3rd Commodity (Sugar SB or Cotton CT)
**Type:** Growth | **Effort:** 1 week | **Status:** Not started

With G.5 infrastructure complete, adding a 3rd commodity is now trivial: create `CommodityProfile`, install systemd service, `mkdir data/SB`. The deploy pipeline auto-detects it. Main work is profile data (growing regions, weather thresholds, logistics hubs, research prompts).

**Why #5:** Marginal cost near zero — all infra exists. Each commodity is a multiplicative revenue lever.

**Revenue impact:** High — new revenue stream for ~1 week of work.

**Prerequisites:** G.5 (done). IBKR market data subscription for the new commodity.

---

### #6 — G.2 A/B Testing Framework
**Type:** Safety | **Effort:** 2-3 weeks | **Status:** Not started

When DSPy/TextGrad propose prompt changes, run them as A/B tests. 50% of cycles use old prompt, 50% new. Statistical significance test determines winner. Auto-promote at p<0.05, auto-reject at p>0.95 (null).

**Why #6:** DSPy is live and generating optimized prompts. Without A/B testing, there's no safe way to validate them on live markets. Currently `dspy.use_optimized_prompts` is a binary config toggle — no gradual rollout, no statistical validation. Multi-commodity doubles the sample size for faster convergence.

**Revenue impact:** Medium — safety gate for the optimization pipeline. Prevents bad prompts from going live.

**Prerequisites:** A.1 (done)

---

### #7 — D.5 Reasoning Quality Metrics
**Type:** Insight | **Effort:** 3-4 weeks | **Status:** Not started

Text-Based Information Coefficient (sentiment signal x confidence vs actual returns), Faithfulness via NLI (are claims supported by retrieved docs?), Debate Divergence (semantic distance between Permabear/Permabull — low = mode collapse).

**Why #7:** Identifies *why* bad decisions happen, not just *that* they happened. Brier scoring (done) tells you accuracy; this tells you reasoning quality. Catches mode collapse (all agents agreeing for wrong reasons) and hallucination (confident claims unsupported by data).

**Revenue impact:** Medium — diagnostic, not directly revenue-generating. But saves weeks of debugging when something goes wrong.

---

### #8 — F.5 Prediction Market Deep Integration
**Type:** Validation | **Effort:** 2-3 weeks | **Status:** Partial (trigger only)

`PredictionMarketSentinel` monitors Polymarket and generates triggers, but integration is one-way. Need: post-Council cross-check (does Council's bullish call align with prediction market odds?), divergence scoring, auto-flag or veto when Council diverges sharply from liquid prediction markets.

**Why #8:** Free second opinion from real-money markets. A Council bullish call with Polymarket pricing in bearish outcomes is a red flag worth catching.

**Revenue impact:** Medium — prevents bad trades when markets already know something the Council missed.

---

### #9 — D.1 Event-Driven Backtest Engine
**Type:** Validation | **Effort:** 6-8 weeks | **Status:** Not started

Discrete-event simulation replaying historical data through the full system. Priority queue architecture, simulated latency, strict temporal isolation. Uses surrogate (C.4) for normal regimes, full Council for crises.

**Why #9:** The only rigorous way to validate strategy changes. Table-stakes for serious trading.

**Prerequisites:** C.4, B.2 (done)

---

### #10 — D.4 Paper Trading Shadow Mode
**Type:** Safety | **Effort:** 3-4 weeks | **Status:** Not started

Run proposed changes in a paper-trading sandbox alongside live system. Both see same data; only live executes. Compare decisions and outcomes. Promotion gate: shadow must outperform live for N days.

**Why #10:** Cheaper, faster validation than full backtesting for incremental changes.

---

### #11 — F.1 Dynamic Agent Recruiting
**Type:** Intelligence | **Effort:** 3-4 weeks | **Status:** Not started

Meta-agent spawns temporary specialists for unusual events (strikes, new regulations, pest outbreaks). Decommissions when event passes. Template library per event category.

---

### #12 — B.3 Agentic RAG for Research
**Type:** Knowledge | **Effort:** 4-5 weeks | **Status:** Not started

Researcher agent monitors preprint servers, USDA/ICO reports. Extracts causal mechanisms and quantitative findings. Stores in TMS as `type: RESEARCH_FINDING`.

**Prerequisites:** B.2 (done)

---

### #13 — E.3 Liquidity-Aware Execution
**Type:** Execution | **Effort:** 2-3 weeks | **Status:** Basic

Pre-execution book depth analysis, VWAP/TWAP for larger orders, thin-market detection. Currently uses static LMT orders with fixed slippage. Matters more as account grows or multi-commodity ships.

---

### #14 — G.1 Formal Observability (AgentOps)
**Type:** Ops | **Effort:** 2-3 weeks | **Status:** Partial (internal only)

Custom `observability.py` exists with HallucinationDetector, AgentTrace, and ObservabilityHub. Missing: third-party integration (AgentOps or LangSmith) for trace replay, cost attribution dashboards, and alert rules. Keep custom hallucination detection as it's commodity-aware.

---

### #15 — F.6 Synthetic Rare Event Generation
**Type:** Stress test | **Effort:** 3-4 weeks | **Status:** Not started

Combine real historical events in novel ways for stress testing. "What if frost + strike + BRL crash simultaneously?" Augments DSPy training set.

**Prerequisites:** D.1

---

## Dependency Graph

```
✅ A.3 Reflexion (done) ──→ ✅ A.1 DSPy (done) ──→ A.2 TextGrad ──→ G.2 A/B Testing
                                    │
✅ A.5 Trade Journal (done) ────────┘

✅ C.1 Semantic Cache (done)
✅ C.5 Budget Guard (done)

C.4 Surrogate ──→ C.2 Regime Switching
       │
       └──→ D.1 Backtest Engine ──→ D.4 Shadow Mode
                    │
                    └──→ F.6 Synthetic Events

✅ F.4 Neuro-Symbolic (done) ──→ E.1 Portfolio VaR ──→ E.4 Greeks Monitor

✅ E.2 Exit Enhancements (done)

✅ G.5 Multi-Commodity (done) ──→ G.6 3rd Commodity
                                      │
                                      └──→ E.1 VaR (urgency ↑)
```

## Recommended Execution Plan

| Phase | Items | Duration | Focus |
|-------|-------|----------|-------|
| ~~Phase 1~~ | ~~A.3, C.1, C.5~~ | ~~3-5 weeks~~ | **Done** |
| ~~Phase 2~~ | ~~F.4, E.2~~ | ~~4-6 weeks~~ | **Done** |
| ~~Phase 3~~ | ~~A.1 DSPy~~ | ~~3-4 weeks~~ | **Done** |
| ~~Phase 4a~~ | ~~G.5 Multi-Commodity~~ | ~~2 weeks~~ | **Done** (CC launched Feb 17) |
| **Phase 4b** | **#1 C.4/C.2 Surrogate+Regime + #2 E.1 VaR** | **4-6 weeks** | **Cost reduction + risk management** |
| Phase 5 | #3 E.4 Greeks + #4 A.2 TextGrad + #5 G.6 3rd Commodity | 4-6 weeks | Risk visibility + growth + decision quality |
| Phase 6 | #6 G.2 A/B Testing + #7 D.5 Reasoning Metrics | 4-6 weeks | Optimization safety + diagnostics |
| Phase 7 | #8-#15 (depth + validation) | 20-30 weeks | Foundation for scale |

**Phase 4b rationale:** Surrogate/regime-switching is the highest-ROI item — LLM costs doubled with CC launch, so savings are amplified. VaR moved up because two correlated commodities (coffee + cocoa share growing regions) create portfolio risk that per-trade checks can't detect. Both are independent and can run in parallel.
