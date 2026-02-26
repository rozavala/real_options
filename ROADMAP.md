# Real Options — Engineering Roadmap

**Last updated:** 2026-02-27
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
| E.1 Portfolio-Level VaR | **Done.** PR #976 (merged Feb 20). `trading_bot/var_calculator.py` (1127 lines): Full Revaluation HS VaR at 95%/99% with B-S repricing, batched IV fetch, yfinance historical returns, AI Risk Agent (L1 Interpreter + L2 Scenario Architect). Compliance gate with 3-mode enforcement (`log_only`→`warn`→`enforce`), startup grace period, emergency bypass. VaR dampener in weighted voting (80-100% utilization → 1.0-0.5x confidence). Shared `data/var_state.json` (portfolio-wide, not per-commodity). 34 tests (25 VaR + 9 compliance). Ships as Phase A (`log_only`). |
| F.5 Prediction Market Integration | **Done.** `PredictionMarketSentinel` overhauled with Gamma API. `TopicDiscoveryAgent` automates topic finding using Claude Haiku (dynamic interest areas, LLM relevance filtering). Integrated with TMS for zombie position protection. |
| E.3 Liquidity-Aware Execution | **Done.** `order_manager.py:check_liquidity_conditions()` — pre-execution bid/ask depth analysis, BAG combo leg liquidity aggregation, per-order spread logging. Remaining VWAP/TWAP only matters at much larger position sizes ($500K+). |
| G.6 3rd Commodity (NG) | **Done.** Natural Gas (NG) launched Feb 27. Commodity profile created, systemd service installed, data dirs initialized. Running live alongside KC and CC. |
| **Schedule Optimization** | **Done.** Reduced signal frequency from 4 to 3 cycles per session (15%, 47.5%, 80%) to eliminate illiquid pre-open window and reduce cost. |

---

## Prioritized Backlog (optimized for return on capital)

### #1 — C.4 Surrogate Model Activation → C.2 Regime-Switching Inference
**Type:** Cost → P&L | **Effort:** 3-4 weeks (both) | **Status:** Partial (surrogate exists, not wired)

**C.4 (2 weeks):** `backtesting/surrogate_models.py` has a complete GradientBoosting pipeline (`DecisionSurrogate` class with 10 features: price trends, SMA cross, ATR, IV rank, RSI, sentiment, weather, inventory, day-of-week). Need: wire to live `council_history.csv` for training data, automated weekly retraining, model versioning in `data/{ticker}/surrogate_models/`, accuracy tracking via Brier-like metric. Now multi-commodity: train separate models per commodity.

**C.2 (2 weeks, after C.4):** Route inference by market regime: quiet/range-bound days use surrogate model (~$0 per decision), news-driven days use lightweight 2-agent panel, crises use full Council. ~70% of trading days are quiet. Note: `weighted_voting.py` already has `RegimeDetector.detect_regime()` (volatility/trend from 5-day bars) and `detect_market_regime_simple()` fallback, plus regime-adjusted agent weights. Missing piece: conditional routing in `signal_generator.py` to skip the full Council when surrogate suffices.

**Why #1:** Direct bottom-line impact. With three commodities running (KC, CC, NG), LLM costs tripled. If LLM costs are $500-1000/month per commodity, regime routing saves 60-80% ($900-2400/month). The surrogate also provides a fast "second opinion" that can flag when the Council is hallucinating.

**Revenue impact:** High — cost savings drop straight to profit. Impact multiplied by number of commodities.

**Prerequisites:** C.4 before C.2.

---

### #2 — E.4 Options Greeks Monitor
**Type:** Survival | **Effort:** 2-3 weeks | **Status:** Partial (IV in VaR only)

Real-time portfolio Greeks (Delta, Gamma, Theta, Vega). Options spreads have non-linear risk — a "delta-neutral" straddle becomes directional after a price move. Currently the system uses delta for strike selection (`strategy.py:find_strike_by_delta`) and IV for VaR B-S repricing, but no Greeks monitoring after entry.

Need: per-position and portfolio-level Greeks via IB's model, threshold alerts (e.g., portfolio delta exceeds ±50), theta-burn warnings, integration with position audit exit cycle. Now multi-commodity: aggregate Greeks across KC+CC+NG positions.

**Why #2:** Prevents invisible loss accumulation. Without Greeks monitoring, the exit cycle can't distinguish between "position is slightly down but structurally fine" and "position has become a naked directional bet due to gamma." Critical with three commodities — more concurrent positions mean more gamma risk. Now unblocked by E.1 (VaR already fetches IVs from IB).

**Revenue impact:** Medium-high — loss prevention on existing positions.

**Prerequisites:** E.1 (done — VaR provides the IV fetch infrastructure)

---

### #3 — A.2 TextGrad
**Type:** Optimization | **Effort:** 3-4 weeks | **Status:** Not started

Textual backpropagation — when a trade loses, a Judge LLM generates specific critique ("the analysis failed to account for X") and suggests prompt edits. Completes the self-learning loop: DSPy (done) optimizes structure, TextGrad optimizes reasoning.

**Why #3:** Now that DSPy is live, TextGrad is the next compounding investment. Every losing trade becomes a training signal. The trade journal (A.5, done) provides the raw material — TextGrad adds the "what should we do differently" layer. With three commodities, the training signal triples.

**Revenue impact:** Medium — improves win rate over time. Effect compounds.

**Prerequisites:** A.1 (done)

---

### #4 — G.2 A/B Testing Framework
**Type:** Safety | **Effort:** 2-3 weeks | **Status:** Not started

When DSPy/TextGrad propose prompt changes, run them as A/B tests. 50% of cycles use old prompt, 50% new. Statistical significance test determines winner. Auto-promote at p<0.05, auto-reject at p>0.95 (null).

**Why #4:** DSPy is live and generating optimized prompts. Without A/B testing, there's no safe way to validate them on live markets. Currently `dspy.use_optimized_prompts` is a binary config toggle — no gradual rollout, no statistical validation. Multi-commodity triples the sample size for faster convergence.

**Revenue impact:** Medium — safety gate for the optimization pipeline. Prevents bad prompts from going live.

**Prerequisites:** A.1 (done)

---

### #5 — D.5 Reasoning Quality Metrics
**Type:** Insight | **Effort:** 3-4 weeks | **Status:** Not started

Text-Based Information Coefficient (sentiment signal x confidence vs actual returns), Faithfulness via NLI (are claims supported by retrieved docs?), Debate Divergence (semantic distance between Permabear/Permabull — low = mode collapse).

**Why #5:** Identifies *why* bad decisions happen, not just *that* they happened. Brier scoring (done) tells you accuracy; this tells you reasoning quality. Catches mode collapse (all agents agreeing for wrong reasons) and hallucination (confident claims unsupported by data).

**Revenue impact:** Medium — diagnostic, not directly revenue-generating. But saves weeks of debugging when something goes wrong.

---

### #6 — D.1 Event-Driven Backtest Engine
**Type:** Validation | **Effort:** 6-8 weeks | **Status:** Not started

Discrete-event simulation replaying historical data through the full system. Priority queue architecture, simulated latency, strict temporal isolation. Uses surrogate (C.4) for normal regimes, full Council for crises.

**Why #6:** The only rigorous way to validate strategy changes. Table-stakes for serious trading.

**Prerequisites:** C.4, B.2 (done)

---

### #7 — D.4 Paper Trading Shadow Mode
**Type:** Safety | **Effort:** 3-4 weeks | **Status:** Not started

Run proposed changes in a paper-trading sandbox alongside live system. Both see same data; only live executes. Compare decisions and outcomes. Promotion gate: shadow must outperform live for N days.

**Why #7:** Cheaper, faster validation than full backtesting for incremental changes.

---

### #8 — F.1 Dynamic Agent Recruiting
**Type:** Intelligence | **Effort:** 3-4 weeks | **Status:** Not started

Meta-agent spawns temporary specialists for unusual events (strikes, new regulations, pest outbreaks). Decommissions when event passes. Template library per event category.

---

### #9 — B.3 Agentic RAG for Research
**Type:** Knowledge | **Effort:** 4-5 weeks | **Status:** Not started

Researcher agent monitors preprint servers, USDA/ICO reports. Extracts causal mechanisms and quantitative findings. Stores in TMS as `type: RESEARCH_FINDING`.

**Prerequisites:** B.2 (done)

---

### #10 — G.1 Formal Observability (AgentOps)
**Type:** Ops | **Effort:** 2-3 weeks | **Status:** Partial (internal only)

Custom `observability.py` exists with HallucinationDetector, AgentTrace, and ObservabilityHub. Missing: third-party integration (AgentOps or LangSmith) for trace replay, cost attribution dashboards, and alert rules. Keep custom hallucination detection as it's commodity-aware.

---

### #11 — F.6 Synthetic Rare Event Generation
**Type:** Stress test | **Effort:** 3-4 weeks | **Status:** Not started

Combine real historical events in novel ways for stress testing. "What if frost + strike + BRL crash simultaneously?" Augments DSPy training set. Note: E.1 VaR already includes basic stress scenarios (price + IV shocks via `compute_stress_scenario()`); this extends to multi-factor combinatorial scenarios.

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

✅ F.4 Neuro-Symbolic (done) ──→ ✅ E.1 Portfolio VaR (done) ──→ E.4 Greeks Monitor

✅ E.2 Exit Enhancements (done)
✅ E.3 Liquidity-Aware (done)

✅ G.5 Multi-Commodity (done) ──→ ✅ G.6 3rd Commodity (done)
```

## Recommended Execution Plan

| Phase | Items | Duration | Focus |
|-------|-------|----------|-------|
| ~~Phase 1~~ | ~~A.3, C.1, C.5~~ | ~~3-5 weeks~~ | **Done** |
| ~~Phase 2~~ | ~~F.4, E.2~~ | ~~4-6 weeks~~ | **Done** |
| ~~Phase 3~~ | ~~A.1 DSPy~~ | ~~3-4 weeks~~ | **Done** |
| ~~Phase 4a~~ | ~~G.5 Multi-Commodity~~ | ~~2 weeks~~ | **Done** (CC launched Feb 17) |
| ~~Phase 4b~~ | ~~E.1 VaR, F.5 Prediction Market~~ | ~~3-4 weeks~~ | **Done** (VaR Feb 20, PM Integration) |
| ~~Phase 4c~~ | ~~G.6 3rd Commodity (NG)~~ | ~~1 week~~ | **Done** (NG launched Feb 27) |
| **Phase 5** | **#1 C.4/C.2 Surrogate+Regime** | **3-4 weeks** | **Cost reduction — LLM costs tripled with 3 commodities, surrogate saves 60-80%** |
| Phase 6 | #2 E.4 Greeks | 2-3 weeks | Risk visibility |
| Phase 7 | #3 A.2 TextGrad + #4 G.2 A/B Testing | 4-6 weeks | Decision quality + optimization safety |
| Phase 8 | #5 D.5 Reasoning Metrics | 4-6 weeks | Diagnostics + validation |
| Phase 9 | #6-#11 (depth + scale) | 20-30 weeks | Foundation for scale |

**Phase 5 rationale:** Surrogate/regime-switching is now the clear #1 priority. E.1 VaR shipped, so the survival gap is closed. LLM costs are the biggest drag on profitability — three commodities each running 3 scheduled signal cycles/day (optimized from 4) plus sentinels through a 7-agent Council is expensive. Regime routing on quiet days (70% of sessions) could save $900-2400/month across KC+CC+NG.
