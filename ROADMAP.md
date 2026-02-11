# Real Options — Engineering Roadmap

**Last updated:** 2026-02-10
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
| A.4 Dynamic Agent Weight Self-Tuning | **Done.** `weighted_voting.py` integrates Brier scores via `brier_bridge.py` + regime multipliers + staleness decay |
| A.5 Automated Trade Journal | **Done.** `trade_journal.py` generates LLM post-mortems, called from `reconciliation.py`, stored in TMS |
| B.2 TMS Temporal Filtering | **Done.** Exponential decay, `valid_from` timestamps, simulation clock for backtests in `tms.py` |
| F.2 Generative Simulacra | Research-grade (6-8 wk), unproven alpha source. Revisit after core profitability validated |
| F.3 Liquid Neural Networks | Research-grade (8-12 wk), needs tick infrastructure that doesn't exist |
| C.3 Knowledge Distillation (SLM) | Model API costs dropping fast. Training custom SLM for 4-6 weeks unlikely to break even |
| B.1 Temporal GraphRAG | Heavy infra (Neo4j, 6-8 wk). TMS with temporal filtering covers 80% of use case |
| G.6 Cross-Commodity Correlation | Premature — no second commodity running yet |
| D.2 Time-Travel Data Layer | Massive data engineering, depends on D.1 which doesn't exist |
| D.3 Walk-Forward Optimization | Depends on D.1 + D.2 |
| G.8 Notification Expansion | Pushover works for single operator |
| G.3 Advanced Dashboard | Incremental — build pages as modules ship |
| E.5 Market Impact Modeling | Position sizes 1-5 contracts on $50K, impact negligible. Revisit at $500K+ |
| G.7 Automated Incident Post-Mortems | Trade journal (A.5) + self-healing (G.4) cover 80% |
| B.4 Cross-Agent Knowledge Graph | TMS handles knowledge sharing; active cross-cueing unproven need |

---

## Prioritized Backlog

### #1 — A.3 Reflexion Expansion
**Type:** Optimization | **Effort:** 1-2 weeks | **Status:** Partial

Extend the self-critique + episodic memory retrieval loop from agronomist/macro to ALL Tier 2 agents (technical, sentiment, geopolitical, supply_chain, inventory, volatility, microstructure). Pattern already exists in `agents.py:research_topic_with_reflexion()`. Every agent should say "last time I saw this pattern I was wrong because..."

**Why #1:** Fastest win — pattern exists, just extend. Immediate decision quality lift across all agents.

---

### #2 — C.1 Semantic Cache Completion
**Type:** Cost | **Effort:** 1-2 weeks | **Status:** Partial

Module is 95% built (`semantic_cache.py`). Was removed from `heterogeneous_router.py` on 2026-02-04 (P0-A fix) because it was wired at the wrong level (prompt-level vs market-state-level). Comment says "should be implemented at orchestrator level" — never done. Config has `"enabled": false`. Need to: wire into orchestrator at the council-decision level, enable in config, add sentinel-alert invalidation.

**Why #2:** 95% built, 0% active. $220-330/year savings once enabled.

---

### #3 — F.4 Neuro-Symbolic Compliance
**Type:** Survival | **Effort:** 2-3 weeks | **Status:** Not started

Extract deterministic rules from LLM compliance prompts into hard-coded symbolic checks: position limits, max loss per trade, concentration limits, wash trade detection. LLM compliance layer stays as a second opinion. Deterministic rules can't be "convinced" by a persuasive agent.

**Why #3:** LLM-based compliance is the single biggest structural risk on a live money system.

---

### #4 — E.2 Dynamic Exit Enhancements
**Type:** P&L | **Effort:** 2-3 weeks | **Status:** Partial

Add trailing stops for winners, theta-burn acceleration for options approaching expiry, regime-aware exit thresholds. Current exit logic is binary (thesis valid/invalid). Need: ratcheting stops as positions move in favor, time-decay awareness, multi-factor thesis invalidation.

**Why #4:** Biggest P&L lever. System enters fine but leaves money on the table or holds losers too long.

---

### #5 — C.5 Budget Guard Wiring
**Type:** Cost | **Effort:** 1 week | **Status:** Partial

Budget guard module has graduated throttling logic but `record_cost()` is never called from the orchestrator/router — the meter isn't connected. The `is_budget_hit` binary gate works but the per-priority throttling (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND) is dead code without actual cost tracking.

**Why #5:** 1 week to activate existing dead code. Prevents budget overruns.

---

### #6 — A.1 DSPy Prompt Optimization
**Type:** Optimization | **Effort:** 3-4 weeks | **Status:** Not started

Replace manual prompt engineering with programmatic optimization. DSPy treats prompts as learnable parameters — bootstraps few-shot examples from winning trades, uses MIPROv2 to evolve instructions. New module `trading_bot/dspy_optimizer.py`. Training metric: composite of Sharpe, Brier improvement, max drawdown.

**Why #6:** Highest-leverage automation investment. Compounds over time. Trade journal (A.5) and reflexion (A.3) provide the training signal it needs.

**Prerequisites:** A.3, A.5 (both done or in progress by this point)

---

### #7 — E.1 Portfolio-Level VaR
**Type:** Survival | **Effort:** 3-4 weeks | **Status:** Not started

Historical simulation VaR at 95%/99% confidence. Currently only per-trade risk exists in `risk_management.py`. Need portfolio-level aggregation with correlation between positions, stress testing against known scenarios, integration with compliance to block new trades on VaR breach.

**Why #7:** Critical for capital preservation, but lower urgency on single-commodity $50K account with 1-3 positions. Becomes top priority at multi-commodity stage.

---

### #8 — E.4 Options Greeks Monitor
**Type:** Survival | **Effort:** 2-3 weeks | **Status:** Not started

Real-time portfolio Greeks (Delta, Gamma, Theta, Vega). Options positions have non-linear risk — a "delta-neutral" straddle becomes directional after a price move. Need: per-position and portfolio-level Greeks, threshold alerts, hedging recommendations, integration with exit cycle.

**Why #8:** Same reasoning as E.1 — important but manageable at current portfolio size.

---

### #9 — A.2 TextGrad
**Type:** Optimization | **Effort:** 3-4 weeks | **Status:** Not started

Textual backpropagation — when a trade loses, a Judge LLM generates specific critique ("the analysis failed to account for X") and suggests prompt edits. Completes the self-learning loop with DSPy: DSPy optimizes structure, TextGrad optimizes reasoning.

**Why #9:** Depends on A.1 (DSPy) for full value.

**Prerequisites:** A.1

---

### #10 — C.4 Surrogate Model Activation
**Type:** Foundation | **Effort:** 2 weeks | **Status:** Partial

`backtesting/surrogate_models.py` has a complete GradientBoosting pipeline but isn't wired to live `council_history.csv`. Need: automated feature extraction, training trigger (weekly or on 50 new decisions), model versioning, accuracy tracking.

**Why #10:** Foundation for backtesting (D.1) and regime switching (C.2).

---

### #11 — C.2 Regime-Switching Inference
**Type:** Cost | **Effort:** 3-4 weeks | **Status:** Not started

Route inference by market regime: quiet days use surrogate model (~$0), news events use lightweight models, crises use full Council. ~70% of trading days are quiet — running full Council every time is waste.

**Why #11:** 60-80% cost reduction. Needs surrogate (C.4).

**Prerequisites:** C.4

---

### #12 — G.2 A/B Testing Framework
**Type:** Safety | **Effort:** 2-3 weeks | **Status:** Not started

When DSPy/TextGrad propose prompt changes, run them as A/B tests. 50% of cycles use old prompt, 50% new. Statistical significance test determines winner. Auto-promote at p<0.05.

**Why #12:** Without this, DSPy/TextGrad deploy unvalidated changes to a live money system.

**Prerequisites:** A.1

---

### #13 — D.1 Event-Driven Backtest Engine
**Type:** Validation | **Effort:** 6-8 weeks | **Status:** Not started

Discrete-event simulation replaying historical data through the full system. Priority queue architecture, simulated latency, strict temporal isolation. Uses surrogate (C.4) for normal regimes, full Council for crises.

**Why #13:** The only rigorous way to validate strategy changes. Table-stakes for serious trading.

**Prerequisites:** C.4, B.2 (done)

---

### #14 — D.4 Paper Trading Shadow Mode
**Type:** Safety | **Effort:** 3-4 weeks | **Status:** Not started

Run proposed changes in a paper-trading sandbox alongside live system. Both see same data; only live executes. Compare decisions and outcomes. Promotion gate: shadow must outperform live for N days.

**Why #14:** Cheaper, faster validation than full backtesting for incremental changes.

---

### #15 — D.5 Reasoning Quality Metrics
**Type:** Insight | **Effort:** 3-4 weeks | **Status:** Not started

Text-Based Information Coefficient (sentiment signal x confidence vs actual returns), Faithfulness via NLI (are claims supported by retrieved docs?), Debate Divergence (semantic distance between Permabear/Permabull — low = mode collapse).

**Why #15:** Identifies *why* bad decisions happen, not just *that* they happened.

---

### #16 — G.5 Multi-Commodity (1st new market)
**Type:** Growth | **Effort:** 3-4 weeks | **Status:** Not started

Create complete profiles for Sugar (SB), Cocoa (CC), or Cotton (CT). Run end-to-end validation. Fix any remaining coffee-specific hardcoding.

**Why #16:** Revenue diversification + validates the commodity-agnostic architecture.

---

### #17 — F.1 Dynamic Agent Recruiting
**Type:** Intelligence | **Effort:** 3-4 weeks | **Status:** Not started

Meta-agent spawns temporary specialists for unusual events (strikes, new regulations, pest outbreaks). Decommissions when event passes. Template library per event category.

---

### #18 — B.3 Agentic RAG for Research
**Type:** Knowledge | **Effort:** 4-5 weeks | **Status:** Not started

Researcher agent monitors preprint servers, USDA/ICO reports. Extracts causal mechanisms and quantitative findings. Stores in TMS as `type: RESEARCH_FINDING`.

**Prerequisites:** B.2 (done)

---

### #19 — E.3 Liquidity-Aware Execution
**Type:** Execution | **Effort:** 2-3 weeks | **Status:** Basic

Pre-execution book depth analysis, VWAP/TWAP for larger orders, thin-market detection. Matters more as account grows.

---

### #20 — F.5 Prediction Market Deep Integration
**Type:** Validation | **Effort:** 2-3 weeks | **Status:** Partial

Use Polymarket/Kalshi as cross-check against Council decisions. Divergence scoring, tiebreaker logic for close votes.

---

### #21 — G.1 Formal Observability (AgentOps)
**Type:** Ops | **Effort:** 2-3 weeks | **Status:** Not started

Replace custom `observability.py` with AgentOps or LangSmith. Full trace replay, cost attribution, alert rules. Keep custom as fallback.

---

### #22 — F.6 Synthetic Rare Event Generation
**Type:** Stress test | **Effort:** 3-4 weeks | **Status:** Not started

Combine real historical events in novel ways for stress testing. "What if frost + strike + BRL crash simultaneously?" Augments DSPy training set.

**Prerequisites:** D.1

---

## Dependency Graph

```
A.3 Reflexion ──→ A.1 DSPy ──→ A.2 TextGrad ──→ G.2 A/B Testing
                     │
A.5 Trade Journal ───┘  (done, provides training data)

C.1 Semantic Cache (wire into orchestrator)
C.5 Budget Guard (wire record_cost into router)

C.4 Surrogate ──→ C.2 Regime Switching
       │
       └──→ D.1 Backtest Engine ──→ D.4 Shadow Mode
                    │
                    └──→ F.6 Synthetic Events

F.4 Neuro-Symbolic ──→ E.1 Portfolio VaR ──→ E.4 Greeks Monitor

E.2 Exit Enhancements (independent)

G.5 Multi-Commodity (independent, unlocks G.6 later)
```

## Estimated Timeline (sequential, single-threaded)

| Phase | Items | Duration | Cumulative |
|-------|-------|----------|------------|
| Phase 1 | #1, #2, #5 (quick completions) | 3-5 weeks | 3-5 weeks |
| Phase 2 | #3, #4 (structural improvements) | 4-6 weeks | 7-11 weeks |
| Phase 3 | #6, #7, #8 (self-learning + risk) | 8-11 weeks | 15-22 weeks |
| Phase 4 | #9-#12 (cost + validation) | 10-15 weeks | 25-37 weeks |
| Phase 5 | #13-#22 (growth + depth) | 25-35 weeks | 50-72 weeks |

Note: Many items are parallelizable. With concurrent work, Phase 1-3 could compress to ~12-16 weeks.
