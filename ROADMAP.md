# Real Options — Engineering Roadmap

**Last updated:** 2026-03-16
**Reviewed by:** Claude

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
| G.8 Notification Expansion | **Done.** Replaced with severity-gated operator alerts (Pushover) directly prioritizing critical anomalies. Routine noise is silenced to avoid operator fatigue |
| G.3 Advanced Dashboard | Incremental — build pages as modules ship |
| E.5 Market Impact Modeling | Position sizes 1-5 contracts on $50K, impact negligible. Revisit at $500K+ |
| G.7 Automated Incident Post-Mortems | Trade journal (A.5) + self-healing (G.4) cover 80% |
| B.4 Cross-Agent Knowledge Graph | TMS handles knowledge sharing; active cross-cueing unproven need |
| G.5 Multi-Commodity | **Done.** Cocoa (CC) launched Feb 17. Full path isolation across 34 modules, per-commodity systemd services, staggered boot, auto-detect deploy pipeline. PRs #879-#949. KC + CC running simultaneously on DEV. Key infra: `set_data_dir()` pattern across all stateful modules, `COMMODITY_ID_OFFSET` for IB client IDs, `_resolve_data_path()` for dashboard, `migrate_data_dirs.py` for migration, per-commodity deploy verification |
| Data Pipeline Automation | **Done.** Standalone backfill scripts (e.g., Execution Funnel, Contribution Scores) wrapped into idempotent `run_migrations.py` framework for consistent deployment via GitHub Actions |
| E.1 Portfolio-Level VaR | **Done.** PR #976 (merged Feb 20). `trading_bot/var_calculator.py` (1127 lines): Full Revaluation HS VaR at 95%/99% with B-S repricing, batched IV fetch, yfinance historical returns, AI Risk Agent (L1 Interpreter + L2 Scenario Architect). Compliance gate with 3-mode enforcement (`log_only`→`warn`→`enforce`), startup grace period, emergency bypass. VaR dampener in weighted voting (80-100% utilization → 1.0-0.5x confidence). Shared `data/var_state.json` (portfolio-wide, not per-commodity). 34 tests (25 VaR + 9 compliance). Ships as Phase A (`log_only`). |
| F.5 Prediction Market Integration | **Done.** `PredictionMarketSentinel` overhauled with Gamma API. `TopicDiscoveryAgent` automates topic finding using Claude Haiku (dynamic interest areas, LLM relevance filtering). Integrated with TMS for zombie position protection. |
| E.3 Liquidity-Aware Execution | **Done.** `order_manager.py:check_liquidity_conditions()` — pre-execution bid/ask depth analysis using a Hybrid Tick/Percentage Liquidity Filter (replacing the pure-ratio model), BAG combo leg liquidity aggregation, per-order spread logging. Remaining VWAP/TWAP only matters at much larger position sizes ($500K+). |
| G.6 3rd Commodity (NG) | **Done.** Natural Gas (NG) launched Feb 27. Commodity profile created, systemd service installed, data dirs initialized. Running live alongside KC and CC. |
| **Schedule Optimization** | **Done.** Reduced signal frequency from 4 to 3 cycles per session (20%, 62%, 80%) to eliminate illiquid pre-open window and reduce cost. |
| **Execution Scaling** | **Done.** Introduced atomic BAG combo closes for multi-leg option positions with limit-and-walk fallback. Widened Drawdown circuit breaker thresholds (6% halt, 9% panic) to account for multi-commodity mark-to-market bid-ask noise. Added Conviction Gate to suppress weak signals, and Sentinel IC suppression. Further hardened in March 2026: disconnect guard clientId filtering (#1279), cross-commodity close prevention (#1281), profile-driven market order fallback timeouts (#1275), NG emergency trigger tuning (#1261), CC liquidity threshold adjustment (#1259) |
| **Contribution-Based Scoring** | **Done.** PRs #1249-#1255 (March 2026). Replaced Brier-only accuracy scoring with contribution-based agent attribution — measures whether each agent helped or hurt the council's final decision. `trading_bot/contribution_scoring.py` computes per-agent scores across regimes (NORMAL, HIGH_VOL, RANGE_BOUND). DSPy evaluation and enablement checks updated to use contribution metric. NEUTRAL penalized on real moves. Per-agent abstention ceilings added. Migration script backfills from `council_history.csv`. |
| **Execution Funnel** | **Done.** PRs #1287-#1291 (March 2026). Signal-to-P&L diagnostic pipeline: append-only CSV event log tracking each decision from council signal through order placement, fill, and realized P&L. Dashboard page (10_The_Funnel.py) with regime comparison, slippage stats, and cycle drill-down. Backfill script reconstructs funnel from historical `council_history.csv` + `trade_ledger.csv`. |
| **Master Strategist Forensics** | **Done.** PRs #1256-#1257 (March 2026). `scripts/master_strategist_forensics.py` analyzes Master Strategist override patterns, agent agreement rates, and decision quality. Integrated into Scorecard dashboard page. |
| **Reconciliation Hardening** | **Done.** PRs #1217-#1218, #1229, #1233-#1235, #1247 (March 2026). Eliminated phantom position alerts from double-counted reconciliation entries. Fixed multi-lot combo fill recording. Increased timestamp tolerance to 30s. Disabled phantom reconciliation that was creating false RECON_MISSING entries. Vectorized trade reconciliation loop (#1303). |
| **MasterOrchestrator (Multi-Engine)** | **Done.** Consolidated KC and NG into a single `MasterOrchestrator` process with ContextVar isolation per engine. Replaced per-commodity systemd services. Staggered boot, shared IB Gateway, per-commodity client ID offsets. Deployed to production March 2026 (#1294). |
| **Production Deployment** | **Done.** System deployed to production (March 2026). 11-step `deploy.sh` with automatic rollback, `verify_deploy.sh` (8 checks), `run_migrations.py` framework with `--verify` flag. KC and NG engines running live. Trading mode OFF (observation only) pending validation. |
| G.7 Three-Tier Market State | **Done.** `ACTIVE`/`PASSIVE`/`SLEEPING` state model for extended-hours commodities (NG). Uses JSON config profiles. |

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
**Type:** Insight | **Effort:** 2-3 weeks | **Status:** Partial

Text-Based Information Coefficient (sentiment signal x confidence vs actual returns), Faithfulness via NLI (are claims supported by retrieved docs?), Debate Divergence (semantic distance between Permabear/Permabull — low = mode collapse).

**Partial progress:** Contribution scoring (done) now tracks per-agent value-add vs the council decision. Execution Funnel (done) traces signal-to-P&L with slippage analysis. Master Strategist forensics (done) analyzes override patterns and agent agreement rates. Direction-evidence mismatch detection already flags agents whose directional call contradicts their own evidence. **Remaining:** formal NLI faithfulness checking, debate divergence metric, systematic mode-collapse detection.

**Why #5:** Identifies *why* bad decisions happen, not just *that* they happened. Contribution scoring tells you *who* helps/hurts; this tells you *why*. Catches mode collapse (all agents agreeing for wrong reasons) and hallucination (confident claims unsupported by data).

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
**Type:** Ops | **Effort:** 2-3 weeks | **Status:** Partial (internal + GitHub)

Custom `observability.py` exists with HallucinationDetector, AgentTrace, and ObservabilityHub. The decoupled `scripts/error_reporter.py` script now handles out-of-band log telemetry, intelligent filtering of transient operational noise (e.g., rate limits, lock timeouts, 503 errors), and auto-generation of structured GitHub issues with full backtraces for true system anomalies. Real-time operator alerts (via Pushover) are now natively severity-gated. Missing: third-party integration (AgentOps or LangSmith) for trace replay, cost attribution dashboards, and advanced real-time alert rules. Keep custom hallucination detection as it's commodity-aware.

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
✅ Contribution Scoring (done) ──→ D.5 Reasoning Metrics (partial)
✅ Execution Funnel (done) ────────┘

C.4 Surrogate ──→ C.2 Regime Switching
       │
       └──→ D.1 Backtest Engine ──→ D.4 Shadow Mode
                    │
                    └──→ F.6 Synthetic Events

✅ F.4 Neuro-Symbolic (done) ──→ ✅ E.1 Portfolio VaR (done) ──→ E.4 Greeks Monitor

✅ E.2 Exit Enhancements (done)
✅ E.3 Liquidity-Aware (done)
✅ Execution Scaling (done)
✅ Reconciliation Hardening (done)

✅ G.5 Multi-Commodity (done) ──→ ✅ G.6 3rd Commodity (done) ──→ ✅ MasterOrchestrator (done)
                                                                          │
✅ G.7 Three-Tier State (done) ────────────────────────────────────────────┘
✅ Production Deploy (done)
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
| ~~Phase 4d~~ | ~~Data Pipeline Migrations~~ | ~~1 week~~ | **Done** (Execution Funnel & Contribution Scores backfilled via framework) |
| ~~Phase 4e~~ | ~~Contribution Scoring, Execution Funnel, Forensics, Reconciliation, MasterOrchestrator, Production Deploy~~ | ~~3 weeks~~ | **Done** (March 2026). Contribution scoring replaces Brier-only. Execution Funnel traces signal→P&L. Forensics dashboard. Reconciliation hardened. MasterOrchestrator consolidates engines. Deployed to production. |
| **Phase 5** | **#1 C.4/C.2 Surrogate+Regime** | **3-4 weeks** | **Cost reduction — LLM costs tripled with 3 commodities, surrogate saves 60-80%** |
| Phase 6 | #2 E.4 Greeks | 2-3 weeks | Risk visibility |
| Phase 7 | #3 A.2 TextGrad + #4 G.2 A/B Testing | 4-6 weeks | Decision quality + optimization safety |
| Phase 8 | #5 D.5 Reasoning Metrics | 2-3 weeks | Diagnostics (partially done — contribution scoring, funnel, forensics) |
| Phase 9 | #6-#11 (depth + scale) | 20-30 weeks | Foundation for scale |

**Phase 5 rationale:** Surrogate/regime-switching is now the clear #1 priority. The system is live in production with KC and NG engines running (trading mode OFF, observation only). LLM costs are the biggest drag on profitability — two active commodities each running 3 scheduled signal cycles/day plus sentinels through a 7-agent Council is expensive. Regime routing on quiet days (70% of sessions) could save $600-1600/month across KC+NG.
