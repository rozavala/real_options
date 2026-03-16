# Real Options: System Architecture and Knowledge

This document provides a detailed breakdown of how the Real Options system operates.

## Architecture: The Federated Cognitive Lattice

The system is organized into a tiered architecture:

### Tier 1: Always-On Sentinels
Sentinels are lightweight monitors that observe external data sources and trigger the decision pipeline when significant events occur.
- **PriceSentinel:** Monitors for rapid price movements or liquidity gaps.
- **WeatherSentinel:** Tracks weather patterns in key commodity regions (e.g., Brazil for Coffee (KC) or Cocoa (CC)) using energy-aware stages for non-agricultural commodities.
- **LogisticsSentinel:** Monitors RSS feeds for strikes, port closures, or canal disruptions.
- **NewsSentinel:** Scans news feeds for fundamental shifts.
- **XSentimentSentinel:** Analyzes social media sentiment on X (via xAI) and incorporates a multi-level spend-cap circuit breaker to throttle API calls during inactive market regimes or upon hitting budget limits.
- **PredictionMarketSentinel:** Monitors Polymarket odds for geopolitical or macro events.
- **TopicDiscoveryAgent:** Dynamically scans Polymarket for new, relevant topics using Claude Haiku (with bulk fetch, commodity filtering, and state pruning) to auto-configure the PredictionMarketSentinel.
- **MacroContagionSentinel:** Detects cross-asset contagion (e.g., DXY, Gold correlation).
- **FundamentalRegimeSentinel:** Determines long-term surplus/deficit regimes.
- **MicrostructureSentinel:** Monitors order book depth and flow toxicity.

### Tier 2: Specialist Analysts (The Council)
When a sentinel triggers or a scheduled cycle runs, a Council of specialized AI agents is convened via the **Heterogeneous Router**.
- **Agronomist:** Evaluates crop conditions and weather impacts.
- **Macro Economist:** Assesses global economic trends and currency impacts.
- **Fundamentalist:** Focuses on supply/demand balance and inventory reports.
- **Technical Analyst:** Interprets chart patterns and price action.
- **Volatility Analyst:** Analyzes implied volatility and options pricing.
- **Geopolitical Analyst:** Evaluates the impact of international relations and conflicts.
- **Sentiment Analyst:** Gauges the market mood from social and news sources.
- **Inventory Analyst:** Monitors certified stocks.
- **Supply Chain Analyst:** Tracks logistics bottlenecks.

### Tier 3: Decision Council
A set of agents that synthesize the analysts' reports.
- **Permabear:** Attacks the bullish thesis.
- **Permabull:** Defends the bullish thesis.
- **Master Strategist:** Weighs all evidence and makes the final directional decision.
- **Devil's Advocate:** Performs a pre-mortem to identify risks in the master strategy.

### Tier 4: Execution & Risk Management
- **Compliance Guardian:** The final arbiter of all trades, enforcing risk limits (VaR, Margin, etc.), the Conviction Gate (blocking weak signals, e.g., |weighted_score| < 0.20), and Sentinel IC Suppression (blocking short premium positions during sentinel-triggered high-volatility events).
- **Dynamic Position Sizer:** Calculates optimal trade size.
- **Order Manager:** Queues and executes orders via Interactive Brokers, utilizing a Hybrid Tick/Percentage Liquidity Filter for combo orders. It executes CONTRADICT closures immediately prior to new entries to prevent quantity aggregation race conditions. Multi-leg positions are closed atomically via single BAG (combo) orders, and the execution is verified via `reqPositionsAsync` before falling back to individual leg closures to prevent orphan legs. All exits utilize limit orders with adaptive price walking, falling back to market orders via a profile-driven timeout (e.g., KC=90s, CC=120s, NG=60s). Catastrophe stops are gated by an account Net Liquidation Value (NLV) minimum threshold to fail-closed without stop protection if margin is insufficient.

## Infrastructure

- **Master Orchestrator:** Manages multi-commodity instances for active tickers (Coffee, Cocoa, Natural Gas). Notifications are isolated per commodity ticker via ContextVar. Multi-commodity isolation requires strict `clientId` filtering and symbol prefix matching to prevent cross-engine order corruption or false untracked positions.
- **Commodity Engine:** Runs the per-commodity loop.
- **Heterogeneous Router:** Dispatches LLM calls to Gemini, OpenAI, Anthropic, or xAI. Dynamically routes specific agent roles to optimized providers (e.g., Gemini Pro for the Geopolitical Analyst and xAI for the Trade Analyst), incorporating multiple fallback providers for resilience.
- **Semantic Cache:** Caches decisions to optimize costs and latency.
- **DSPy Optimizer:** Offline pipeline that refines agent prompts using historical feedback (BootstrapFewShot). Evaluates based on directional accuracy and abstention rate directly from `enhanced_brier.json` (legacy CSV paths have been deprecated).
- **Error Reporter Pipeline:** A standalone telemetry script (`scripts/error_reporter.py`) decoupled from the orchestrator. It ensures fail-safe operational awareness by parsing system logs, filtering out expected transient noise (e.g., `503 UNAVAILABLE`, `RESOURCE_EXHAUSTED`, rate limits, `CIRCUIT BREAKER`, emergency lock timeouts), and uses fingerprinting to deduplicate and auto-generate structured GitHub issues for true anomalies.
- **System Readiness Verifier:** A comprehensive pre-flight diagnostic script (`verify_system_readiness.py`) that performs 27 comprehensive component checks (infrastructure, connections, data fallbacks, agent array, and execution pipeline) to diagnose system state before runtime.
- **Market Data Diagnostic:** A lightweight utility (`scripts/check_market_data.py`) to verify Interactive Brokers data feed status, ensuring `FORCE_DELAYED_DATA` overrides function correctly across DEV and PROD environments by resolving front-month contracts and reporting LIVE/DELAYED quote status.
- **Migration Pipeline:** Wraps standalone data backfill operations (such as the Execution Funnel backfill via `scripts/migrations/backfill_execution_funnel.py` and the Contribution Scores migration via `scripts/migrations/migrate_contribution_scores.py`) into the idempotent `run_migrations.py` framework for automated deployment consistency.

## Knowledge Generation and Memory

The system uses a **Transactive Memory System (TMS)** powered by ChromaDB. This allows agents to:
1. **Store Insights:** Analysts record their findings from one cycle to be retrieved in future cycles.
2. **Share Knowledge:** Different agents can access shared context to ensure consistency.
3. **Validate Theses:** Active trade theses are stored in TMS and re-evaluated during the **Position Audit Cycle**.

## Operational Cycles

1. **Emergency Cycle:** Triggered by a Sentinel. Fast-tracked through the council for immediate action. Emergency closures execute concurrently via true market orders.
2. **Scheduled Cycle:** Runs 3 times per session (at 20%, 62%, and 80% of session duration) to generate daily orders.
3. **Position Audit Cycle:** Regularly reviews open positions against their original entry thesis to decide on early exits.
4. **Reconciliation Cycle:** Syncs the local trade ledger with IBKR records at the end of the day using aggregate quantity matching to accurately group spread legs into thesis groups. It utilizes secure XML parsing (`defusedxml`) and algorithmically optimizes the 1-to-1 time-window matching logic by using `.groupby()` on deterministic keys (`local_symbol`, `action`, `quantity`) rather than computationally expensive full-pass `.iterrows()` loops. To prevent persistent false alerts, it automatically invalidates superseded synthetic (phantom) entries that duplicate verified IBKR trades. Drawdown circuit breaker thresholds are sized to be multi-commodity aware (e.g. 6% halt, 9% panic) to filter out bid-ask widening mark-to-market noise.
5. **System Health Digest Cycle:** Generates a daily post-close JSON summary of system health, agent calibration, and error telemetry without interacting with IBKR or live LLMs.

### Three-Tier Market State
To handle 24/7 trading availability for commodities like Natural Gas (NG) on CME Globex, the system dynamically shifts between three states:
- **ACTIVE:** Full orchestration, strategy evaluation, and active order execution.
- **PASSIVE:** Active cycles are paused. Sentinel surveillance runs continuously. If a catastrophe is detected, a passive emergency cycle is launched (dry run optional) to execute defensive closures.
- **SLEEPING:** The system completely drops connections and performs no active polling or LLM calls (e.g., weekends, maintenance windows, specific holidays).
