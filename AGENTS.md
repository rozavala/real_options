# Agent Instructions for Real Options

This document provides instructions and guidelines for AI agents (like Jules) working on the Real Options trading system.

## Principles

1.  **Safety First:** This is a live trading system. Never modify execution logic (e.g., `order_manager.py`, `ib_interface.py`) without thorough testing.
2.  **Heterogeneity:** The system uses multiple LLM providers via `HeterogeneousRouter`. Maintain provider diversity.
3.  **Fail-Closed:** All components must fail closed. If an error occurs, the trade is blocked.
4.  **Traceability:** All decisions must be logged in `council_history.csv` and `decision_signals.csv`.
5.  **State Management:** State reconciliation must group raw IBKR legs into semantic thesis groups, use aggregate quantities to avoid false-positive orphan detection, and accurately record all lots in multi-lot combo fills. The reconciliation process explicitly detects and idempotent-cancels superseded synthetic (phantom) entries to eliminate false-positive phantom position alerts. For performance, it must utilize vectorized pandas `.groupby()` on deterministic keys (e.g., `local_symbol`, `action`, `quantity`) rather than computationally expensive `iterrows()` loops for precise 1-to-1 time-window matching. Multi-commodity operations require strict isolation via `clientId` filtering (to prevent order cancellation across engines) and IBKR symbol prefix matching (e.g., `LN1` for Natural Gas weekly options) to prevent cross-engine order corruption and false untracked positions.
6.  **Market States:** The system operates in a Three-Tier Market State model (`Active`, `Passive`, `Sleeping`). Agents must be aware that during `Passive` mode (e.g., CME Globex overnight sessions), normal cycles are skipped but emergency surveillance continues, allowing for passive emergency closes if triggers are met.

## Core Components

-   **Master Orchestrator (`trading_bot/master_orchestrator.py`):** The top-level supervisor for multi-commodity operations. Manages shared services (Equity, Macro, VaR).
-   **Commodity Engine (`trading_bot/commodity_engine.py`):** The isolated runtime for a single commodity (ticker).
-   **Heterogeneous Router (`trading_bot/heterogeneous_router.py`):** Routes LLM calls to different providers. Dynamically routes specific agent roles to optimized providers (e.g., Gemini Pro for the Geopolitical Analyst and xAI for the Trade Analyst), incorporating multiple fallback providers for resilience.
-   **Semantic Cache (`trading_bot/semantic_cache.py`):** Caches decisions to avoid redundant API calls.
-   **Sentinels (`trading_bot/sentinels.py`):** Monitor external events (Price, Weather, News, Polymarket, Macro). Must implement graceful rate-limiting and API spend-cap circuit breakers to control costs autonomously.
-   **Council (`trading_bot/agents.py`):** Specialized agents that debate and decide on trades.
-   **Compliance (`trading_bot/compliance.py`):** Enforces risk limits, "dead checks", the "conviction gate" (suppressing weak directional signals), and "Sentinel IC suppression" (blocking Iron Condors during sentinel-triggered high-vol events).
-   **Portfolio Risk Guard (`trading_bot/var_calculator.py`):** Calculates portfolio-wide VaR (95%/99%) and runs the **AI Risk Agent** for narrative analysis and stress testing.
-   **Drawdown Circuit Breaker (`trading_bot/drawdown_circuit_breaker.py`):** Monitors intraday P&L drops from the previous day's close and enforces multi-commodity-aware halt/panic thresholds (e.g., 6% halt, 9% panic).
-   **TMS (`trading_bot/tms.py`):** Transactive Memory System for institutional memory.

## Infrastructure Agents

-   **Topic Discovery Agent (`trading_bot/topic_discovery.py`):** Scans Prediction Markets (Polymarket) for new topics using Claude Haiku. Auto-configures `PredictionMarketSentinel`.
-   **Risk Agent (Embedded in `var_calculator.py`):**
    -   **L1 Interpreter:** Explains VaR metrics in plain English.
    -   **L2 Scenario Architect:** Generates plausible stress scenarios based on sentinel intelligence.

## LLM Routing & Caching

-   **Routing:** Agents must use `HeterogeneousRouter`. Do not call LLM APIs directly unless authorized (e.g., Sentinels with specific provider needs).
-   **Caching:** Be aware of `SemanticCache`. Fresh triggers bypass cache via specific flags.

## Knowledge Generation

-   **`.jules/` Directory:** Store agent-level knowledge and architectural notes here.
-   **Readiness Check:** Use `verify_system_readiness.py` to diagnose system state.

## Learning & Optimization

-   **Prompt Optimization (DSPy):** The system uses `trading_bot/dspy_optimizer.py` to optimize agent prompts. This offline process analyzes `enhanced_brier.json` to find effective few-shot examples (BootstrapFewShot) that improve reasoning accuracy. Metrics evaluated are directional accuracy (BULLISH/BEARISH matches) and abstention rate (NEUTRAL predictions).
-   **Trade Journal:** Every closed trade triggers a post-mortem analysis (`trading_bot/trade_journal.py`). This generates a "Key Lesson" which is stored in TMS and retrieved via **Reflexion** during future cycles to prevent repeating mistakes.
-   **Reflexion:** Agents query the TMS for past mistakes ("Self-Critique") before finalizing their analysis. Error states from API failures during research are explicitly short-circuited to prevent poisoned-state feedback loops where the LLM analyzes the error text as legitimate evidence.

## Operational Procedures

-   **Backtesting:** Use `backtesting/` directory.
-   **Dashboard:** Streamlit dashboard (`dashboard.py`) is the primary interface.
-   **Observability/Telemetry:** Error telemetry and log parsing are handled out-of-band by the `scripts/error_reporter.py` script. Do not add logic for agents to parse their own system-level execution exceptions. The system filters transient noise (like 429 rate limits, lock timeouts, and 503 unavailability errors) and auto-generates structured GitHub issues.
-   **Market Data Diagnostic:** Use `scripts/check_market_data.py` to quickly verify if the Interactive Brokers data feed is LIVE or DELAYED.
