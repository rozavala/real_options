# Agent Instructions for Real Options

This document provides instructions and guidelines for AI agents (like Jules) working on the Real Options trading system.

## Principles

1.  **Safety First:** This is a live trading system. Never modify execution logic (e.g., `order_manager.py`, `ib_interface.py`) without thorough testing.
2.  **Heterogeneity:** The system uses multiple LLM providers via `HeterogeneousRouter`. Maintain provider diversity.
3.  **Fail-Closed:** All components must fail closed. If an error occurs, the trade is blocked.
4.  **Traceability:** All decisions must be logged in `council_history.csv` and `decision_signals.csv`.

## Core Components

-   **Master Orchestrator (`trading_bot/master_orchestrator.py`):** The top-level supervisor for multi-commodity operations. Manages shared services (Equity, Macro, VaR).
-   **Commodity Engine (`trading_bot/commodity_engine.py`):** The isolated runtime for a single commodity (ticker).
-   **Heterogeneous Router (`trading_bot/heterogeneous_router.py`):** Routes LLM calls to different providers (Gemini, OpenAI, Anthropic, xAI).
-   **Semantic Cache (`trading_bot/semantic_cache.py`):** Caches decisions to avoid redundant API calls.
-   **Sentinels (`trading_bot/sentinels.py`):** Monitor external events (Price, Weather, News, Polymarket, Macro).
-   **Council (`trading_bot/agents.py`):** Specialized agents that debate and decide on trades.
-   **Compliance (`trading_bot/compliance.py`):** Enforces risk limits and "dead checks".
-   **Portfolio Risk Guard (`trading_bot/var_calculator.py`):** Calculates portfolio-wide VaR (95%/99%) and runs the **AI Risk Agent** for narrative analysis and stress testing.
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

-   **Prompt Optimization (DSPy):** The system uses `trading_bot/dspy_optimizer.py` to optimize agent prompts. This offline process analyzes `council_history.csv` to find effective few-shot examples (BootstrapFewShot) that improve reasoning accuracy.
-   **Trade Journal:** Every closed trade triggers a post-mortem analysis (`trading_bot/trade_journal.py`). This generates a "Key Lesson" which is stored in TMS and retrieved via **Reflexion** during future cycles to prevent repeating mistakes.
-   **Reflexion:** Agents query the TMS for past mistakes ("Self-Critique") before finalizing their analysis.

## Operational Procedures

-   **Backtesting:** Use `backtesting/` directory.
-   **Dashboard:** Streamlit dashboard (`dashboard.py`) is the primary interface.
