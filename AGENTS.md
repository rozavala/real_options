# Agent Instructions for Real Options

This document provides instructions and guidelines for AI agents (like Jules) working on the Real Options trading system.

## Principles

1.  **Safety First:** This is a live trading system. Never modify execution logic (e.g., `order_manager.py`, `ib_interface.py`) without thorough testing and understanding of the implications.
2.  **Heterogeneity:** The system is designed to use multiple LLM providers via the `HeterogeneousRouter`. When adding or modifying agent roles, maintain this heterogeneity to prevent algorithmic monoculture.
3.  **Fail-Closed:** All components must fail closed. If an error occurs during a trade cycle, the trade should be blocked by default. See `trading_bot/risk_management.py` and `trading_bot/compliance.py` for examples of this pattern.
4.  **Traceability:** All decisions must be logged. Ensure that any new logic maintains the forensic record in `council_history.csv` and `decision_signals.csv`.

## Core Components

-   **Master Orchestrator (`master_orchestrator.py`):** The top-level supervisor for multi-commodity operations.
-   **Commodity Engine (`trading_bot/commodity_engine.py`):** The runtime for a single commodity.
-   **Heterogeneous Router (`trading_bot/heterogeneous_router.py`):** Routes LLM calls to different providers (Gemini, OpenAI, Anthropic, xAI) based on the agent's role.
-   **Semantic Cache (`trading_bot/semantic_cache.py`):** Caches decisions to avoid redundant API calls when the market state is unchanged.
-   **Sentinels (`trading_bot/sentinels.py`):** Monitor external events and trigger emergency cycles.
-   **Council (`trading_bot/agents.py`):** A collection of specialized agents that debate and decide on trades.
-   **Compliance (`trading_bot/compliance.py`):** The final arbiter of all trades, enforcing risk limits.
-   **TMS (`trading_bot/tms.py`):** The Transactive Memory System, used for sharing knowledge across agents and cycles.

## LLM Routing & Caching

-   **Routing:** Agents do not call LLM APIs directly. They must go through the `HeterogeneousRouter` (or `TradingCouncil` wrapper) which handles provider selection, rate limiting, and fallbacks.
-   **Caching:** Be aware that agent outputs may be cached by `SemanticCache`. If your agent relies on highly real-time data (seconds precision), ensure it is correctly integrated with the Sentinel system which bypasses cache for fresh triggers.

## Knowledge Generation

-   **`.jules/` Directory:** Use this directory to store agent-level knowledge, learnings, and architectural notes.
-   **Readiness Check:** Use `verify_system_readiness.py` to diagnose the system state.

## Operational Procedures

-   **Backtesting:** Use the `backtesting/` directory for strategy validation.
-   **Dashboard:** The Streamlit dashboard (`dashboard.py`) is the primary interface for human monitoring.
