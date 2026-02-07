# Agent Instructions for Real Options

This document provides instructions and guidelines for AI agents (like Jules) working on the Real Options trading system.

## Principles

1. **Safety First:** This is a live trading system. Never modify execution logic (e.g., `order_manager.py`, `ib_interface.py`) without thorough testing and understanding of the implications.
2. **Heterogeneity:** The system is designed to use multiple LLM providers. When adding or modifying agent roles, maintain this heterogeneity to prevent algorithmic monoculture.
3. **Fail-Closed:** All components should fail closed. If an error occurs during a trade cycle, the trade should be blocked by default.
4. **Traceability:** All decisions must be logged. Ensure that any new logic maintains the forensic record in `council_history.csv` and `decision_signals.csv`.

## Core Components

- **Orchestrator (`orchestrator.py`):** The central nervous system. Manages scheduling and event-driven cycles.
- **Sentinels (`trading_bot/sentinels.py`):** Monitor external events and trigger emergency cycles.
- **Council (`trading_bot/agents.py`):** A collection of specialized agents that debate and decide on trades.
- **Compliance (`trading_bot/compliance.py`):** The final arbiter of all trades, enforcing risk limits.
- **TMS (`trading_bot/tms.py`):** The Transactive Memory System, used for sharing knowledge across agents and cycles.

## Knowledge Generation

- **`.jules/` Directory:** Use this directory to store agent-level knowledge, learnings, and architectural notes.
- **Readiness Check:** Use `verify_system_readiness.py` to diagnose the system state.

## Operational Procedures

- **Backtesting:** Use the `backtesting/` directory for strategy validation.
- **Dashboard:** The Streamlit dashboard (`dashboard.py`) is the primary interface for human monitoring.
