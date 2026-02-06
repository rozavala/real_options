# Mission Control: System Architecture and Knowledge

This document provides a detailed breakdown of how the Mission Control system operates.

## Architecture: The Federated Cognitive Lattice

The system is organized into a tiered architecture:

### Tier 1: Always-On Sentinels
Sentinels are lightweight monitors that observe external data sources and trigger the decision pipeline when significant events occur.
- **PriceSentinel:** Monitors for rapid price movements.
- **WeatherSentinel:** Tracks weather patterns in key commodity regions (e.g., Brazil for Coffee).
- **LogisticsSentinel:** Monitors RSS feeds for strikes, port closures, or canal disruptions.
- **NewsSentinel:** Scans news feeds for fundamental shifts.
- **XSentimentSentinel:** Analyzes social media sentiment on X (formerly Twitter).
- **PredictionMarketSentinel:** Monitors Polymarket and other prediction markets for geopolitical or macro events.

### Tier 2: Specialist Analysts (The Council)
When a sentinel triggers, a Council of specialized AI agents is convened.
- **Agronomist:** Evaluates crop conditions and weather impacts.
- **Macro Economist:** Assesses global economic trends and currency impacts.
- **Fundamentalist:** Focuses on supply/demand balance and inventory reports.
- **Technical Analyst:** Interprets chart patterns and price action.
- **Volatility Analyst:** Analyzes implied volatility and options pricing.
- **Geopolitical Analyst:** Evaluates the impact of international relations and conflicts.
- **Sentiment Analyst:** Gauges the market mood from social and news sources.

### Tier 3: Decision Council
A set of agents that synthesize the analysts' reports.
- **Permabear:** Attacks the bullish thesis.
- **Permabull:** Defends the bullish thesis.
- **Master Strategist:** Weighs all evidence and makes the final directional decision.
- **Devil's Advocate:** Performs a pre-mortem to identify risks in the master strategy.

### Tier 4: Compliance and Risk Management
The **Compliance Guardian** has final veto power. It checks:
- Position sizing relative to account value.
- Value at Risk (VaR) limits.
- Margin requirements.
- Market liquidity.

## Knowledge Generation and Memory

The system uses a **Transactive Memory System (TMS)** powered by ChromaDB. This allows agents to:
1. **Store Insights:** Analysts record their findings from one cycle to be retrieved in future cycles.
2. **Share Knowledge:** Different agents can access shared context to ensure consistency.
3. **Validate Theses:** Active trade theses are stored in TMS and re-evaluated during the **Position Audit Cycle**.

## Operational Cycles

1. **Emergency Cycle:** Triggered by a sentinel. Fast-tracked through the council for immediate action.
2. **Scheduled Cycle:** Runs at specific times (e.g., 9:00 AM ET) to generate daily orders.
3. **Position Audit Cycle:** Regularly reviews open positions against their original entry thesis to decide on early exits.
4. **Reconciliation Cycle:** Syncs the local trade ledger with IBKR records at the end of the day.
