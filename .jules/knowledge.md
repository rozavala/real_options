# Real Options: System Architecture and Knowledge

This document provides a detailed breakdown of how the Real Options system operates.

## Architecture: The Federated Cognitive Lattice

The system is organized into a tiered architecture:

### Tier 1: Always-On Sentinels
Sentinels are lightweight monitors that observe external data sources and trigger the decision pipeline when significant events occur.
- **PriceSentinel:** Monitors for rapid price movements or liquidity gaps.
- **WeatherSentinel:** Tracks weather patterns in key commodity regions (e.g., Brazil for Coffee).
- **LogisticsSentinel:** Monitors RSS feeds for strikes, port closures, or canal disruptions.
- **NewsSentinel:** Scans news feeds for fundamental shifts.
- **XSentimentSentinel:** Analyzes social media sentiment on X (via xAI).
- **PredictionMarketSentinel:** Monitors Polymarket odds for geopolitical or macro events.
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
- **Compliance Guardian:** The final arbiter of all trades, enforcing risk limits (VaR, Margin, etc.).
- **Dynamic Position Sizer:** Calculates optimal trade size.
- **Order Manager:** Queues and executes orders via Interactive Brokers.

## Infrastructure

- **Master Orchestrator:** Manages multi-commodity instances.
- **Commodity Engine:** Runs the per-commodity loop.
- **Heterogeneous Router:** Dispatches LLM calls to Gemini, OpenAI, Anthropic, or xAI.
- **Semantic Cache:** Caches decisions to optimize costs and latency.
- **DSPy Optimizer:** Offline pipeline that refines agent prompts using historical feedback (BootstrapFewShot).

## Knowledge Generation and Memory

The system uses a **Transactive Memory System (TMS)** powered by ChromaDB. This allows agents to:
1. **Store Insights:** Analysts record their findings from one cycle to be retrieved in future cycles.
2. **Share Knowledge:** Different agents can access shared context to ensure consistency.
3. **Validate Theses:** Active trade theses are stored in TMS and re-evaluated during the **Position Audit Cycle**.

## Operational Cycles

1. **Emergency Cycle:** Triggered by a Sentinel. Fast-tracked through the council for immediate action.
2. **Scheduled Cycle:** Runs 3 times per session (at 20%, 62%, and 80% of session duration) to generate daily orders.
3. **Position Audit Cycle:** Regularly reviews open positions against their original entry thesis to decide on early exits.
4. **Reconciliation Cycle:** Syncs the local trade ledger with IBKR records at the end of the day.
