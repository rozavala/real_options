# Real Options — Algorithmic Commodity Futures Trading

An event-driven, multi-agent AI trading system for commodity futures options. Uses a Council of specialized AI analysts with adversarial debate to generate trading decisions, managed by a constitutional compliance framework.

## Architecture: The Federated Cognitive Lattice

The system operates on a tiered, event-driven architecture designed for modularity, fail-safety, and heterogeneous intelligence. The default execution mode is **Multi-Commodity**, where a `MasterOrchestrator` manages isolated `CommodityEngine` instances for each active ticker (e.g., Coffee (KC), Cocoa (CC), Natural Gas (NG)).

```mermaid
graph TD
    subgraph "Tier 1: Always-On Sentinels"
        XS[🐦 X Sentiment]
        WS[🌤️ Weather]
        NS[📰 News/RSS]
        PM[🔮 Prediction Market]
        TD[🔍 Topic Discovery]
        MC[🌍 Macro Contagion]
        FR[🏗️ Fundamental Regime]
        LS[🚢 Logistics]
        PS[📊 Price/Microstructure]
    end

    subgraph "Infrastructure Layer"
        MO[👑 Master Orchestrator] --> CE[⚙️ Commodity Engine]
        MO --> SS[🔄 Shared Services (Equity, Macro, VaR)]
        CE --> Sentinels
        CE --> Council
        HR[🔀 Heterogeneous Router]
        SC[🧠 Semantic Cache]
        TMS[💾 Transactive Memory (ChromaDB)]
    end

    subgraph "Tier 2: Specialist Analysts (The Council)"
        AG[🌱 Agronomist]
        MA[💹 Macro Economist]
        FU[📈 Fundamentalist]
        TE[📐 Technical]
        VO[📊 Volatility]
        GE[🌍 Geopolitical]
        SE[🐦 Sentiment]
        IN[📦 Inventory]
        SC_A[🔗 Supply Chain]
    end

    subgraph "Tier 3: Decision & Risk"
        PB[🐻 Permabear]
        PL[🐂 Permabull]
        MS[👑 Master Strategist]
        DA[😈 Devil's Advocate]
        RA[🛡️ AI Risk Agent]
    end

    subgraph "Execution & Risk"
        CG[🛡️ Compliance Guardian]
        VAR[📉 Portfolio Risk Guard]
        DCB[🔌 Drawdown Circuit Breaker]
        DPS[⚖️ Dynamic Sizer]
        OM[⚡ Order Manager]
        IB[Interactive Brokers Gateway]
    end

    XS & WS & NS & PM & MC & FR & LS & PS -->|Trigger| HR
    TD -->|Discover Topics| PM
    HR -->|Route Request| AG & MA & FU & TE & VO & GE & SE & IN & SC_A
    AG & MA & FU & TE & VO & GE & SE & IN & SC_A -->|Reports| PB & PL
    PB & PL -->|Debate| MS
    MS -->|Decision| DA
    DA -->|Approved| CG
    VAR -->|Risk Limits| CG
    DCB -->|Drawdown Limits| CG
    CG -->|Validated| DPS
    DPS -->|Sized Order| OM
    OM -->|Execute| IB

    %% Cache interactions
    Sentinels -.->|Check| SC
    SC -.->|Hit| MS
```

### Core Components

1.  **Master Orchestrator (`trading_bot/master_orchestrator.py`):** The top-level supervisor. Spawns and monitors `CommodityEngine` processes for each active ticker. Manages **Shared Services** (Equity Polling, Macro Research, Post-Close Reconciliation, System Health Digest) to prevent API redundancy.
2.  **Commodity Engine (`trading_bot/commodity_engine.py`):** The isolated runtime for a single commodity. Manages its own Sentinels, Council, and Schedule. Ensures strict data isolation using `ContextVar` to prevent cross-engine contamination of flags (like shutdown states) and dynamically generates commodity-specific configs for dashboard utilities.
3.  **Heterogeneous Router (`trading_bot/heterogeneous_router.py`):** Routes LLM requests to the best-fit provider based on the agent's role (e.g., Gemini Pro for the Geopolitical Analyst and xAI for the Trade Analyst), incorporating multiple fallback providers for resilience.
4.  **Semantic Cache (`trading_bot/semantic_cache.py`):** Caches Council decisions based on market state vectors. Prevents redundant LLM calls.
5.  **Transactive Memory System (`trading_bot/tms.py`):** ChromaDB-based vector store for "institutional memory" across cycles.
6.  **Brier Bridge (`trading_bot/brier_bridge.py`):** Tracks agent accuracy using an Enhanced Probabilistic Brier Score system to weight agent opinions dynamically.
7.  **DSPy Optimizer (`trading_bot/dspy_optimizer.py`):** Offline pipeline that refines agent prompts using historical feedback (BootstrapFewShot).
8.  **Automated Trade Journal (`trading_bot/trade_journal.py`):** Generates structured post-mortem narratives for every closed trade, stored in TMS for future learning.
9.  **System Health Digest (`trading_bot/system_digest.py`):** Generates a daily post-close JSON summary of system health, agent calibration, and error telemetry.
10. **Error Reporter & Telemetry:** A standalone telemetry script (`scripts/error_reporter.py`, decoupled from the orchestrator) scans system logs, uses fingerprinting to deduplicate errors, intelligently filters out transient operational noise (e.g., 429 rate limits, 503 unavailable, lock timeouts), and auto-generates structured GitHub issues to track true system anomalies. Real-time operator alerts (via Pushover) are explicitly severity-gated (e.g., `PriceSentinel` >= 8, `WeatherSentinel` >= 7) to eliminate routine noise.
11. **Three-Tier Market State Resolver (`trading_bot/utils.py`):** Dynamically dictates the state of each commodity (`Active`, `Passive`, `Sleeping`), allowing continuous 24/7 surveillance of extended sessions (e.g., CME Globex overnight) without the risk of generating unnecessary active cycle trades.
12. **System Readiness Verifier (`verify_system_readiness.py`):** A comprehensive pre-flight diagnostic script that runs checks across 27 distinct system components, verifying infrastructure, connections, data fallbacks, agent health, and execution pipelines.
13. **Market Data Diagnostic (`scripts/check_market_data.py`):** A lightweight utility to verify Interactive Brokers data feed status, ensuring `FORCE_DELAYED_DATA` overrides function correctly across DEV and PROD environments by resolving front-month contracts and reporting LIVE/DELAYED quote status.

### Tier 1: Sentinels (`trading_bot/sentinels.py`)
Lightweight monitors that scan 24/7 for specific triggers.
*   **PriceSentinel:** Monitors for rapid price shocks or liquidity gaps.
*   **WeatherSentinel:** Tracks precipitation/temp in key growing regions (via Open-Meteo) using energy-aware stages for non-agricultural commodities.
*   **LogisticsSentinel:** Monitors supply chain disruptions via RSS & Gemini Flash.
*   **NewsSentinel:** Scans global news for fundamental shifts.
*   **XSentimentSentinel:** Analyzes real-time social sentiment on X (via xAI) with automated multi-level circuit breakers for API cost control.
*   **PredictionMarketSentinel:** Monitors Polymarket odds for geopolitical/macro events via Gamma API.
*   **TopicDiscoveryAgent (`trading_bot/topic_discovery.py`):** Dynamically scans Polymarket for new, relevant topics using Claude Haiku (with bulk fetch, commodity filtering, and state pruning) and auto-configures the PredictionMarketSentinel.
*   **MacroContagionSentinel:** Detects cross-asset contagion (DXY shocks, Fed policy shifts, gold/silver correlation).
*   **FundamentalRegimeSentinel:** Determines long-term surplus/deficit regimes.
*   **MicrostructureSentinel:** Monitors order book depth and flow toxicity.

### Tier 2: The Council (`trading_bot/agents.py`)
Specialized LLM personas that analyze grounded data.
*   **Agronomist:** Crop health, weather impact.
*   **Macro Economist:** FX, interest rates, global demand.
*   **Fundamentalist:** Supply/demand balance, COT reports.
*   **Technical Analyst:** Chart patterns, momentum.
*   **Volatility Analyst:** IV rank, term structure, skew. Provides a non-directional (expensive/cheap options) signal that is excluded from the directional weighted score.
*   **Geopolitical Analyst:** Trade wars, sanctions, conflict.
*   **Sentiment Analyst:** Crowd psychology, fear/greed.
*   **Inventory Analyst:** Stockpile levels (ICE certified stocks).
*   **Supply Chain Analyst:** Shipping routes, freight rates.

### Tier 3: Decision & Risk
*   **Permabear & Permabull:** Engage in dialectical debate. Model assignment and debate order are randomized to prevent anchoring bias.
*   **Master Strategist:** Synthesizes reports and debate to render a verdict.
*   **Devil's Advocate:** Runs a pre-mortem check.
*   **Compliance Guardian (`trading_bot/compliance.py`):** Deterministic veto power based on risk limits.
*   **Portfolio Risk Guard (`trading_bot/var_calculator.py`):** Calculates portfolio-wide **Full Revaluation Historical Simulation VaR** (95%/99%) using Black-Scholes repricing. Captures gamma risk and non-linearities.
*   **Drawdown Circuit Breaker (`trading_bot/drawdown_circuit_breaker.py`):** Monitors intraday P&L drops from the previous day's close and enforces multi-commodity-aware halt/panic thresholds (e.g., 3% warning, 6% halt, 9% panic) to protect the portfolio from bid-ask widening noise.
*   **AI Risk Agent:** Embedded in the Risk Guard. **L1 Interpreter** provides narrative risk analysis, and **L2 Scenario Architect** generates stress scenarios (e.g., "Commodity Crash", "IV Spike").
*   **Dynamic Position Sizer (`trading_bot/position_sizer.py`):** Calculates trade size based on conviction (Kelly Criterion adjusted by Volatility).

## Information Flow

1.  **Ingestion:** Sentinels detect a signal (e.g., "Frost in Brazil" or "Polymarket Odds Shift").
2.  **Trigger:** An **Emergency Cycle** is initiated. The `TriggerDeduplicator` prevents storming.
3.  **Routing/Caching:** Checks `SemanticCache` for recent similar decisions.
4.  **Analysis:** `HeterogeneousRouter` dispatches prompts to Agents.
5.  **Synthesis:** Agents submit reports to TMS. Permabear/Permabull debate. Master Strategist decides.
6.  **Validation:** Devil's Advocate challenges.
7.  **Compliance & Risk:**
    *   **Portfolio Risk Guard** calculates new VaR impact.
    *   **Compliance Guardian** checks VaR limits, margin, and concentration. Also enforces a **Conviction Gate** (suppressing low-conviction signals) and **Sentinel IC Suppression** (blocking short premium positions during sentinel-triggered high-volatility events).
8.  **Execution:** `OrderManager` immediately executes closures for contradicted positions to prevent quantity aggregation race conditions. Multi-leg positions (e.g., spreads) are closed atomically via single BAG (combo) orders to prevent orphaned legs, falling back to sequential closures only if necessary. All exits use limit orders with adaptive price walking, dropping to a profile-driven timeout for market order fallbacks (e.g., KC=90s, CC=120s, NG=60s). Stale-close fallbacks are automatically skipped if the primary close succeeds. New orders are constructed and submitted to `ib_interface.py` using a Hybrid Tick/Percentage Liquidity Filter. A **Contract Overlap Guard** blocks proposed orders with opposing-direction overlaps against active theses to prevent invisible IBKR netting, while allowing same-direction additive overlaps. Catastrophe stops are placed dynamically, gated by account Net Liquidation Value (NLV) thresholds to fail-closed without stop protection if margin is insufficient. Emergency closures execute as concurrent market orders.
9.  **Monitoring:** System monitors positions at the *thesis level* (grouping spread legs) for P&L, regime shifts, and thesis invalidation. Reconciliation uses aggregate quantity matching and securely parses IBKR Flex Queries via `defusedxml` to prevent XXE attacks. The reconciliation loop is algorithmically optimized via vectorized pandas `.groupby()` chunking on deterministic characteristics (like symbol and quantity) to map real trades to local synthetic state with $O(1)$ precision, while automatically invalidating superseded synthetic entries to eliminate false-positive phantom position alerts. During `Passive` market hours, only emergency surveillance runs and passive emergency closures can be triggered. To ensure execution reliability, the `IBConnectionPool` explicitly disconnects failed API handshakes to prevent CLOSE-WAIT TCP socket accumulation.
10. **Digest:** Post-close `System Health Digest` generation summarizes multi-commodity system state and errors for observability.

## Running

### Prerequisites
- Python 3.11+
- Interactive Brokers Gateway (IB Gateway or TWS) running on configured port
- API keys in `.env` (see `.env.example`)

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run Pre-flight Checks
python verify_system_readiness.py

# Default: Multi-Commodity Mode (MasterOrchestrator)
# Spawns isolated CommodityEngine processes for all active tickers defined in config
python orchestrator.py

# Single commodity (Debug/Legacy mode)
python orchestrator.py --commodity KC

# Dashboard
streamlit run dashboard.py
```

### Configuration
-   **`config.json`**: Primary configuration (model registry, thresholds, sentinel intervals).
-   **`commodity_overrides`**: Per-commodity runtime config overrides in `config.json` (schedule, strategy, risk).
-   **`config/profiles/<ticker>.json`**: Commodity-specific profiles (market states, contract specs, growing regions).
-   **`ACTIVE_COMMODITIES`**: Comma-separated list of tickers in `.env` or config (default: `KC,CC,NG`).

### Deployment
```bash
# DEV: Push to main → auto-deploys via GitHub Actions
git push origin main

# PROD: Push to production branch
git push origin production
```

## Tech Stack

-   **Runtime:** Python 3.11+, `asyncio`
-   **Execution:** Interactive Brokers Gateway (`ib_insync`)
-   **AI:** Google Gemini (1.5 Pro/Flash), OpenAI (GPT-5.2/o3), Anthropic (Claude 4.6 Sonnet/Haiku), xAI (Grok 4.1), Perplexity (Sonar)
-   **Memory:** ChromaDB (Vector Store)
-   **Data Sources:**
    -   **Market Data:** IBKR, yfinance (History/VaR)
    -   **Weather:** Open-Meteo
    -   **Prediction Markets:** Polymarket (via Gamma API)
    -   **News:** Google News RSS
    -   **Search:** Perplexity Sonar (with Gemini fallback)
-   **Dashboard:** Streamlit
-   **Orchestration:** MasterOrchestrator (centralized shared services: Equity, Macro, Post-Close Reconciliation) → CommodityEngine
-   **Observability:** Custom logging + Pushover notifications

## License

Proprietary. All rights reserved.
