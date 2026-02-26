# Real Options — Algorithmic Commodity Futures Trading

An event-driven, multi-agent AI trading system for commodity futures options. Uses a Council of specialized AI analysts with adversarial debate to generate trading decisions, managed by a constitutional compliance framework.

## Architecture: The Federated Cognitive Lattice

The system operates on a tiered, event-driven architecture designed for modularity, fail-safety, and heterogeneous intelligence. The default execution mode is **Multi-Commodity**, where a `MasterOrchestrator` manages isolated `CommodityEngine` instances for each active ticker (e.g., Coffee, Cocoa).

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
    CG -->|Validated| DPS
    DPS -->|Sized Order| OM
    OM -->|Execute| IB

    %% Cache interactions
    Sentinels -.->|Check| SC
    SC -.->|Hit| MS
```

### Core Components

1.  **Master Orchestrator (`trading_bot/master_orchestrator.py`):** The top-level supervisor. Spawns and monitors `CommodityEngine` processes for each active ticker. Manages **Shared Services** (Equity Polling, Macro Research, Post-Close Reconciliation) to prevent API redundancy.
2.  **Commodity Engine (`trading_bot/commodity_engine.py`):** The isolated runtime for a single commodity. Manages its own Sentinels, Council, and Schedule. Ensures strict data isolation.
3.  **Heterogeneous Router (`trading_bot/heterogeneous_router.py`):** Routes LLM requests to the best-fit provider (Gemini, OpenAI, Anthropic, xAI) based on the agent's role.
4.  **Semantic Cache (`trading_bot/semantic_cache.py`):** Caches Council decisions based on market state vectors. Prevents redundant LLM calls.
5.  **Transactive Memory System (`trading_bot/tms.py`):** ChromaDB-based vector store for "institutional memory" across cycles.
6.  **Brier Bridge (`trading_bot/brier_bridge.py`):** Tracks agent accuracy using an Enhanced Probabilistic Brier Score system to weight agent opinions dynamically.
7.  **DSPy Optimizer (`trading_bot/dspy_optimizer.py`):** Offline pipeline that refines agent prompts using historical feedback (BootstrapFewShot).
8.  **Automated Trade Journal (`trading_bot/trade_journal.py`):** Generates structured post-mortem narratives for every closed trade, stored in TMS for future learning.

### Tier 1: Sentinels (`trading_bot/sentinels.py`)
Lightweight monitors that scan 24/7 for specific triggers.
*   **PriceSentinel:** Monitors for rapid price shocks or liquidity gaps.
*   **WeatherSentinel:** Tracks precipitation/temp in key growing regions (via Open-Meteo).
*   **LogisticsSentinel:** Monitors supply chain disruptions via RSS & Gemini Flash.
*   **NewsSentinel:** Scans global news for fundamental shifts.
*   **XSentimentSentinel:** Analyzes real-time social sentiment on X (via xAI).
*   **PredictionMarketSentinel:** Monitors Polymarket odds for geopolitical/macro events via Gamma API.
*   **TopicDiscoveryAgent (`trading_bot/topic_discovery.py`):** Dynamically scans Polymarket for new, relevant topics using Claude Haiku and auto-configures the PredictionMarketSentinel.
*   **MacroContagionSentinel:** Detects cross-asset contagion (DXY shocks, Fed policy shifts, gold/silver correlation).
*   **FundamentalRegimeSentinel:** Determines long-term surplus/deficit regimes.
*   **MicrostructureSentinel:** Monitors order book depth and flow toxicity.

### Tier 2: The Council (`trading_bot/agents.py`)
Specialized LLM personas that analyze grounded data.
*   **Agronomist:** Crop health, weather impact.
*   **Macro Economist:** FX, interest rates, global demand.
*   **Fundamentalist:** Supply/demand balance, COT reports.
*   **Technical Analyst:** Chart patterns, momentum.
*   **Volatility Analyst:** IV rank, term structure, skew.
*   **Geopolitical Analyst:** Trade wars, sanctions, conflict.
*   **Sentiment Analyst:** Crowd psychology, fear/greed.
*   **Inventory Analyst:** Stockpile levels (ICE certified stocks).
*   **Supply Chain Analyst:** Shipping routes, freight rates.

### Tier 3: Decision & Risk
*   **Permabear & Permabull:** Engage in dialectical debate.
*   **Master Strategist:** Synthesizes reports and debate to render a verdict.
*   **Devil's Advocate:** Runs a pre-mortem check.
*   **Compliance Guardian (`trading_bot/compliance.py`):** Deterministic veto power based on risk limits.
*   **Portfolio Risk Guard (`trading_bot/var_calculator.py`):** Calculates portfolio-wide **Full Revaluation Historical Simulation VaR** (95%/99%) using Black-Scholes repricing. Captures gamma risk and non-linearities.
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
    *   **Compliance Guardian** checks VaR limits, margin, and concentration.
8.  **Execution:** `OrderManager` constructs the order and submits to `ib_interface.py`.
9.  **Monitoring:** System monitors positions for P&L, regime shifts, and thesis invalidation.

## Running

### Prerequisites
- Python 3.11+
- Interactive Brokers Gateway (IB Gateway or TWS) running on configured port
- API keys in `.env` (see `.env.example`)

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

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
-   **`commodity_overrides`**: Per-commodity config overrides in `config.json` (e.g., `commodity_overrides.CC`).
-   **`ACTIVE_COMMODITIES`**: Comma-separated list of tickers in `.env` or config (default: `KC,CC`).

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
-   **AI:** Google Gemini (1.5 Pro/Flash), OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet/Haiku), xAI (Grok)
-   **Memory:** ChromaDB (Vector Store)
-   **Data Sources:**
    -   **Market Data:** IBKR, yfinance (History/VaR)
    -   **Weather:** Open-Meteo
    -   **Prediction Markets:** Polymarket (via Gamma API)
    -   **News:** Google News RSS
-   **Dashboard:** Streamlit
-   **Orchestration:** MasterOrchestrator → CommodityEngine
-   **Observability:** Custom logging + Pushover notifications

## License

Proprietary. All rights reserved.
