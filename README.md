# Real Options — Algorithmic Commodity Futures Trading

An event-driven, multi-agent AI trading system for commodity futures options. Uses a Council of specialized AI analysts with adversarial debate to generate trading decisions, managed by a constitutional compliance framework.

## Architecture: The Federated Cognitive Lattice

The system operates on a tiered, event-driven architecture designed for modularity, fail-safety, and heterogeneous intelligence.

```mermaid
graph TD
    subgraph "Tier 1: Always-On Sentinels"
        XS[🐦 X Sentiment]
        WS[🌤️ Weather]
        NS[📰 News/RSS]
        PS[📊 Microstructure]
        PM[🔮 Prediction Market]
        MC[🌍 Macro Contagion]
        FR[🏗️ Fundamental Regime]
        LS[🚢 Logistics]
    end

    subgraph "Infrastructure Layer"
        MO[👑 Master Orchestrator] --> CE[⚙️ Commodity Engine]
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

    subgraph "Tier 3: Decision Council"
        PB[🐻 Permabear]
        PL[🐂 Permabull]
        MS[👑 Master Strategist]
        DA[😈 Devil's Advocate]
    end

    subgraph "Execution & Risk"
        CG[🛡️ Compliance Guardian]
        DPS[⚖️ Dynamic Sizer]
        OM[⚡ Order Manager]
        IB[Interactive Brokers Gateway]
    end

    XS & WS & NS & PS & PM & MC & FR & LS -->|Trigger| HR
    HR -->|Route Request| AG & MA & FU & TE & VO & GE & SE & IN & SC_A
    AG & MA & FU & TE & VO & GE & SE & IN & SC_A -->|Reports| PB & PL
    PB & PL -->|Debate| MS
    MS -->|Decision| DA
    DA -->|Approved| CG
    CG -->|Validated| DPS
    DPS -->|Sized Order| OM
    OM -->|Execute| IB

    %% Cache interactions
    Sentinels -.->|Check| SC
    SC -.->|Hit| MS
```

### Core Components

1.  **Master Orchestrator (`master_orchestrator.py`):** The top-level supervisor. Spawns and manages `CommodityEngine` instances for each active commodity (e.g., Coffee, Cocoa). Handles shared services like Equity Polling and Macro Research.
2.  **Commodity Engine (`trading_bot/commodity_engine.py`):** The runtime for a single commodity. Manages its own Sentinels, Council, and Schedule. Ensures strict data isolation.
3.  **Heterogeneous Router (`trading_bot/heterogeneous_router.py`):** Routes LLM requests to the best-fit provider (Gemini, OpenAI, Anthropic, xAI) based on the agent's role (e.g., 'Agronomist' uses Gemini Pro for large context, 'Volatility' uses xAI for reasoning).
4.  **Semantic Cache (`trading_bot/semantic_cache.py`):** Caches Council decisions based on market state vectors (Price/Vol/Sentiment/Regime). Prevents redundant LLM calls when the market hasn't materially changed.
5.  **Transactive Memory System (`trading_bot/tms.py`):** A ChromaDB-based vector store that allows agents to store and retrieve insights across cycles, enabling "institutional memory."

### Tier 1: Sentinels (`trading_bot/sentinels.py`)
Lightweight monitors that scan 24/7 for specific triggers.
*   **PriceSentinel:** Monitors for rapid price shocks or liquidity gaps.
*   **WeatherSentinel:** Tracks precipitation/temp in key growing regions (via Open-Meteo).
*   **LogisticsSentinel:** Monitors supply chain disruptions (ports, canals) via RSS.
*   **NewsSentinel:** Scans global news for fundamental shifts.
*   **XSentimentSentinel:** Analyzes real-time social sentiment on X (via xAI).
*   **PredictionMarketSentinel:** Monitors Polymarket odds for geopolitical/macro events.
*   **MacroContagionSentinel:** Detects cross-asset contagion (DXY, Gold, etc.).
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
*   **Permabear & Permabull:** Engage in dialectical debate to stress-test the thesis.
*   **Master Strategist:** Synthesizes all reports and the debate to render a verdict (Direction + Confidence).
*   **Devil's Advocate:** Runs a pre-mortem ("Assume this trade failed. Why?") to identify blind spots.
*   **Compliance Guardian (`trading_bot/compliance.py`):** Deterministic veto power. Checks VaR, margin, concentration, and blacklist.
*   **Dynamic Position Sizer (`trading_bot/position_sizer.py`):** Calculates trade size based on conviction (Kelly Criterion adjusted by Volatility).

## Information Flow

1.  **Ingestion:** Sentinels detect a signal (e.g., "Frost in Brazil").
2.  **Trigger:** An **Emergency Cycle** is initiated. The `TriggerDeduplicator` prevents storming.
3.  **Routing/Caching:** The system checks the `SemanticCache`. If a similar event occurred recently in the same regime, the cached decision is reused.
4.  **Analysis:** If fresh analysis is needed, the `HeterogeneousRouter` dispatches prompts to specialized Agents. Agents use `GroundedDataPacket` (web search results) to form opinions.
5.  **Synthesis:** Agents submit reports to the TMS. The Permabear/Permabull debate the findings. The Master Strategist issues a decision.
6.  **Validation:** The Devil's Advocate challenges the decision.
7.  **Compliance:** The Compliance Guardian validates the trade against risk limits.
8.  **Execution:** The `OrderManager` (`trading_bot/order_manager.py`) constructs the order (e.g., Bull Call Spread) and submits it to `ib_interface.py`.
9.  **Monitoring:** The system monitors the position for P&L triggers, regime shifts (via `check_iron_condor_theses`), or thesis invalidation.

## Running

### Prerequisites
- Python 3.11+
- Interactive Brokers Gateway (IB Gateway or TWS) running on configured port
- API keys in `.env` (see `.env.example`)

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Default: MasterOrchestrator (all active commodities in one process)
python orchestrator.py

# Single commodity (legacy mode)
python orchestrator.py --commodity KC

# Force legacy mode via environment
LEGACY_MODE=true python orchestrator.py

# Dashboard
streamlit run dashboard.py
```

### Configuration
- **`config.json`**: Primary configuration (model registry, thresholds, sentinel intervals)
- **`ACTIVE_COMMODITIES`**: Comma-separated list of tickers in `.env` (default: `KC,CC`)
- **`LEGACY_MODE=true`**: Disables MasterOrchestrator, runs a single CommodityEngine per process
- **`commodity_overrides`**: Per-commodity config overrides in `config.json` (e.g., `commodity_overrides.CC`)

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
-   **AI:** Google Gemini (1.5 Pro/Flash), OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), xAI (Grok)
-   **Memory:** ChromaDB (Vector Store)
-   **Dashboard:** Streamlit
-   **Orchestration:** MasterOrchestrator → CommodityEngine (`master_orchestrator.py`, `commodity_engine.py`)
-   **Observability:** Custom logging + Pushover notifications

## License

Proprietary. All rights reserved.
