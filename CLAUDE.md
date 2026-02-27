# CLAUDE.md - Real Options Trading System

## What This Project Is

Real Options is an event-driven, multi-agent AI trading system for commodity futures options (currently Coffee Arabica / KC on NYBOT). It uses a 4-tier hierarchical decision-making architecture ("Federated Cognitive Lattice") with multiple LLM providers to prevent algorithmic monoculture.

**Production system.** This trades real money via Interactive Brokers. Every change must be treated with the care that implies.

## Session Startup

At the start of every conversation, check open GitHub issues and briefly summarize any that need attention:
```bash
gh issue list --state open --limit 10
```

## Quick Reference

```bash
# Run tests
pytest tests/

# Run a specific test file
pytest tests/test_exit_integration.py

# Run the orchestrator (requires IB Gateway + .env)
python orchestrator.py

# Run the dashboard
streamlit run dashboard.py

# Pre-flight system check
python verify_system_readiness.py

# Run database migrations
bash scripts/run_migrations.sh
```

## Architecture Overview

### 4-Tier Decision Hierarchy

**Tier 1 - Sentinels** (always-on monitors in `trading_bot/sentinels.py`):
- PriceSentinel, WeatherSentinel, LogisticsSentinel, NewsSentinel
- XSentimentSentinel, PredictionMarketSentinel, MicrostructureSentinel
- MacroContagionSentinel, FundamentalRegimeSentinel
- Detect events and trigger emergency or scheduled trading cycles

**Tier 2 - Specialist Analysts** (triggered by sentinels or scheduled cycles, defined in `trading_bot/agents.py`):
- Agronomist, Macro Economist, Fundamentalist, Technical Analyst
- Volatility Analyst, Geopolitical Analyst, Sentiment Analyst
- Each uses research prompts defined in `config.json` under `gemini.personas`

**Tier 3 - Decision Council** (in `trading_bot/agents.py`):
- Permabear (attacks bullish thesis)
- Permabull (defends bullish thesis)
- Master Strategist (renders final decision via weighted voting)
- Devil's Advocate (pre-mortem analysis)

**Tier 4 - Compliance & Execution**:
- ComplianceGuardian (`trading_bot/compliance.py`) - veto power, VaR/margin checks
- Order execution via IB Gateway (`trading_bot/order_manager.py`, `trading_bot/ib_interface.py`)

### Trading Strategies

- Bull Call Spreads (bullish directional)
- Bear Put Spreads (bearish directional)
- Long Straddles (volatility plays)
- Iron Condors (range-bound markets)

## Project Structure

```
real_options/
├── orchestrator.py              # Main event loop (4k lines) - central nervous system
├── dashboard.py                 # Streamlit entry point
├── dashboard_utils.py           # Dashboard helpers
├── config.json                  # Primary configuration (models, thresholds, sentinels)
├── config_loader.py             # Config + env var override system
├── requirements.txt             # Python dependencies (32 packages)
├── pyproject.toml               # Project metadata, pytest config
├── deploy.sh                    # Production deployment script
├── verify_system_readiness.py   # Pre-flight checks
│
├── trading_bot/                 # Core trading system (~49 modules)
│   ├── agents.py                # Council implementation (all analyst/debater agents)
│   ├── sentinels.py             # Sentinel base + implementations
│   ├── order_manager.py         # Order generation, placement, execution
│   ├── compliance.py            # Compliance Guardian, risk validation
│   ├── strategy.py              # Spread definitions
│   ├── heterogeneous_router.py  # Multi-model LLM routing (Gemini/OpenAI/Anthropic/xAI)
│   ├── tms.py                   # Transactive Memory System (ChromaDB vector store)
│   ├── weighted_voting.py       # Master Strategist voting + regime detection
│   ├── brier_scoring.py         # Prediction accuracy tracking
│   ├── risk_management.py       # VaR, margin, position limits
│   ├── position_sizer.py        # Dynamic position sizing
│   ├── drawdown_circuit_breaker.py  # Drawdown protection
│   ├── budget_guard.py          # Capital allocation tracking
│   ├── ib_interface.py          # Interactive Brokers API wrapper
│   ├── connection_pool.py       # IB connection pooling
│   ├── market_data_provider.py  # Quote/level data handling
│   ├── observability.py         # Tracing & health checks
│   ├── state_manager.py         # Atomic state updates (JSON file-based)
│   ├── reconciliation.py        # Trade ledger sync with IBKR
│   ├── notifications.py         # Pushover notifications
│   ├── utils.py                 # Logging, tick rounding, exchange calendars
│   ├── calendars.py             # Trading hours, holiday handling
│   └── ...                      # ~25 more modules (see trading_bot/)
│
├── pages/                       # Streamlit dashboard pages
│   ├── 1_Cockpit.py             # Live operations, system health, emergency controls
│   ├── 2_The_Scorecard.py       # Decision quality, win rates
│   ├── 3_The_Council.py         # Agent reports, consensus visualization
│   ├── 4_Financials.py          # ROI, equity curve, strategy performance
│   ├── 5_Utilities.py           # Log analysis, manual overrides
│   ├── 6_Signal_Overlay.py      # Price action vs signals, trade forensics
│   ├── 7_Brier_Analysis.py      # Prediction accuracy, agent scoring
│   ├── 8_LLM_Monitor.py         # Cost tracking, provider health, latency
│   └── 9_Portfolio.py           # Cross-commodity risk, VaR, engine health
│
├── config/                      # Configuration system
│   ├── commodity_profiles.py    # Extensible commodity profiles (CommodityProfile dataclass)
│   ├── profiles/template.json   # Template for new commodity profiles
│   └── api_costs.json           # LLM token pricing for cost tracking
│
├── backtesting/                 # Backtesting framework
│   ├── simple_backtest.py       # Basic strategy backtesting
│   └── surrogate_models.py      # ML-based price prediction
│
├── scripts/                     # Utility and deployment scripts
│   ├── migrations/              # Idempotent database migrations (001-00x)
│   ├── start_orchestrator.sh    # Service startup
│   ├── verify_deploy.sh         # Post-deployment health checks
│   └── run_migrations.sh        # Migration orchestration
│
├── tests/                       # Test suite (50+ files)
│   ├── test_agents.py           # Council agent tests
│   ├── test_exit_integration.py # Exit logic integration tests
│   ├── test_risk_management.py  # Risk validation tests
│   ├── test_order_manager.py    # Order lifecycle tests
│   ├── test_brier_scoring_new.py# Prediction accuracy tests
│   └── ...                      # Many more test files
│
├── .github/workflows/deploy.yml # CI/CD: auto-deploy on push to main/production
└── .jules/                      # Agent knowledge documentation
```

## Key Principles

### 1. Safety First
This is a **live trading system**. Never modify execution logic (`order_manager.py`, `ib_interface.py`, `compliance.py`) without thorough testing. All components must **fail closed** - if an error occurs during a trade cycle, the trade is blocked by default.

### 2. LLM Heterogeneity
The system deliberately uses multiple LLM providers (Gemini, OpenAI, Anthropic, xAI) to prevent algorithmic monoculture. The `heterogeneous_router.py` handles model routing. When adding or modifying agent roles, maintain this diversity. Model assignments are configured in `config.json` under `model_registry`.

### 3. Traceability
All decisions must be logged. Maintain the forensic record in:
- `data/council_history.csv` - Full forensic record (30+ columns, agent reports, debate text)
- `data/decision_signals.csv` - Lightweight decision log (10 columns)
- `trade_ledger.csv` - Executed trades with fills and P&L

### 4. Configuration over Code
Most tunable parameters live in `config.json`, not hardcoded. Check there before adding constants to source files. Environment variables (loaded from `.env`) override sensitive values like API keys.

## Development Workflow

### Testing
```bash
# Full test suite
pytest tests/

# Specific test file
pytest tests/test_agents.py

# Async tests are auto-configured via pyproject.toml:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"
```

### Dependencies
- Python 3.9+
- Install: `pip install -r requirements.txt`
- Key frameworks: `ib_insync`, `streamlit`, `chromadb`, `google-genai`, `openai`, `anthropic`
- ML stack: `xgboost`, `tensorflow`, `scikit-learn`, `arch`, `pandas-ta`

### Deployment
- **CI/CD**: Push to `main` deploys to DEV, push to `production` deploys to PROD
- **Target**: Digital Ocean droplets via SSH, systemd-managed service
- **Process**: `deploy.sh` handles stop -> log rotation -> install deps -> migrations -> verify -> start
- **Service name**: `trading-bot` (systemd)

### Environment Variables (`.env`)
Required API keys and secrets (never committed):
- `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY` - LLM providers
- `IB_PORT`, `IB_CLIENT_ID` - Interactive Brokers connection
- `FLEX_TOKEN`, `FLEX_QUERY_ID`, `FLEX_POSITIONS_ID` - IBKR Flex reporting
- `PUSHOVER_USER_KEY`, `PUSHOVER_API_TOKEN` - Notifications
- `FRED_API_KEY`, `NASDAQ_API_KEY` - Economic data providers
- `X_BEARER_TOKEN` - Twitter/X API

## Common Patterns

### Adding a New Sentinel
1. Subclass the sentinel base in `trading_bot/sentinels.py`
2. Register it in `orchestrator.py`
3. Add configuration thresholds in `config.json` under `sentinels`
4. Add tests in `tests/`

### Adding a New Analyst Agent
1. Define the persona prompt in `config.json` under `gemini.personas`
2. Wire it into the council flow in `trading_bot/agents.py`
3. Ensure it routes through `heterogeneous_router.py` (maintain model diversity)
4. Update `weighted_voting.py` if it should influence the final decision

### Adding a New Commodity
1. Create a profile using `config/profiles/template.json` as a base
2. Register it in `config/commodity_profiles.py`
3. Update relevant sentinels for commodity-specific data sources

## Data Files (Runtime, Not in Git)

These files are generated at runtime and tracked in `.gitignore`:
- `data/` - All CSV data files, state files, TMS vector store
- `trade_ledger.csv` - Executed trades
- `state.json` / `data/state.json` - Orchestrator persistent state
- `data/drawdown_state.json` - Circuit breaker state
- `*.log` - Log files in `logs/`
- `*.sqlite3` - SQLite databases
- `.env` - Secrets

## Automated Issue-to-PR Pipeline

Issues are automatically created from log errors (`scripts/error_reporter.py`), triaged by Claude (`issue-triage.yml`), and fixed by Claude Code (`claude-fix.yml`). The `claude-fix` label triggers the fix workflow.

**When fixing issues (in CI or locally):**
- Always check `git log --oneline -20` to understand recent changes before modifying code
- Look for related recent commits that may provide context for the issue
- Read the full issue including any triage comments for context

## Things to Watch Out For

- **Tick rounding**: Coffee options have specific tick sizes. Use `utils.py` tick-rounding helpers, not raw floats.
- **Market hours**: The system respects exchange calendars (`calendars.py`). Don't bypass trading hour checks.
- **Async code**: The orchestrator and many modules use `asyncio`. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- **State atomicity**: State updates go through `state_manager.py` for atomic JSON writes. Don't write state files directly.
- **Confidence levels**: Agents use exactly one of: LOW, MODERATE, HIGH, EXTREME. This vocabulary is strict.
- **Regime naming**: Regimes must use harmonized names (bullish, bearish, neutral, range_bound). See the regime harmonization PR (#713) for context.
- **Brier scoring**: Agent accuracy is tracked via Brier scores with reliability multipliers. Changes to agent names or outputs can break scoring continuity.
