# Automated Coffee Options Trading Bot

This repository contains the source code for an automated trading bot designed to trade coffee futures options (Symbol: KC) on the NYBOT exchange. The bot uses a prediction API to generate trading signals and executes complex option strategies based on those signals.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Configuration](#configuration)
  - [Anatomy of `config.json`](#anatomy-of-configjson)
- [How to Run](#how-to-run)
  - [1. Run the Prediction API](#1-run-the-prediction-api)
  - [2. Run the Main Orchestrator](#2-run-the-main-orchestrator)
- [Core Modules](#core-modules)

---

## Architecture Overview

The system is designed with a modular architecture, orchestrated by a central scheduling script. The key components are:

1.  **Orchestrator (`orchestrator.py`)**: The heart of the application. It runs on a schedule and coordinates all other components, including data pulling, trade execution, and performance analysis.

2.  **Data Puller (`coffee_factors_data_pull_new.py`)**: A script that gathers a wide range of data (market prices, economic indicators, weather data) from various APIs (Yahoo Finance, FRED, etc.) and consolidates it into a single CSV file.

3.  **Prediction API (`prediction_api.py`)**: A FastAPI-based web service that accepts the consolidated data, simulates a prediction job, and returns mock price change predictions.

4.  **Trading Bot (`trading_bot/`)**: The core package containing the logic for trade execution. It includes modules for:
    - **IB Interface (`ib_interface.py`)**: Interacting with the Interactive Brokers TWS/Gateway.
    - **Signal Generation (`signal_generator.py`)**: Converting API predictions into actionable trading signals.
    - **Strategy (`strategy.py`)**: Implementing option strategies like Bull Call Spreads and Bear Put Spreads.
    - **Risk Management (`risk_management.py`)**: Aligning positions with signals and monitoring open positions for stop-loss or take-profit conditions.

5.  **Position Monitor (`position_monitor.py`)**: A separate, long-running process managed by the orchestrator. It is responsible for the intraday monitoring of open positions for risk.

## Getting Started

### Prerequisites

- Python 3.10+
- An active Interactive Brokers TWS or Gateway session.
- API keys for FRED and Nasdaq Data Link (to be placed in `config.json`).
- A Pushover account for notifications (optional).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    The project uses standard Python packaging. It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: The original project also mentioned `uv`, a fast Python package installer. If you have `uv` installed, you can use `uv pip sync requirements.txt` for a potentially faster installation.)*

3.  **Configure the bot:**
    Rename the `config.example.json` to `config.json` (or create your own) and fill in the required values, especially your API keys and connection settings. See the [Configuration](#configuration) section for details.

## Configuration

The application's behavior is controlled by the `config.json` file.

### Anatomy of `config.json`

-   `connection`: Settings for connecting to IB TWS/Gateway (`host`, `port`, `clientId`).
-   `symbol`, `exchange`: Defines the primary asset to trade (e.g., "KC" on "NYBOT").
-   `strategy`:
    -   `quantity`: The number of contracts to trade for each order.
-   `signal_thresholds`:
    -   `bullish`/`bearish`: The numerical thresholds from the API prediction to trigger a signal.
-   `strategy_tuning`:
    -   Parameters to fine-tune strategy construction, such as `spread_width_usd` and `slippage_usd_per_contract`.
-   `risk_management`:
    -   `check_interval_seconds`: How often the position monitor checks P&L.
    -   `stop_loss_pct`/`take_profit_pct`: The P&L percentage thresholds to close a position.
-   `notifications`:
    -   `enabled`: Set to `true` to enable Pushover notifications.
    -   `pushover_user_key`/`pushover_api_token`: Your Pushover credentials.
-   `fred_api_key`, `nasdaq_api_key`: Your API keys for data services.
-   `weather_stations`: A dictionary of locations for which to pull weather data.
-   `fred_series`, `yf_series_map`: Mappings of economic and market data series to fetch.
-   `final_column_order`: Specifies the column order for the final consolidated data CSV.

## How to Run

### Run the Main Orchestrator

This is the main entry point for the bot. It will run continuously, executing tasks based on its schedule.

```bash
python orchestrator.py
```

The orchestrator will then handle starting and stopping the position monitor and executing the daily trading cycle automatically.

## Core Modules

-   **`orchestrator.py`**: Main entry point. Schedules all tasks.
-   **`config_loader.py`**: Loads the `config.json` file.
-   **`trading_bot/logging_config.py`**: Sets up centralized logging.
-   **`notifications.py`**: Handles sending Pushover notifications.
-   **`coffee_factors_data_pull_new.py`**: Fetches and validates all external data.
-   **`trading_bot/`**: The core trading logic package.
    -   `main.py`: Orchestrates a single trading cycle.
    -   `ib_interface.py`: Low-level interaction with the IB API.
    -   `signal_generator.py`: Converts API results to trade signals.
    -   `strategy.py`: Defines option strategies (e.g., Bull Call Spread).
    -   `risk_management.py`: Manages position alignment and P&L monitoring.
    -   `utils.py`: Helper functions (e.g., Black-Scholes pricing).
-   **`position_monitor.py`**: Standalone script for intraday risk monitoring.
-   **`performance_analyzer.py`**: Reads the trade ledger and generates a daily performance report.
-   **`trade_ledger.csv`**: A CSV file where all filled trades are logged.