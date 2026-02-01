# Archived: ML Signal Pipeline

**Removed:** February 2026
**Reason:** Coffee-specific; blocking commodity-agnostic operation
**Replaced by:** `trading_bot/market_data_provider.py` (IBKR-native)

## Contents
- `coffee_factors_data_pull_new.py` — KC-specific data fetcher
- `inference.py` — 10-model ensemble (Transformer + XGBoost + blender)
- `model_signals.py` — ML signal CSV logger
- `models/` — Pre-trained model binaries
- `test_data_pull.py` — Tests for data pull
- `model_signals_final.csv` — Final ML signal history

## Why Removed
1. Models trained exclusively on coffee history (30 coffee-specific features)
2. Retraining for a new commodity would cost ~$30K
3. Council regularly overrode ML direction
4. Emergency cycles proved the system works without ML
5. Removed ~500MB of dependencies (tensorflow, xgboost, arch)
