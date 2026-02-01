"""
Backtesting Package.

This package contains multi-level backtesting infrastructure:
- Level 1: Price-only simulation (fast)
- Level 2: Surrogate model evaluation (medium)
- Level 3: Full council backtesting (slow)

FIX (MECE V2 #3): Corrected class name from CouncilSurrogate to DecisionSurrogate
FIX (V4): Phase-safe conditional imports — only imports submodules that exist.
           Empty __init__.py is safe to deploy in Phase 1 before Phase 3 files arrive.
"""

# Phase-safe imports: only import what actually exists.
# This allows deploy.sh to scaffold the directory in Phase 1
# without crashing on missing Phase 3 submodules.
_available = []

try:
    from .simple_backtest import SimpleBacktester, BacktestConfig, BacktestResult
    _available.extend(['SimpleBacktester', 'BacktestConfig', 'BacktestResult'])
except ImportError:
    pass  # Phase 3 not yet deployed

try:
    from .surrogate_models import DecisionSurrogate, SurrogateConfig
    _available.extend(['DecisionSurrogate', 'SurrogateConfig'])
except ImportError:
    pass  # Phase 3 not yet deployed

__all__ = _available
