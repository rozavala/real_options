#!/bin/bash
set -e
# =============================================================================
# Coffee Bot Real Options — Post-Deploy Verification Gate
#
# Runs lightweight checks that MUST pass before the orchestrator starts.
# If any check fails, returns non-zero → deploy.sh triggers rollback.
#
# Checks:
#   1. Required directories exist
#   2. Required files exist (new __init__.py, config files)
#   3. Critical Python imports succeed
#   4. Config files are valid JSON
#   5. Per-commodity data sanity (data dir writable, config loads, key paths resolve)
#   6. verify_system_readiness.py passes (quick mode, skip IBKR)
#   7. MasterOrchestrator mode (--multi) import check
# =============================================================================

# Determine Repo Root
if [ -f "pyproject.toml" ]; then
    REPO_ROOT=$(pwd)
elif [ -f "../pyproject.toml" ]; then
    REPO_ROOT=$(dirname $(pwd))
else
    # Fallback to hardcoded if not found (standard deploy path)
    REPO_ROOT=~/real_options
fi

cd "$REPO_ROOT"

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

ERRORS=0

echo "  [1/7] Checking required directories..."
REQUIRED_DIRS=(
    "config"
    "config/profiles"
    "trading_bot/prompts"
    "backtesting"
    "data"
    "data/KC"              # Primary commodity data directory
    "data/KC/surrogate_models"
)
# Auto-add directories for any commodity data dirs (e.g. data/CC, data/SB)
for _data_dir in data/*/; do
    [ -d "$_data_dir" ] || continue
    _tk=$(basename "$_data_dir")
    [[ "$_tk" =~ ^[A-Z]{2,4}$ ]] || continue
    [ "$_tk" = "KC" ] && continue  # Already in list
    REQUIRED_DIRS+=("data/$_tk")
done
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "    ❌ Missing directory: $dir"
        ERRORS=$((ERRORS + 1))
    fi
done

echo "  [2/7] Checking required files..."
REQUIRED_FILES=(
    # New packages (__init__.py)
    "config/__init__.py"
    "config/commodity_profiles.py"
    "trading_bot/prompts/__init__.py"
    "trading_bot/prompts/base_prompts.py"
    "backtesting/__init__.py"
    "trading_bot/enhanced_brier.py"
    "trading_bot/observability.py"
    "trading_bot/budget_guard.py"
    # Core files (existing — ensure not deleted)
    "trading_bot/__init__.py"
    "trading_bot/agents.py"
    "trading_bot/sentinels.py"
    "trading_bot/state_manager.py"
    "trading_bot/connection_pool.py"
    "trading_bot/tms.py"
    "trading_bot/brier_scoring.py"
    "orchestrator.py"
    "dashboard.py"
    "config.json"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "    ❌ Missing file: $file"
        ERRORS=$((ERRORS + 1))
    fi
done

echo "  [3/7] Checking Python imports..."
# Each import block tests one phase of the HRO guide.
# We test them individually so failures are pinpointed.

python -c "
import sys
errors = []

# Phase 1: Commodity Profiles
try:
    from config import get_commodity_profile, CommodityProfile, GrowingRegion
    profile = get_commodity_profile('KC')
    # Updated to match new profile name with parens
    assert profile.name == 'Coffee (Arabica)', f\"Expected 'Coffee (Arabica)', got '{profile.name}'\"
except Exception as e:
    errors.append(f'  config.commodity_profiles: {e}')

# Phase 1: Prompts
try:
    from trading_bot.prompts import get_agent_prompt, AgentPromptTemplate
except Exception as e:
    errors.append(f'  trading_bot.prompts: {e}')

# Phase 3: Backtesting (soft check)
try:
    from backtesting import SimpleBacktester, BacktestConfig
except ImportError:
    pass  # Acceptable if Phase 3 not yet implemented
except Exception as e:
    errors.append(f'  backtesting: {e}')

# Phase 5: Observability (soft check)
try:
    from trading_bot.observability import ObservabilityHub
except ImportError:
    pass  # Acceptable if Phase 5 not yet implemented
except Exception as e:
    errors.append(f'  trading_bot.observability: {e}')

# Core modules (MUST always pass)
try:
    from trading_bot.tms import TransactiveMemory
    # from trading_bot.agents import GroundedDataPacket # Not exported in __init__?
    # from trading_bot.heterogeneous_router import HeterogeneousRouter # Not in __init__?
    # Just check module import
    import trading_bot.agents
    import trading_bot.heterogeneous_router
    import trading_bot.weighted_voting
except Exception as e:
    errors.append(f'  core modules: {e}')

if errors:
    print('    ❌ Import failures:')
    for err in errors:
        print(f'      {err}')
    sys.exit(1)
else:
    print('    ✅ All imports OK')
"
if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

echo "  [4/7] Validating config files..."
python -c "
import json, sys
configs = ['config.json']

# Optional configs (check if exist, validate if present)
import os
if os.path.exists('config/api_costs.json'):
    configs.append('config/api_costs.json')

errors = []
for path in configs:
    try:
        with open(path) as f:
            json.load(f)
    except FileNotFoundError:
        errors.append(f'{path}: not found')
    except json.JSONDecodeError as e:
        errors.append(f'{path}: invalid JSON — {e}')

if errors:
    for err in errors:
        print(f'    ❌ {err}')
    sys.exit(1)
else:
    print(f'    ✅ {len(configs)} config files valid')
"
if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

echo "  [5/7] Per-commodity data sanity..."
# Auto-detect all commodity tickers from data/ directories.
# To add a new commodity: create data/{TICKER}/ and register a commodity profile.
ALL_TICKERS=()
for _data_dir in data/*/; do
    [ -d "$_data_dir" ] || continue
    _ticker=$(basename "$_data_dir")
    # Only uppercase directory names (KC, CC, SB) — skip surrogate_models etc.
    [[ "$_ticker" =~ ^[A-Z]{2,4}$ ]] || continue
    ALL_TICKERS+=("$_ticker")
done
# Fallback: at least KC
[ ${#ALL_TICKERS[@]} -eq 0 ] && ALL_TICKERS=("KC")

COMMODITY_WARNINGS=0
for _ticker in "${ALL_TICKERS[@]}"; do
    _data_dir="data/$_ticker"

    # 1. Data directory must exist (created by deploy.sh step 5)
    if [ ! -d "$_data_dir" ]; then
        echo "    ❌ [$_ticker] Missing data directory: $_data_dir"
        ERRORS=$((ERRORS + 1))
        continue  # Skip remaining checks for this ticker
    fi

    # 2. Data directory must be writable (orchestrator needs to create state files)
    if [ ! -w "$_data_dir" ]; then
        echo "    ❌ [$_ticker] Data directory not writable: $_data_dir"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    # 3. Config loads correctly with this commodity's ticker
    if ! COMMODITY_TICKER="$_ticker" python -c "
from config_loader import load_config
config = load_config()
import os, sys
expected_dir = os.path.join(os.path.dirname(os.path.abspath('config_loader.py')), 'data', '$_ticker')
actual = config.get('data_dir', '')
# Normalize for comparison
if os.path.abspath(actual) != os.path.abspath(expected_dir):
    print(f'    data_dir mismatch: expected {expected_dir}, got {actual}')
    sys.exit(1)
" 2>/dev/null; then
        echo "    ❌ [$_ticker] Config fails to load or data_dir mismatch"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    # 4. Commodity profile exists
    if ! python -c "
from config.commodity_profiles import get_commodity_profile
p = get_commodity_profile('$_ticker')
assert p is not None, 'No profile for $_ticker'
" 2>/dev/null; then
        echo "    ❌ [$_ticker] No commodity profile registered"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    # 5. Key runtime files (warnings only — may not exist for new commodities)
    for _file in "state.json" "council_history.csv" "enhanced_brier.json"; do
        if [ ! -f "$_data_dir/$_file" ]; then
            echo "    ⚠️  [$_ticker] Missing $_file (expected after first trading session)"
            COMMODITY_WARNINGS=$((COMMODITY_WARNINGS + 1))
        fi
    done

    echo "    ✅ [$_ticker] Data sanity OK"
done

if [ $COMMODITY_WARNINGS -gt 0 ]; then
    echo "    ($COMMODITY_WARNINGS warnings — normal for new commodities)"
fi

echo "  [6/7] Running system readiness check..."
if [ -f "verify_system_readiness.py" ]; then
    # Quick mode, skip IBKR (Gateway may not be ready during deploy)
    # CRITICAL: Use `if !` pattern so set -e doesn't abort on non-zero exit.
    # The readiness check is NON-BLOCKING because IBKR/LLM/YFinance
    # may not be reachable during deploy but will be at runtime.
    if ! python verify_system_readiness.py --quick --skip-ibkr --skip-llm 2>/dev/null; then
        echo "    ⚠️  System readiness check had failures (non-blocking)"
    else
        echo "    ✅ System readiness OK"
    fi
else
    echo "    ⏭️  verify_system_readiness.py not found, skipping"
fi

echo "  [7/7] Checking MasterOrchestrator mode..."
LEGACY_MODE="${LEGACY_MODE:-false}"
if [ "$LEGACY_MODE" = "true" ]; then
    echo "    ℹ️  LEGACY_MODE=true — running per-commodity services"
else
    # Verify MasterOrchestrator imports work
    if ! python -c "from trading_bot.master_orchestrator import main" 2>/dev/null; then
        echo "    ❌ MasterOrchestrator import failed"
        ERRORS=$((ERRORS + 1))
    else
        echo "    ✅ MasterOrchestrator mode ready (${#ALL_TICKERS[@]} commodities: ${ALL_TICKERS[*]})"
    fi
fi

# Final verdict
echo ""
if [ $ERRORS -gt 0 ]; then
    echo "  🚫 VERIFICATION FAILED ($ERRORS errors) — deploy will rollback"
    exit 1
else
    echo "  ✅ ALL CHECKS PASSED — safe to start"
    exit 0
fi
