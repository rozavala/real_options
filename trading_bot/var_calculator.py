"""
Portfolio-Level VaR Calculator — Full Revaluation Historical Simulation.

Computes portfolio VaR at 95% and 99% confidence by repricing all option legs
via Black-Scholes under historical return scenarios. Captures gamma risk that
delta-normal methods miss.

Also hosts the AI Risk Agent (L1 Interpreter + L2 Scenario Architect) for
generating human-readable risk narratives and stress scenarios.

Architecture:
    ib.portfolio() → _snapshot_portfolio() → _fetch_aligned_returns()
    → _compute_scenarios() → VaRResult → data/var_state.json

State file lives at data/var_state.json (shared root, NOT per-commodity)
because VaR is portfolio-wide across all commodities.
"""

import asyncio
import fcntl
import json
import logging
import os
import time as _time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from trading_bot.utils import price_option_black_scholes, get_dollar_multiplier
from config.commodity_profiles import get_commodity_profile

logger = logging.getLogger(__name__)

# --- Module-level state ---
_var_data_dir: str = "data"
_calculator_instance: Optional["PortfolioVaRCalculator"] = None
_yf_cache: dict = {}  # {ticker: (df, fetch_epoch)}
_YF_CACHE_TTL = 4 * 3600  # 4 hours


def set_var_data_dir(data_dir: str):
    """Set the base data directory for VaR state file."""
    global _var_data_dir
    _var_data_dir = data_dir


def get_var_calculator(config: dict) -> "PortfolioVaRCalculator":
    """Singleton accessor. Config is NOT cached — passed to methods."""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = PortfolioVaRCalculator()
    return _calculator_instance


# --- Dataclasses ---

@dataclass
class PositionSnapshot:
    """Snapshot of a single position for VaR computation."""
    symbol: str               # Underlying ticker (e.g., "KC", "CC")
    sec_type: str             # "FOP", "FUT", "OPT"
    qty: float                # Signed quantity (+long, -short)
    strike: float             # 0 for futures
    right: str                # "C", "P", or "" for futures
    expiry_years: float       # Time to expiry in years
    iv: float                 # Implied volatility (annualized)
    underlying_price: float   # Current underlying price
    current_price: float      # Current option/future price
    dollar_multiplier: float  # P&L multiplier per 1.0 price unit move


@dataclass
class VaRResult:
    """Result of a VaR computation."""
    var_95: float = 0.0           # Dollar VaR at 95%
    var_99: float = 0.0           # Dollar VaR at 99%
    var_95_pct: float = 0.0       # VaR as % of equity
    var_99_pct: float = 0.0       # VaR as % of equity
    equity: float = 0.0           # Account equity used
    position_count: int = 0
    commodities: list = field(default_factory=list)  # Unique commodity tickers
    computed_epoch: float = 0.0   # time.time() for staleness math
    timestamp: str = ""           # ISO string for display
    is_stale: bool = False
    positions_detail: list = field(default_factory=list)  # List of position summaries
    narrative: dict = field(default_factory=dict)          # L1 Interpreter output
    scenarios: list = field(default_factory=list)          # L2 Scenario Architect output
    last_attempt_status: str = "OK"   # "OK" or "FAILED"
    last_attempt_error: str = ""      # Error message on failure

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VaRResult":
        return cls(
            var_95=d.get("var_95", 0.0),
            var_99=d.get("var_99", 0.0),
            var_95_pct=d.get("var_95_pct", 0.0),
            var_99_pct=d.get("var_99_pct", 0.0),
            equity=d.get("equity", 0.0),
            position_count=d.get("position_count", 0),
            commodities=d.get("commodities", []),
            computed_epoch=d.get("computed_epoch", 0.0),
            timestamp=d.get("timestamp", ""),
            is_stale=d.get("is_stale", False),
            positions_detail=d.get("positions_detail", []),
            narrative=d.get("narrative", {}),
            scenarios=d.get("scenarios", []),
            last_attempt_status=d.get("last_attempt_status", "OK"),
            last_attempt_error=d.get("last_attempt_error", ""),
        )


# --- Main Calculator ---

class PortfolioVaRCalculator:
    """
    Full Revaluation Historical Simulation VaR.

    Reprices each option leg under historical return scenarios using B-S,
    then computes portfolio P&L distribution to extract percentiles.
    """

    def __init__(self):
        self._cached_result: Optional[VaRResult] = None

    # --- Public API ---

    async def compute_portfolio_var(self, ib, config: dict) -> VaRResult:
        """Full VaR computation pipeline. Saves result to state file."""
        now = _time.time()
        try:
            # 1. Snapshot current positions
            positions = await self._snapshot_portfolio(ib, config)
            if not positions:
                result = VaRResult(
                    computed_epoch=now,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    last_attempt_status="OK",
                )
                self._cached_result = result
                self._save_state(result)
                return result

            # 2. Fetch aligned historical returns
            symbols = list({p.symbol for p in positions})
            lookback = config.get("compliance", {}).get("var_lookback_days", 252)
            returns_df = self._fetch_aligned_returns(symbols, lookback)
            if returns_df is None:
                raise ValueError("Failed to fetch aligned returns — data quality guard triggered")

            # 3. Compute account equity
            equity = self._get_equity(ib)

            # 4. Full revaluation
            r = config.get("compliance", {}).get("var_risk_free_rate", 0.04)
            scenarios = self._compute_scenarios(positions, returns_df, r)

            # 5. Extract percentiles
            var_95 = float(-np.percentile(scenarios, 5))
            var_99 = float(-np.percentile(scenarios, 1))
            var_95 = max(var_95, 0.0)
            var_99 = max(var_99, 0.0)

            result = VaRResult(
                var_95=round(var_95, 2),
                var_99=round(var_99, 2),
                var_95_pct=round(var_95 / equity, 6) if equity > 0 else 0.0,
                var_99_pct=round(var_99 / equity, 6) if equity > 0 else 0.0,
                equity=round(equity, 2),
                position_count=len(positions),
                commodities=sorted(set(p.symbol for p in positions)),
                computed_epoch=now,
                timestamp=datetime.now(timezone.utc).isoformat(),
                positions_detail=[
                    {
                        "symbol": p.symbol,
                        "sec_type": p.sec_type,
                        "qty": p.qty,
                        "strike": p.strike,
                        "right": p.right,
                        "iv": round(p.iv, 4),
                        "underlying_price": p.underlying_price,
                    }
                    for p in positions
                ],
                last_attempt_status="OK",
            )
            self._cached_result = result
            self._save_state(result)
            return result

        except Exception as e:
            logger.error(f"VaR computation failed: {e}")
            # Save failure state
            fail_result = VaRResult(
                computed_epoch=now,
                timestamp=datetime.now(timezone.utc).isoformat(),
                last_attempt_status="FAILED",
                last_attempt_error=str(e),
            )
            # Preserve previous good result in cache if available
            if self._cached_result and self._cached_result.last_attempt_status == "OK":
                self._cached_result.last_attempt_status = "OK"  # Keep good cache
            self._save_state(fail_result)
            raise

    async def compute_var_with_proposed_trade(
        self, ib, config: dict, proposed_snapshots: list[PositionSnapshot]
    ) -> VaRResult:
        """Pre-trade VaR: appends proposed positions and re-runs revaluation."""
        positions = await self._snapshot_portfolio(ib, config)
        positions.extend(proposed_snapshots)

        if not positions:
            return VaRResult(
                computed_epoch=_time.time(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        symbols = list({p.symbol for p in positions})
        lookback = config.get("compliance", {}).get("var_lookback_days", 252)
        returns_df = self._fetch_aligned_returns(symbols, lookback)
        if returns_df is None:
            raise ValueError("Failed to fetch aligned returns for pre-trade VaR")

        equity = self._get_equity(ib)
        r = config.get("compliance", {}).get("var_risk_free_rate", 0.04)
        scenarios = self._compute_scenarios(positions, returns_df, r)

        var_95 = float(max(-np.percentile(scenarios, 5), 0.0))
        var_99 = float(max(-np.percentile(scenarios, 1), 0.0))

        return VaRResult(
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            var_95_pct=round(var_95 / equity, 6) if equity > 0 else 0.0,
            var_99_pct=round(var_99 / equity, 6) if equity > 0 else 0.0,
            equity=round(equity, 2),
            position_count=len(positions),
            commodities=sorted(set(p.symbol for p in positions)),
            computed_epoch=_time.time(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_cached_var(self) -> Optional[VaRResult]:
        """Return cached result if fresh, else try loading from state file."""
        if self._cached_result and self._cached_result.last_attempt_status == "OK":
            return self._cached_result
        loaded = self._load_state()
        if loaded and loaded.last_attempt_status == "OK":
            self._cached_result = loaded
            return loaded
        return None

    async def compute_stress_scenario(
        self, ib, config: dict, scenario: dict
    ) -> dict:
        """
        Run a stress scenario with price and/or IV shocks.

        scenario = {
            "name": "Commodity crash",
            "price_shock_pct": -0.15,   # -15% underlying move
            "iv_shock_pct": 0.30,       # +30% IV increase
            "time_horizon_weeks": 2,    # Optional: theta decay
        }
        """
        positions = await self._snapshot_portfolio(ib, config)
        if not positions:
            return {"name": scenario.get("name", ""), "pnl": 0.0, "positions": 0}

        price_shock = scenario.get("price_shock_pct", 0.0)
        iv_shock = scenario.get("iv_shock_pct", 0.0)
        time_horizon_weeks = scenario.get("time_horizon_weeks", 0)
        r = config.get("compliance", {}).get("var_risk_free_rate", 0.04)

        total_pnl = 0.0
        for pos in positions:
            shocked_price = pos.underlying_price * (1.0 + price_shock)
            shocked_iv = pos.iv * (1.0 + iv_shock)
            shocked_iv = max(shocked_iv, 0.01)

            t_shocked = pos.expiry_years
            if time_horizon_weeks > 0:
                t_shocked = max(pos.expiry_years - time_horizon_weeks / 52, 1e-6)

            if pos.sec_type in ("FOP", "OPT"):
                new_price = self._bs_price(
                    shocked_price, pos.strike, t_shocked, r, shocked_iv, pos.right
                )
                pnl = (new_price - pos.current_price) * pos.qty * pos.dollar_multiplier
            else:
                # Futures: linear P&L
                pnl = (shocked_price - pos.underlying_price) * pos.qty * pos.dollar_multiplier
            total_pnl += pnl

        return {
            "name": scenario.get("name", ""),
            "pnl": round(total_pnl, 2),
            "positions": len(positions),
            "price_shock_pct": price_shock,
            "iv_shock_pct": iv_shock,
        }

    # --- Internal helpers ---

    async def _snapshot_portfolio(
        self, ib, config: dict
    ) -> list[PositionSnapshot]:
        """Read ib.portfolio(), batch-fetch IV for options, build snapshots."""
        try:
            portfolio_items = ib.portfolio()
        except Exception as e:
            logger.error(f"Failed to read ib.portfolio(): {e}")
            return []

        if not portfolio_items:
            return []

        snapshots = []
        # Collect option positions that need IV
        option_items = []
        # Build underlying price map from FUT positions (symbol → marketPrice)
        # so options can look up the underlying future price for B-S repricing
        underlying_price_map: dict[str, float] = {}
        for item in portfolio_items:
            if item.position == 0:
                continue
            c = item.contract
            if c.secType in ("FOP", "OPT"):
                option_items.append(item)
            elif c.secType == "FUT":
                ticker_sym = c.symbol
                try:
                    multiplier = float(c.multiplier) if c.multiplier else get_dollar_multiplier(config)
                except (ValueError, TypeError):
                    multiplier = get_dollar_multiplier(config)

                underlying_price_map[ticker_sym] = item.marketPrice

                snapshots.append(PositionSnapshot(
                    symbol=ticker_sym,
                    sec_type="FUT",
                    qty=float(item.position),
                    strike=0.0,
                    right="",
                    expiry_years=0.0,
                    iv=0.0,
                    underlying_price=item.marketPrice,
                    current_price=item.marketPrice,
                    dollar_multiplier=multiplier,
                ))

        # Batched IV fetch for all options simultaneously
        if option_items:
            iv_map = await self._batch_fetch_iv(ib, option_items, config)

            for item in option_items:
                c = item.contract
                con_id = c.conId
                iv_val = iv_map.get(con_id)

                # Use fallback IV from commodity profile if market IV unavailable
                if iv_val is None or iv_val <= 0:
                    try:
                        profile = get_commodity_profile(c.symbol)
                        iv_val = profile.fallback_iv
                    except (ValueError, KeyError):
                        iv_val = 0.35
                    logger.warning(
                        f"IV unavailable for {c.localSymbol} (conId={con_id}) "
                        f"— using fallback IV={iv_val:.2f}"
                    )

                ticker_sym = c.symbol
                try:
                    multiplier = float(c.multiplier) if c.multiplier else get_dollar_multiplier(config)
                except (ValueError, TypeError):
                    multiplier = get_dollar_multiplier(config)

                # Calculate time to expiry
                expiry_str = c.lastTradeDateOrContractMonth
                expiry_years = self._calc_expiry_years(expiry_str)
                if expiry_years <= 0:
                    logger.warning(f"Expired or invalid expiry for {c.localSymbol} — excluded from VaR")
                    continue

                # Resolve underlying future price from FUT positions in portfolio
                und_price = underlying_price_map.get(ticker_sym)
                if und_price is None or und_price <= 0:
                    logger.warning(
                        f"No underlying future price for {c.localSymbol} "
                        f"(symbol={ticker_sym}) — excluded from VaR"
                    )
                    continue

                snapshots.append(PositionSnapshot(
                    symbol=ticker_sym,
                    sec_type=c.secType,
                    qty=float(item.position),
                    strike=float(c.strike),
                    right=c.right,
                    expiry_years=expiry_years,
                    iv=iv_val,
                    underlying_price=und_price,
                    current_price=item.marketPrice,  # option's current market price
                    dollar_multiplier=multiplier,
                ))

        return snapshots

    async def _batch_fetch_iv(
        self, ib, option_items: list, config: dict
    ) -> dict[int, Optional[float]]:
        """
        Batch-request market data for all options simultaneously.
        Returns {conId: iv_float_or_None}.
        """
        iv_map: dict[int, Optional[float]] = {}
        tickers = []

        for item in option_items:
            c = item.contract
            try:
                t = ib.reqMktData(c, "106", False, False)  # 106 = implied vol
                tickers.append((c.conId, t))
            except Exception as e:
                logger.warning(f"reqMktData failed for {c.localSymbol}: {e}")
                iv_map[c.conId] = None

        # Single wait for all tickers
        if tickers:
            await asyncio.sleep(3)

        for con_id, t in tickers:
            try:
                # modelGreeks contains the IV from IB
                if hasattr(t, 'modelGreeks') and t.modelGreeks:
                    iv_val = t.modelGreeks.impliedVol
                    if iv_val and not _is_nan(iv_val) and iv_val > 0:
                        iv_map[con_id] = float(iv_val)
                    else:
                        iv_map[con_id] = None
                else:
                    iv_map[con_id] = None
            except Exception:
                iv_map[con_id] = None
            finally:
                try:
                    ib.cancelMktData(t.contract)
                except Exception:
                    pass

        return iv_map

    def _fetch_aligned_returns(
        self, symbols: list[str], lookback_days: int = 252
    ) -> Optional["pd.DataFrame"]:
        """
        Fetch aligned daily returns from yfinance with quality guards.
        Returns DataFrame with columns per symbol, or None on failure.
        """
        import pandas as pd

        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — cannot compute VaR")
            return None

        now = _time.time()
        all_returns = {}

        for sym in symbols:
            yf_ticker = self._get_yf_ticker(sym)

            # Check cache
            if yf_ticker in _yf_cache:
                cached_df, cached_epoch = _yf_cache[yf_ticker]
                if now - cached_epoch < _YF_CACHE_TTL:
                    all_returns[sym] = cached_df
                    continue

            try:
                data = yf.download(
                    yf_ticker,
                    period=f"{lookback_days + 30}d",  # Extra buffer for alignment
                    progress=False,
                    auto_adjust=True,
                )
                if data is None or data.empty or len(data) < 20:
                    logger.warning(f"Insufficient yfinance data for {yf_ticker}: {len(data) if data is not None else 0} rows")
                    return None

                close = data["Close"].squeeze()
                returns = close.pct_change().dropna().tail(lookback_days)
                _yf_cache[yf_ticker] = (returns, now)
                all_returns[sym] = returns
            except Exception as e:
                logger.error(f"yfinance fetch failed for {yf_ticker}: {e}")
                return None

        if not all_returns:
            return None

        # Align on common dates
        df = pd.DataFrame(all_returns)
        df = df.fillna(0.0)

        # Quality guard: >20% zeros → warn
        for col in df.columns:
            zero_pct = (df[col] == 0.0).sum() / len(df)
            if zero_pct > 0.2:
                logger.warning(f"VaR data quality: {col} has {zero_pct:.0%} zero returns")

            # 5+ consecutive zeros → error
            max_consec_zeros = _max_consecutive_zeros(df[col].values)
            if max_consec_zeros >= 5:
                logger.error(f"VaR data quality: {col} has {max_consec_zeros} consecutive zero returns — unreliable")
                return None

            # Near-zero variance guard
            if df[col].var() < 1e-12:
                logger.error(f"VaR data quality: {col} has near-zero variance — cannot compute VaR")
                return None

        return df

    def _compute_scenarios(
        self,
        positions: list[PositionSnapshot],
        returns_df: "pd.DataFrame",
        risk_free_rate: float,
    ) -> np.ndarray:
        """
        Full B-S revaluation per historical scenario.
        Returns array of portfolio P&L per scenario.
        """
        n_scenarios = len(returns_df)
        pnl_array = np.zeros(n_scenarios)

        for pos in positions:
            sym = pos.symbol
            if sym not in returns_df.columns:
                # No return data for this symbol — skip
                logger.warning(f"No return data for {sym} — excluded from scenarios")
                continue

            sym_returns = returns_df[sym].values

            if pos.sec_type in ("FOP", "OPT"):
                # Full B-S revaluation for options
                for i, ret in enumerate(sym_returns):
                    shocked_S = pos.underlying_price * (1.0 + ret)
                    new_price = self._bs_price(
                        shocked_S, pos.strike, pos.expiry_years,
                        risk_free_rate, pos.iv, pos.right
                    )
                    pnl = (new_price - pos.current_price) * pos.qty * pos.dollar_multiplier
                    pnl_array[i] += pnl
            else:
                # Futures: linear P&L
                for i, ret in enumerate(sym_returns):
                    pnl = ret * pos.underlying_price * pos.qty * pos.dollar_multiplier
                    pnl_array[i] += pnl

        return pnl_array

    def _bs_price(
        self, S: float, K: float, T: float, r: float, sigma: float, right: str
    ) -> float:
        """
        Wrapper around price_option_black_scholes() from utils.py.
        Returns option price. On None (T<=0, sigma<=0): returns intrinsic value.
        """
        result = price_option_black_scholes(S, K, T, r, sigma, right)
        if result is None:
            # Fallback to intrinsic value
            if right.upper() == "C":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)
        return result["price"]

    def _get_yf_ticker(self, symbol: str) -> str:
        """Resolve yfinance ticker from CommodityProfile or fallback."""
        try:
            profile = get_commodity_profile(symbol)
            if hasattr(profile, 'yfinance_ticker') and profile.yfinance_ticker:
                return profile.yfinance_ticker
        except (ValueError, KeyError):
            pass
        return f"{symbol}=F"

    def _get_equity(self, ib) -> float:
        """Get account equity from IB."""
        try:
            account_values = ib.accountSummary()
            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    return float(av.value)
        except Exception as e:
            logger.warning(f"Failed to get equity from IB: {e}")

        # Fallback: sum portfolio market values
        try:
            total = sum(
                item.marketValue for item in ib.portfolio()
                if item.position != 0
            )
            return max(total, 1.0)
        except Exception:
            return 50000.0  # Last resort default

    def _calc_expiry_years(self, expiry_str: str) -> float:
        """Convert YYYYMMDD expiry string to years until expiry."""
        try:
            expiry_date = datetime.strptime(expiry_str[:8], "%Y%m%d")
            now = datetime.now()
            days = (expiry_date - now).days
            return max(days / 365.0, 0.0)
        except (ValueError, TypeError):
            return 0.0

    async def _build_proposed_snapshots(
        self, order_context: dict, config: dict, ib
    ) -> list[PositionSnapshot]:
        """
        Decompose order_context into PositionSnapshot objects for pre-trade VaR.
        Must be async for BAG orders (needs reqContractDetailsAsync).
        """
        snapshots = []
        contract = order_context.get("contract")
        order_obj = order_context.get("order_object")
        if not contract or not order_obj:
            return snapshots

        qty = float(order_obj.totalQuantity)
        if order_obj.action == "SELL":
            qty = -qty

        try:
            multiplier = float(contract.multiplier) if contract.multiplier else get_dollar_multiplier(config)
        except (ValueError, TypeError):
            multiplier = get_dollar_multiplier(config)

        if hasattr(contract, "comboLegs") and contract.comboLegs:
            # BAG order — resolve each combo leg
            for leg in contract.comboLegs:
                try:
                    from ib_insync import Contract as IBContract
                    leg_contract = IBContract(conId=leg.conId)
                    details_list = await asyncio.wait_for(
                        ib.reqContractDetailsAsync(leg_contract), timeout=10
                    )
                    if details_list:
                        det = details_list[0].contract
                        leg_qty = qty * leg.ratio
                        if leg.action == "SELL":
                            leg_qty = -leg_qty

                        expiry_years = self._calc_expiry_years(
                            det.lastTradeDateOrContractMonth
                        )
                        # Use fallback IV from profile
                        try:
                            profile = get_commodity_profile(det.symbol)
                            fallback_iv = profile.fallback_iv
                        except (ValueError, KeyError):
                            fallback_iv = 0.35

                        snapshots.append(PositionSnapshot(
                            symbol=det.symbol,
                            sec_type=det.secType,
                            qty=leg_qty,
                            strike=float(det.strike) if det.strike else 0.0,
                            right=det.right or "",
                            expiry_years=expiry_years,
                            iv=fallback_iv,  # Approximate — real IV from market data
                            underlying_price=order_context.get("price", 0.0),
                            current_price=order_context.get("price", 0.0),
                            dollar_multiplier=multiplier,
                        ))
                except Exception as e:
                    logger.warning(f"Failed to resolve combo leg conId={leg.conId}: {e}")

        elif contract.secType in ("FOP", "OPT"):
            # Single option
            expiry_years = self._calc_expiry_years(
                contract.lastTradeDateOrContractMonth
            )
            try:
                profile = get_commodity_profile(contract.symbol)
                fallback_iv = profile.fallback_iv
            except (ValueError, KeyError):
                fallback_iv = 0.35

            snapshots.append(PositionSnapshot(
                symbol=contract.symbol,
                sec_type=contract.secType,
                qty=qty,
                strike=float(contract.strike) if contract.strike else 0.0,
                right=contract.right or "",
                expiry_years=expiry_years,
                iv=fallback_iv,
                underlying_price=order_context.get("price", 0.0),
                current_price=order_context.get("price", 0.0),
                dollar_multiplier=multiplier,
            ))

        return snapshots

    # --- State persistence ---

    def _save_state(self, result: VaRResult):
        """Atomic save with file locking."""
        state_path = os.path.join(_var_data_dir, "var_state.json")
        tmp_path = state_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            data = result.to_dict()
            with open(tmp_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f, indent=2, default=str)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            os.replace(tmp_path, state_path)
        except Exception as e:
            logger.error(f"Failed to save VaR state: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_state(self) -> Optional[VaRResult]:
        """Load VaR state from JSON file."""
        state_path = os.path.join(_var_data_dir, "var_state.json")
        try:
            with open(state_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return VaRResult.from_dict(data)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to load VaR state: {e}")
            return None


# --- AI Risk Agent ---

async def run_risk_agent(
    var_result: VaRResult, config: dict, ib=None, prev_var: Optional[VaRResult] = None
) -> dict:
    """
    AI Risk Agent: L1 Interpreter + L2 Scenario Architect.
    Returns {interpretation: dict, scenarios: list}.
    All failures non-fatal — VaR result is saved regardless.
    """
    output = {"interpretation": {}, "scenarios": []}

    var_warning = config.get("compliance", {}).get("var_warning_pct", 0.02)
    var_limit = config.get("compliance", {}).get("var_limit_pct", 0.03)

    # L1 Interpreter: triggers when VaR > warning threshold
    if var_result.var_95_pct > var_warning:
        try:
            interpretation = await _run_l1_interpreter(var_result, config, prev_var)
            output["interpretation"] = interpretation
        except Exception as e:
            logger.warning(f"L1 Risk Interpreter failed (non-fatal): {e}")

    # L2 Scenario Architect: triggers when L1 urgency >= MEDIUM or VaR > 80% of limit
    l1_urgency = output["interpretation"].get("urgency", "LOW")
    var_util = var_result.var_95_pct / var_limit if var_limit > 0 else 0

    if l1_urgency in ("MEDIUM", "HIGH", "CRITICAL") or var_util > 0.8:
        try:
            scenarios = await _run_l2_scenario_architect(var_result, config, ib)
            output["scenarios"] = scenarios
        except Exception as e:
            logger.warning(f"L2 Scenario Architect failed (non-fatal): {e}")

    return output


async def _run_l1_interpreter(
    var_result: VaRResult, config: dict, prev_var: Optional[VaRResult] = None
) -> dict:
    """L1: Interpret VaR result into structured narrative."""
    try:
        from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole
        from trading_bot.state_manager import StateManager
    except ImportError:
        return {}

    router = HeterogeneousRouter(config)

    # Build context for the LLM
    trend = "STABLE"
    if prev_var and prev_var.var_95_pct > 0:
        delta = var_result.var_95_pct - prev_var.var_95_pct
        if delta > 0.005:
            trend = "RISING"
        elif delta < -0.005:
            trend = "FALLING"

    sentinel_state = {}
    try:
        sentinel_state = StateManager.load_state("sensors", max_age=7200)
    except Exception:
        pass

    prompt = f"""You are a Risk Analyst interpreting portfolio VaR for a commodity options trading system.

PORTFOLIO VaR SNAPSHOT:
- VaR(95%): {var_result.var_95_pct:.2%} of equity (${var_result.var_95:,.0f})
- VaR(99%): {var_result.var_99_pct:.2%} of equity (${var_result.var_99:,.0f})
- Equity: ${var_result.equity:,.0f}
- Positions: {var_result.position_count} across {', '.join(var_result.commodities)}
- VaR Trend: {trend}

POSITION DETAIL:
{json.dumps(var_result.positions_detail[:10], indent=2)}

SENTINEL STATE:
{json.dumps(sentinel_state, indent=2, default=str)[:2000]}

OUTPUT JSON:
{{
  "dominant_risk": "<brief description of the primary risk driver>",
  "correlation_warning": "<any cross-commodity correlation concern or 'None detected'>",
  "trend": "{trend}",
  "urgency": "LOW|MEDIUM|HIGH|CRITICAL",
  "recommendation": "<1-2 sentence actionable recommendation>"
}}"""

    try:
        response = await router.route(AgentRole.COMPLIANCE_OFFICER, prompt, response_json=True)
        result = json.loads(response) if isinstance(response, str) else response
        # Validate expected keys
        for key in ("dominant_risk", "correlation_warning", "trend", "urgency"):
            if key not in result:
                result[key] = "N/A"
        return result
    except Exception as e:
        logger.warning(f"L1 Interpreter LLM call failed: {e}")
        return {"dominant_risk": "Unknown", "correlation_warning": "N/A", "trend": trend, "urgency": "MEDIUM"}


async def _run_l2_scenario_architect(
    var_result: VaRResult, config: dict, ib=None
) -> list[dict]:
    """L2: Generate and run stress scenarios from sentinel intelligence."""
    try:
        from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole
        from trading_bot.state_manager import StateManager
    except ImportError:
        return []

    router = HeterogeneousRouter(config)

    sentinel_state = {}
    try:
        sentinel_state = StateManager.load_state("sensors", max_age=7200)
    except Exception:
        pass

    prompt = f"""You are a Risk Scenario Architect for a commodity options portfolio.

CURRENT PORTFOLIO:
- VaR(95%): {var_result.var_95_pct:.2%} (${var_result.var_95:,.0f})
- Commodities: {', '.join(var_result.commodities)}
- Position count: {var_result.position_count}

POSITIONS:
{json.dumps(var_result.positions_detail[:10], indent=2)}

SENTINEL INTELLIGENCE:
{json.dumps(sentinel_state, indent=2, default=str)[:2000]}

Generate 2-3 plausible stress scenarios based on current market intelligence.
Each scenario should have a name, probability estimate, and shock parameters.

OUTPUT JSON (array):
[
  {{
    "name": "<scenario name>",
    "probability": "<LOW|MEDIUM|HIGH>",
    "description": "<1 sentence>",
    "price_shock_pct": <float, e.g., -0.10 for -10%>,
    "iv_shock_pct": <float, e.g., 0.20 for +20% IV increase>,
    "time_horizon_weeks": <int, 1-4>
  }}
]"""

    try:
        response = await router.route(AgentRole.COMPLIANCE_OFFICER, prompt, response_json=True)
        scenarios_raw = json.loads(response) if isinstance(response, str) else response
        if not isinstance(scenarios_raw, list):
            scenarios_raw = [scenarios_raw]

        # Run each scenario through the stress engine
        calculator = get_var_calculator(config)
        results = []
        for s in scenarios_raw[:3]:  # Max 3 scenarios
            if ib:
                stress_result = await calculator.compute_stress_scenario(ib, config, s)
                s["pnl"] = stress_result.get("pnl", 0.0)
            results.append(s)
        return results
    except Exception as e:
        logger.warning(f"L2 Scenario Architect LLM call failed: {e}")
        return []


# --- Utilities ---

def _is_nan(val) -> bool:
    """Check if a value is NaN."""
    try:
        return val != val  # NaN != NaN
    except Exception:
        return False


def _max_consecutive_zeros(arr) -> int:
    """Count max consecutive zeros in a numpy array."""
    max_count = 0
    current = 0
    for val in arr:
        if val == 0.0:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count
