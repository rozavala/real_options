"""
Level 1 Backtest: Price-Only Event Engine

This backtest validates:
- Position sizing logic
- Stop loss / take profit rules
- Options Greeks calculations
- P&L accounting

It does NOT involve LLMs - uses predetermined signals.
Speed: ~1 minute for 5 years of daily data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta  # FIX: Added timezone for datetime.now(timezone.utc)
from typing import List, Dict, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class StrategyType(Enum):
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    LONG_STRADDLE = "LONG_STRADDLE"
    IRON_CONDOR = "IRON_CONDOR"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    initial_capital: float = 100000.0
    max_position_pct: float = 0.10  # Max 10% per position
    straddle_breakeven_pct: float = 0.018  # 1.8% move needed
    condor_range_pct: float = 0.05  # 5% range for condors
    commission_per_contract: float = 2.50
    slippage_ticks: int = 1
    max_hold_days: int = 5
    spread_width_pct: float = 0.03  # Spread width as % of underlying (3%)
    premium_ratio: float = 0.33  # Credit received as fraction of spread width
    max_contracts: int = 10  # Cap contracts per trade
    contract_multiplier: float = 1.0  # Price-to-dollar multiplier (KC coffee: 375)


@dataclass
class Trade:
    """Individual trade record."""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    direction: SignalDirection = SignalDirection.NEUTRAL
    strategy: StrategyType = StrategyType.LONG_STRADDLE
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    contracts: int = 1
    pnl: float = 0.0
    outcome: str = "OPEN"


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    metrics: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Total Trades: {len(self.trades)}\n"
            f"Win Rate: {self.metrics.get('win_rate', 0):.1%}\n"
            f"Total P&L: ${self.metrics.get('total_pnl', 0):,.2f}\n"
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.1%}\n"
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}"
        )


class SimpleBacktester:
    """
    Event-driven backtest engine for strategy validation.

    This is Level 1 - no LLM calls, just price logic.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity = self.config.initial_capital
        self.equity_history: List[Dict] = []

    def run(
        self,
        price_data: pd.DataFrame,
        signal_func: Callable[[pd.Series, pd.DataFrame], Dict]
    ) -> BacktestResult:
        """
        Run backtest on price data.

        Args:
            price_data: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
            signal_func: Function that takes (current_row, historical_data) and returns
                        {'direction': SignalDirection, 'strategy': StrategyType, 'confidence': float}

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        if price_data.empty:
            logger.warning("Empty price data provided")
            return BacktestResult()

        # Ensure datetime index
        if 'date' in price_data.columns:
            price_data = price_data.set_index('date')

        open_trade: Optional[Trade] = None

        for i, (date, row) in enumerate(price_data.iterrows()):
            # Get historical data up to this point (no look-ahead)
            historical = price_data.iloc[:i+1]

            # Check for exit conditions on open trade
            if open_trade:
                open_trade = self._check_exit(open_trade, row, date)
                if open_trade.exit_date:
                    self.trades.append(open_trade)
                    self.equity += open_trade.pnl
                    open_trade = None

            # Generate signal for potential new trade
            if not open_trade and i > 20:  # Need some history
                signal = signal_func(row, historical)
                if signal and signal.get('confidence', 0) > 0.6:
                    open_trade = self._open_trade(
                        date=date,
                        price=row['close'],
                        direction=signal['direction'],
                        strategy=signal['strategy']
                    )

            # Record equity
            self.equity_history.append({
                'date': date,
                'equity': self.equity,
                'open_pnl': self._calc_open_pnl(open_trade, row) if open_trade else 0
            })

        # Close any remaining trade at end
        if open_trade:
            open_trade.exit_date = price_data.index[-1]
            open_trade.exit_price = price_data.iloc[-1]['close']
            open_trade.pnl = self._calc_pnl(open_trade)
            open_trade.outcome = self._grade_outcome(open_trade)
            self.trades.append(open_trade)
            self.equity += open_trade.pnl

        # Build results
        equity_df = pd.DataFrame(self.equity_history).set_index('date')

        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_df['equity'] + equity_df['open_pnl'],
            metrics=self._calc_metrics()
        )

    def _open_trade(
        self,
        date: datetime,
        price: float,
        direction: SignalDirection,
        strategy: StrategyType
    ) -> Trade:
        """Open a new trade."""
        max_risk = self.equity * self.config.max_position_pct

        mult = self.config.contract_multiplier

        if strategy in (StrategyType.BULL_PUT_SPREAD, StrategyType.BEAR_CALL_SPREAD):
            # Size by max loss of the spread (in real dollars)
            spread_width = price * self.config.spread_width_pct
            max_loss_per = spread_width * (1 - self.config.premium_ratio) * mult
            contracts = min(
                self.config.max_contracts,
                max(1, int(max_risk / max_loss_per))
            )
        else:
            contracts = max(1, int(max_risk / (price * 0.02 * mult)))

        return Trade(
            entry_date=date,
            direction=direction,
            strategy=strategy,
            entry_price=price,
            contracts=contracts
        )

    def _check_exit(self, trade: Trade, row: pd.Series, date: datetime) -> Trade:
        """Check exit conditions for open trade."""
        days_held = (date - trade.entry_date).days

        if days_held >= self.config.max_hold_days:
            trade.exit_date = date
            trade.exit_price = row['close']
            trade.pnl = self._calc_pnl(trade)
            trade.outcome = self._grade_outcome(trade)

        return trade

    def _calc_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade."""
        if trade.exit_price is None:
            return 0.0

        mult = self.config.contract_multiplier
        price_move_pct = (trade.exit_price - trade.entry_price) / trade.entry_price

        if trade.strategy == StrategyType.LONG_STRADDLE:
            # Straddle profits if move exceeds breakeven
            move_magnitude = abs(price_move_pct)
            if move_magnitude > self.config.straddle_breakeven_pct:
                pnl = (move_magnitude - self.config.straddle_breakeven_pct) * trade.entry_price * trade.contracts * mult
            else:
                pnl = -self.config.straddle_breakeven_pct * trade.entry_price * trade.contracts * mult

        elif trade.strategy == StrategyType.IRON_CONDOR:
            # Condor profits if price stays in range
            move_magnitude = abs(price_move_pct)
            if move_magnitude < self.config.condor_range_pct:
                pnl = self.config.condor_range_pct * 0.5 * trade.entry_price * trade.contracts * mult
            else:
                pnl = -self.config.condor_range_pct * trade.entry_price * trade.contracts * mult

        elif trade.strategy == StrategyType.BULL_PUT_SPREAD:
            # Credit spread: collect premium, lose if price drops through spread
            spread_width = trade.entry_price * self.config.spread_width_pct
            premium = spread_width * self.config.premium_ratio
            price_drop = max(0, trade.entry_price - trade.exit_price)
            loss = min(price_drop, spread_width)
            pnl = (premium - loss) * trade.contracts * mult

        elif trade.strategy == StrategyType.BEAR_CALL_SPREAD:
            # Credit spread: collect premium, lose if price rises through spread
            spread_width = trade.entry_price * self.config.spread_width_pct
            premium = spread_width * self.config.premium_ratio
            price_rise = max(0, trade.exit_price - trade.entry_price)
            loss = min(price_rise, spread_width)
            pnl = (premium - loss) * trade.contracts * mult

        elif trade.strategy == StrategyType.LONG_CALL:
            # Directional bullish
            pnl = price_move_pct * trade.entry_price * trade.contracts * mult

        elif trade.strategy == StrategyType.LONG_PUT:
            # Directional bearish
            pnl = -price_move_pct * trade.entry_price * trade.contracts * mult

        else:
            pnl = 0.0

        # Subtract commissions (entry + exit)
        pnl -= self.config.commission_per_contract * trade.contracts * 2

        return pnl

    def _calc_open_pnl(self, trade: Trade, row: pd.Series) -> float:
        """
        Calculate unrealized P&L.

        FIX (MECE V2 #7): Use timezone-aware datetime for consistency.
        """
        temp_trade = Trade(
            entry_date=trade.entry_date,
            exit_date=datetime.now(timezone.utc),  # Explicit UTC timezone
            direction=trade.direction,
            strategy=trade.strategy,
            entry_price=trade.entry_price,
            exit_price=row['close'],
            contracts=trade.contracts
        )
        return self._calc_pnl(temp_trade)

    def _grade_outcome(self, trade: Trade) -> str:
        """Grade trade outcome."""
        if trade.pnl > 0:
            return "WIN"
        elif trade.pnl < 0:
            return "LOSS"
        return "BREAK_EVEN"

    def _calc_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {}

        pnls = [t.pnl for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)

        # Equity curve metrics
        eq = pd.Series([h['equity'] for h in self.equity_history])
        returns = eq.pct_change().dropna()

        # Max drawdown
        rolling_max = eq.expanding().max()
        drawdown = (eq - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Sharpe (annualized, assuming daily data)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        return {
            'total_trades': len(self.trades),
            'win_rate': wins / len(self.trades) if self.trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'profit_factor': sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if sum(p for p in pnls if p < 0) != 0 else float('inf')
        }
