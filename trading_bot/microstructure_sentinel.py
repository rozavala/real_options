"""Market Microstructure Sentinel.

Monitors:
- Flash crash risk (bid-ask spread > 3 std devs)
- Volume spikes (> 500% of moving average)
- Liquidity depletion events
"""

import logging
import asyncio
from datetime import datetime, timezone, time
import pytz
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SentinelTrigger:
    """Trigger from microstructure sentinel."""
    source: str = "MicrostructureSentinel"
    reason: str = ""
    severity: float = 0.0
    payload: dict = field(default_factory=dict)


class MicrostructureSentinel:
    """Monitors Level 2 market data for anomalies."""

    def __init__(self, config: dict, ib):
        self.config = config
        self.ib = ib

        sentinel_config = config.get('sentinels', {}).get('microstructure', {})
        self.spread_std_threshold = sentinel_config.get('spread_std_threshold', 3.0)
        self.volume_spike_pct = sentinel_config.get('volume_spike_pct', 5.0)
        self.depth_drop_pct = sentinel_config.get('depth_drop_pct', 0.5)

        self.spread_history: deque = deque(maxlen=1440)
        self.volume_history: deque = deque(maxlen=60)
        self.depth_history: deque = deque(maxlen=30)

        self.last_trigger_time: Optional[datetime] = None
        self.cooldown_seconds = sentinel_config.get('cooldown_seconds', 300)
        self.tickers = {}

        logger.info(f"MicrostructureSentinel initialized")

    async def subscribe_contract(self, contract):
        """Subscribe to market data for contract with conflict handling."""
        if contract.conId in self.tickers:
            return

        try:
            ticker = self.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(2)

            # Check if we got data (live data should have bid/ask/last)
            # Note: For microstructure we mostly need bid/ask.
            if ticker.last is None and ticker.bid is None and ticker.ask is None:
                raise Exception("No data received - possible competing session")

            self.tickers[contract.conId] = {'contract': contract, 'ticker': ticker}
            logger.info(f"Subscribed to microstructure for {contract.localSymbol}")

        except Exception as e:
            if "10197" in str(e) or "competing" in str(e).lower() or "No data received" in str(e):
                logger.warning(f"Competing session detected for {contract.localSymbol}. Switching to delayed data.")

                # Switch to delayed and retry
                self.ib.reqMarketDataType(3)
                await asyncio.sleep(0.5)

                ticker = self.ib.reqMktData(contract, '', False, False)
                self.tickers[contract.conId] = {'contract': contract, 'ticker': ticker}
                logger.info(f"Subscribed to microstructure (Delayed) for {contract.localSymbol}")
            else:
                logger.error(f"Failed to subscribe to {contract.localSymbol}: {e}")

    def _calculate_spread_stats(self) -> tuple[float, float]:
        """Calculate mean and std of historical spreads."""
        if len(self.spread_history) < 30:
            return 0.0, float('inf')
        spreads = list(self.spread_history)
        mean = sum(spreads) / len(spreads)
        variance = sum((s - mean) ** 2 for s in spreads) / len(spreads)
        return mean, variance ** 0.5

    def _calculate_volume_baseline(self) -> float:
        """Calculate moving average volume."""
        if len(self.volume_history) < 5:
            return 0.0
        return sum(self.volume_history) / len(self.volume_history)

    def _is_in_cooldown(self) -> bool:
        """Check cooldown period."""
        if self.last_trigger_time is None:
            return False
        elapsed = (datetime.now() - self.last_trigger_time).total_seconds()
        return elapsed < self.cooldown_seconds

    def is_core_market_hours(self) -> bool:
        """Check if current time is within core US trading hours (09:00 - 13:30 ET)."""
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)

        # Check for weekends
        if now.weekday() >= 5:  # 5=Sat, 6=Sun
            return False

        # Core hours: 09:00 - 13:30
        market_open = time(9, 0)
        market_close = time(13, 30)

        return market_open <= now.time() <= market_close

    async def check(self) -> Optional[SentinelTrigger]:
        """Check for microstructure anomalies."""
        if not self.is_core_market_hours():
            logger.info("Market Closed")
            return None

        if not self.ib.isConnected():
            # Clear stale data on disconnect
            self.spread_history.clear()
            self.volume_history.clear()
            return None

        if self._is_in_cooldown() or not self.tickers:
            return None

        # Add staleness check - copy items to avoid mutation issues if dict changes
        for con_id, data in list(self.tickers.items()):
            ticker = data['ticker']
            contract = data['contract']

            # Check if ticker data is stale (no updates in 5 minutes)
            if hasattr(ticker, 'time') and ticker.time:
                last_update = ticker.time
                # Ensure we use timezone-aware datetime if ticker.time is aware
                now = datetime.now(timezone.utc) if last_update.tzinfo else datetime.now()
                if (now - last_update).seconds > 300:
                    logger.warning(f"Stale ticker data for {con_id}, skipping")
                    continue

            if not ticker.bid or not ticker.ask or ticker.bid <= 0 or ticker.ask <= 0:
                continue

            current_spread = ticker.ask - ticker.bid
            mid_price = (ticker.ask + ticker.bid) / 2
            spread_pct = current_spread / mid_price if mid_price > 0 else 0

            self.spread_history.append(spread_pct)

            if hasattr(ticker, 'volume') and ticker.volume:
                self.volume_history.append(ticker.volume)

            # Check spread anomaly
            mean_spread, std_spread = self._calculate_spread_stats()
            if std_spread > 0 and std_spread != float('inf'):
                z_score = (spread_pct - mean_spread) / std_spread
                if z_score > self.spread_std_threshold:
                    self.last_trigger_time = datetime.now()
                    return SentinelTrigger(
                        reason=f"Flash Crash Risk: Spread {z_score:.1f} std devs above mean",
                        severity=min(10, z_score * 2),
                        payload={
                            'type': 'SPREAD_ANOMALY',
                            'contract': contract.localSymbol,
                            'z_score': z_score,
                            'bid': ticker.bid,
                            'ask': ticker.ask
                        }
                    )

            # Check volume spike
            baseline_volume = self._calculate_volume_baseline()
            if baseline_volume > 0 and hasattr(ticker, 'volume') and ticker.volume:
                volume_ratio = ticker.volume / baseline_volume
                if volume_ratio > self.volume_spike_pct:
                    self.last_trigger_time = datetime.now()
                    return SentinelTrigger(
                        reason=f"Volume Spike: {volume_ratio*100:.0f}% of average",
                        severity=min(10, volume_ratio),
                        payload={
                            'type': 'VOLUME_SPIKE',
                            'contract': contract.localSymbol,
                            'volume_ratio': volume_ratio
                        }
                    )

            # Check depth depletion
            if hasattr(ticker, 'bidSize') and hasattr(ticker, 'askSize'):
                if ticker.bidSize and ticker.askSize:
                    current_depth = ticker.bidSize + ticker.askSize
                    self.depth_history.append(current_depth)

                    if len(self.depth_history) >= 10:
                        avg_depth = sum(list(self.depth_history)[:-1]) / (len(self.depth_history) - 1)
                        if avg_depth > 0:
                            depth_ratio = current_depth / avg_depth
                            if depth_ratio < self.depth_drop_pct:
                                self.last_trigger_time = datetime.now()
                                return SentinelTrigger(
                                    reason=f"Liquidity Depletion: {depth_ratio*100:.0f}% of average",
                                    severity=min(10, (1 - depth_ratio) * 10),
                                    payload={
                                        'type': 'LIQUIDITY_DEPLETION',
                                        'contract': contract.localSymbol,
                                        'depth_ratio': depth_ratio
                                    }
                                )

        return None

    async def unsubscribe_all(self):
        """Unsubscribe from all market data."""
        for data in self.tickers.values():
            try:
                self.ib.cancelMktData(data['ticker'].contract)
            except Exception as e:
                logger.error(f"Unsubscribe error: {e}")
        self.tickers.clear()
