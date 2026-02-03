from ib_insync import IB
import logging
import math

logger = logging.getLogger(__name__)

class DynamicPositionSizer:
    def __init__(self, config: dict):
        self.config = config
        self.base_qty = config.get('strategy', {}).get('quantity', 1)
        # Default max heat 25% if not set
        self.max_portfolio_heat = config.get('risk_management', {}).get('max_heat_pct', 0.25)

    async def calculate_size(
        self,
        ib: IB,
        signal: dict,
        volatility_sentiment: str,
        account_value: float,
        conviction_multiplier: float = 1.0,  # v7.1: Consensus conviction scaling
    ) -> int:
        """Dynamic sizing based on conviction, volatility, and consensus.

        Args:
            conviction_multiplier: From Consensus Sensor.
                1.0 = vote aligns with Master (full size)
                0.75 = partial alignment
                0.5 = vote diverges (half size)
        """

        # Base size from confidence
        confidence = signal.get('confidence', 0.5)
        # 0.5 confidence -> 0.75x
        # 1.0 confidence -> 1.0x
        # Wait, user snippet was: 0.5 + (confidence * 0.5) -> range [0.5, 1.0]
        confidence_multiplier = 0.5 + (confidence * 0.5)

        # Volatility adjustment
        # BULLISH: Favorable vol (cheap options?), size up.
        # BEARISH: Unfavorable vol (expensive?), size down.
        vol_multiplier = {
            "BULLISH": 1.2,
            "NEUTRAL": 1.0,
            "BEARISH": 0.6
        }.get(volatility_sentiment, 1.0)

        # Portfolio heat check — commodity-agnostic (v7.1)
        # Match positions whose root symbol equals our active ticker.
        # IB contract.symbol is the root (e.g., 'KC' for coffee options/futures).
        from trading_bot.utils import get_active_ticker
        _active_symbol = get_active_ticker(self.config)

        positions = await ib.reqPositionsAsync()
        current_exposure = sum(
            abs(p.position * p.avgCost)
            for p in positions
            if p.contract.symbol == _active_symbol  # Exact match, not substring
        )

        heat_ratio = current_exposure / account_value if account_value > 0 else 0

        heat_multiplier = 1.0
        if heat_ratio > self.max_portfolio_heat:
            heat_multiplier = 0.5  # Reduce if already hot
            logger.warning(f"Portfolio Heat {heat_ratio:.1%} > {self.max_portfolio_heat:.1%}. Reducing size.")

        # v7.1: Apply consensus conviction scaling
        # When the vote diverges from the Master's direction, reduce size.
        raw_qty = self.base_qty * confidence_multiplier * vol_multiplier * heat_multiplier * conviction_multiplier
        # Use ceiling to be slightly more aggressive on sizing (strategy upgrade)
        final_qty = max(1, math.ceil(raw_qty))

        logger.info(
            f"Dynamic Sizing: Base={self.base_qty} * Conf({confidence_multiplier:.2f}) "
            f"* Vol({vol_multiplier}) * Heat({heat_multiplier}) "
            f"* Conviction({conviction_multiplier}) = {final_qty} (raw: {raw_qty:.3f})"
        )
        return final_qty
