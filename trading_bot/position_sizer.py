from ib_insync import IB
import logging

logger = logging.getLogger(__name__)

class DynamicPositionSizer:
    def __init__(self, config: dict):
        self.base_qty = config.get('strategy', {}).get('quantity', 1)
        # Default max heat 25% if not set
        self.max_portfolio_heat = config.get('risk_management', {}).get('max_heat_pct', 0.25)

    async def calculate_size(
        self,
        ib: IB,
        signal: dict,
        volatility_sentiment: str,
        account_value: float
    ) -> int:
        """Dynamic sizing based on conviction and volatility."""

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

        # Portfolio heat check
        # We define exposure as total value of KC/KO positions
        positions = await ib.reqPositionsAsync()
        current_exposure = sum(
            abs(p.position * p.avgCost)
            for p in positions
            if 'KC' in p.contract.symbol or 'KO' in p.contract.symbol
        )

        heat_ratio = current_exposure / account_value if account_value > 0 else 0

        heat_multiplier = 1.0
        if heat_ratio > self.max_portfolio_heat:
            heat_multiplier = 0.5  # Reduce if already hot
            logger.warning(f"Portfolio Heat {heat_ratio:.1%} > {self.max_portfolio_heat:.1%}. Reducing size.")

        raw_qty = self.base_qty * confidence_multiplier * vol_multiplier * heat_multiplier
        final_qty = int(raw_qty)

        logger.info(f"Dynamic Sizing: Base={self.base_qty} * Conf({confidence_multiplier:.2f}) * Vol({vol_multiplier}) * Heat({heat_multiplier}) = {final_qty}")

        return max(1, final_qty)  # Minimum 1 contract
