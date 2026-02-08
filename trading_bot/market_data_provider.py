"""
Market Data Provider — Commodity-Agnostic IBKR Integration

Replaces the ML inference pipeline with live IBKR data.
Provides market_context dicts that the Council uses for decision-making.

This module is the ONLY source of market data for the signal generator.
All data comes from IBKR — no external APIs, no commodity-specific logic.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional
from ib_insync import IB, Contract, Future

from trading_bot.utils import get_active_ticker
from trading_bot.weighted_voting import RegimeDetector

logger = logging.getLogger(__name__)

# Constants
SMA_LOOKBACK_DAYS = 200
VOLATILITY_LOOKBACK_DAYS = 5
HISTORICAL_DATA_DURATION = "1 Y"  # Fetch 1 year for SMA-200


async def build_market_context(
    ib: IB,
    contract: Future,
    config: dict
) -> dict:
    """
    Build a commodity-agnostic market context dict for a single contract.

    The returned dict has the same keys the Council expects
    (agents.py and signal_generator.py).

    Args:
        ib: Connected IB instance
        contract: The futures contract to analyze
        config: Application config

    Returns:
        dict with keys: price, sma_200, regime, action, confidence,
                        expected_price, predicted_return, reason,
                        volatility_5d, price_vs_sma
    """
    contract_name = f"{contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]})"
    logger.info(f"Building market context for {contract_name}...")

    # --- 1. Get Current Price ---
    price = await _get_current_price(ib, contract)
    if price is None:
        logger.warning(f"Could not get price for {contract_name}. Using NaN context.")
        return _empty_market_context(contract, reason="Price unavailable from IBKR")

    # --- 2. Get Historical Data for Indicators ---
    sma_200, volatility_5d = await _get_technical_indicators(ib, contract)

    # --- 3. Detect Regime ---
    regime = await RegimeDetector.detect_regime(ib, contract)
    if regime == "UNKNOWN":
        # Fallback: simple regime from price vs SMA
        if sma_200 and sma_200 > 0:
            regime = "TRENDING" if abs((price - sma_200) / sma_200) > 0.05 else "RANGE_BOUND"

    # --- 4. Compute Price-vs-SMA Relationship ---
    price_vs_sma = None
    if sma_200 and sma_200 > 0:
        price_vs_sma = (price - sma_200) / sma_200  # positive = above SMA

    # --- 5. Build Context Dict ---
    # Keys match what Council/agents.py/weighted_voting.py expect
    context = {
        # === Required by signal_generator / agents ===
        "price": price,
        "sma_200": sma_200,
        "expected_price": price,        # No prediction — use current price
        "predicted_return": 0.0,        # Reserved for future use
        "action": "NEUTRAL",            # No ML prior — Council decides
        "confidence": 0.5,              # Neutral prior
        "reason": "Council-only mode: Market data from IBKR",
        "regime": regime,

        # === NEW: Enriched context for better Council decisions ===
        "volatility_5d": volatility_5d,
        "price_vs_sma": price_vs_sma,
        "data_source": "IBKR_LIVE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"Market context for {contract_name}: "
        f"price={price:.2f}, sma_200={sma_200}, regime={regime}, "
        f"vol_5d={volatility_5d:.4f}" if volatility_5d else
        f"Market context for {contract_name}: price={price:.2f}, regime={regime}"
    )

    return context


async def build_all_market_contexts(
    ib: IB,
    contracts: list[Future],
    config: dict
) -> list[dict]:
    """
    Build market context for all active contracts in parallel.

    Args:
        ib: Connected IB instance
        contracts: List of active futures contracts (sorted chronologically)
        config: Application config

    Returns:
        List of market_context dicts, one per contract, in same order
    """
    logger.info(f"Building market context for {len(contracts)} contracts...")

    # Use semaphore to avoid overwhelming IBKR with concurrent requests
    sem = asyncio.Semaphore(3)

    async def _build_with_semaphore(contract):
        async with sem:
            try:
                return await build_market_context(ib, contract, config)
            except Exception as e:
                logger.error(f"Failed to build context for {contract.localSymbol}: {e}")
                return _empty_market_context(contract, reason=f"Error: {e}")

    contexts = await asyncio.gather(
        *[_build_with_semaphore(c) for c in contracts]
    )

    successful = sum(1 for c in contexts if c.get("data_source") == "IBKR_LIVE")
    logger.info(f"Market context complete: {successful}/{len(contracts)} contracts successful")

    return list(contexts)


def format_market_context_for_prompt(market_context: dict) -> str:
    """
    Format market context as a human-readable string for agent prompts.
    """
    price = market_context.get('price', 'N/A')
    sma = market_context.get('sma_200')
    regime = market_context.get('regime', 'UNKNOWN')
    vol = market_context.get('volatility_5d')
    price_vs_sma = market_context.get('price_vs_sma')

    lines = [
        f"Current Price: {price:.2f}" if isinstance(price, (int, float)) else f"Current Price: {price}",
    ]

    if sma and isinstance(sma, (int, float)):
        sma_relation = "ABOVE" if price > sma else "BELOW"
        pct = abs(price_vs_sma * 100) if price_vs_sma else 0
        lines.append(f"200-day SMA: {sma:.2f} (Price is {pct:.1f}% {sma_relation})")

    lines.append(f"Market Regime: {regime}")

    if vol and isinstance(vol, (int, float)):
        vol_label = "HIGH" if vol > 0.03 else "MODERATE" if vol > 0.015 else "LOW"
        lines.append(f"5-day Volatility: {vol:.2%} ({vol_label})")

    return "\n".join(lines)


# === Private Helpers ===

async def _get_current_price(ib: IB, contract: Future) -> Optional[float]:
    """Fetch current price from IBKR with timeout."""
    try:
        ticker = ib.reqMktData(contract, '', False, False)
        # Wait for price with timeout
        for _ in range(50):  # 5 seconds max
            await asyncio.sleep(0.1)
            if not _is_nan(ticker.last) and ticker.last > 0:
                price = ticker.last
                ib.cancelMktData(contract)
                return price
            if not _is_nan(ticker.close) and ticker.close > 0:
                price = ticker.close
                ib.cancelMktData(contract)
                return price

        ib.cancelMktData(contract)

        # Fallback to delayed/historical
        if not _is_nan(ticker.close) and ticker.close > 0:
            return ticker.close

        logger.warning(f"No price available for {contract.localSymbol}")
        return None

    except Exception as e:
        logger.error(f"Error fetching price for {contract.localSymbol}: {e}")
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass
        return None


async def _get_technical_indicators(
    ib: IB,
    contract: Future
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute SMA-200 and 5-day volatility from IBKR historical bars.

    Returns:
        (sma_200, volatility_5d) — either can be None if insufficient data
    """
    sma_200 = None
    volatility_5d = None

    try:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=HISTORICAL_DATA_DURATION,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )

        if not bars or len(bars) < 5:
            logger.warning(f"Insufficient historical data for {contract.localSymbol}: {len(bars) if bars else 0} bars")
            return None, None

        closes = [bar.close for bar in bars]

        # SMA-200
        if len(closes) >= SMA_LOOKBACK_DAYS:
            sma_200 = sum(closes[-SMA_LOOKBACK_DAYS:]) / SMA_LOOKBACK_DAYS
        elif len(closes) >= 50:
            # Fallback: use available data for SMA (log warning)
            sma_200 = sum(closes) / len(closes)
            logger.info(f"Using SMA-{len(closes)} (not enough data for SMA-200) for {contract.localSymbol}")

        # 5-day volatility
        if len(closes) >= VOLATILITY_LOOKBACK_DAYS + 1:
            recent = closes[-(VOLATILITY_LOOKBACK_DAYS + 1):]
            returns = [(recent[i] - recent[i-1]) / recent[i-1] for i in range(1, len(recent))]
            volatility_5d = (sum(r**2 for r in returns) / len(returns)) ** 0.5

        return sma_200, volatility_5d

    except Exception as e:
        logger.error(f"Error computing indicators for {contract.localSymbol}: {e}")
        return None, None


def _empty_market_context(contract: Future, reason: str = "Data unavailable") -> dict:
    """Return a safe default context when data fetch fails."""
    return {
        "price": None,
        "sma_200": None,
        "expected_price": None,
        "predicted_return": 0.0,
        "action": "NEUTRAL",
        "confidence": 0.0,       # Zero confidence = Council won't trade on bad data
        "reason": reason,
        "regime": "UNKNOWN",
        "volatility_5d": None,
        "price_vs_sma": None,
        "data_source": "FALLBACK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _is_nan(value) -> bool:
    """Check if a value is NaN (handles ib_insync's NaN values)."""
    try:
        import math
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True
