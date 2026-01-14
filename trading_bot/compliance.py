import logging
import json
import os
import asyncio
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class ComplianceOfficer:
    """
    The 'Constitution': A hard-coded safety layer that runs before trade execution.
    Implements the Veto Checklist: Liquidity, Position Limits, and Sanity Checks.
    """

    def __init__(self, config: dict):
        self.config = config
        self.compliance_config = config.get('compliance', {})
        self.personas = config.get('gemini', {}).get('personas', {})

        # 1. Setup API Key
        api_key = config.get('gemini', {}).get('api_key')
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            api_key = os.environ.get('GEMINI_API_KEY')

        self.client = genai.Client(api_key=api_key)

        # 2. Model Selection (Strictly Separate from Agent Model)
        self.model_name = self.compliance_config.get('model', 'gemini-1.5-pro')
        self.temperature = self.compliance_config.get('temperature', 0.0)

        logger.info(f"Compliance Officer initialized with model: {self.model_name}")

    async def review(self, decision: dict, order_context: dict) -> str:
        """
        Reviews a proposed trade against the Veto Checklist.

        Args:
            decision (dict): The Master Strategist's decision (reasoning, direction, confidence).
            order_context (dict): Context including 'spread_width', 'total_position_count', 'market_trend_pct'.

        Returns:
            str: "APPROVED" or "VETOED: <Reason>"
        """
        contract_symbol = order_context.get('symbol', 'Unknown')
        logger.info(f"Compliance Review for {contract_symbol}...")

        # --- CHECK 1: POSITION LIMIT ---
        # "Does this trade put us over 5 contracts total?"
        # Assuming we trade 1 contract per order.
        current_count = order_context.get('total_position_count', 0)
        max_positions = 5 # Hard limit from brief

        if current_count >= max_positions:
            msg = f"VETOED: Position Limit Exceeded ({current_count} >= {max_positions})."
            logger.warning(msg)
            return msg

        # --- CHECK 2: LIQUIDITY CHECK ---
        # "Is the bid/ask spread > 5 ticks?"
        # 1 Tick for Coffee (KC) = 0.05 cents. 5 Ticks = 0.25 cents.
        # However, for spreads/combos, 'ticks' might be defined differently.
        # Let's interpret '5 ticks' as a raw value based on the instrument.
        # For simplicity, let's use the 'spread_width' passed in context (in cents).

        spread_width = order_context.get('bid_ask_spread', 0.0)
        max_spread_ticks = 5
        tick_size = 0.05 # KC Tick Size
        max_spread_val = max_spread_ticks * tick_size

        if spread_width > max_spread_val:
            # Check if config allows override, otherwise Veto.
            # Brief says "If YES, VETO." (Hard Rule)
            msg = f"VETOED: Liquidity Gap. Spread {spread_width:.2f} > {max_spread_val:.2f} (5 ticks)."
            logger.warning(msg)
            return msg

        # --- CHECK 3: SANITY CHECK (The "Constitution") ---
        # "Master says BUY, but trend is down -2%... Is there a specific reason?"
        trend_pct = order_context.get('market_trend_pct', 0.0)
        direction = decision.get('direction', 'NEUTRAL')
        reasoning = decision.get('reasoning', '')

        # We use the LLM to verify if the reasoning justifies fighting the trend.
        audit_result = await self._audit_logic(direction, reasoning, trend_pct)

        if not audit_result['approved']:
            msg = f"VETOED: Sanity Check Failed. {audit_result['flagged_reason']}"
            logger.warning(msg)
            return msg

        logger.info(f"Compliance Review Passed for {contract_symbol}.")
        return "APPROVED"

    async def _audit_logic(self, direction: str, reasoning: str, trend_pct: float) -> dict:
        """Uses the Compliance LLM to audit the logic."""

        prompt = (
            f"You are the Constitution of the Trading Fund.\n"
            f"Your job is to VETO dangerous trades that fight the trend without specific justification.\n\n"
            f"SCENARIO:\n"
            f"- Proposed Action: {direction}\n"
            f"- Market Trend (24h): {trend_pct:.2%}\n"
            f"- Strategist Reasoning: \"{reasoning}\"\n\n"
            f"RULES:\n"
            f"1. IF Action is BULLISH (BUY) AND Trend is < -1.5%:\n"
            f"   - VETO unless reasoning explicitly cites a 'Supply Shock', 'Frost', 'Strike', or 'Weather Event'.\n"
            f"   - Vague terms like 'oversold' or 'technical bounce' are NOT enough to catch a falling knife.\n"
            f"2. IF Action is BEARISH (SELL) AND Trend is > 1.5%:\n"
            f"   - VETO unless reasoning explicitly cites 'Inventory Build', 'Demand Collapse', or 'Macro/Currency' shift.\n"
            f"3. OTHERWISE: Approve.\n\n"
            f"OUTPUT JSON: {{'approved': bool, 'flagged_reason': string}}"
        )

        try:
            # Retry logic
            for attempt in range(3):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            response_mime_type="application/json"
                        )
                    )
                    return json.loads(response.text)
                except Exception as e:
                    if "429" in str(e):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise e

        except Exception as e:
            logger.error(f"Compliance LLM Audit failed: {e}")
            # Fail Open or Fail Closed?
            # Brief says "Hard-coded safety layer". If safety fails, we should probably Block?
            # Or assume safe?
            # Let's Fail Closed (Veto) for safety if we can't verify.
            return {'approved': False, 'flagged_reason': "Audit System Error"}

        return {'approved': False, 'flagged_reason': "Audit Failed (Unknown)"}
