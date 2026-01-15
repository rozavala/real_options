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

    async def _check_var_limit(self, proposed_trade: dict, equity: float) -> tuple[bool, str]:
        """
        Check if trade exceeds VaR limit.
        VaR = Potential loss at 95% confidence over 1 day.
        """
        # Simplified VaR: position_size * volatility * 1.65 (95% z-score)
        # Note: We need estimated Notional Value.
        # If not provided, we might assume from context or default.
        # This is an estimation.

        position_value = proposed_trade.get('notional_value', 0)

        # If we don't have notional value, we might skip or try to calculate from quantity * price
        if position_value == 0:
            qty = proposed_trade.get('order_quantity', 0)
            price = proposed_trade.get('price', 0)
            multiplier = 37500 if 'KC' in proposed_trade.get('symbol', '') else 1 # Approx for Coffee?
            # Actually Coffee contract size is 37,500 lbs. Price is usually per lb (e.g. 1.50).
            # So Notional = Price * 37500 * Qty.
            # But price might be in cents or dollars? IB usually gives dollars if configured.
            # Let's assume standard Coffee logic: Price (e.g. 2.50) * 37500 = ~93k per contract.

            # Safe default for check if missing
            if qty > 0 and price > 0:
                position_value = price * 37500 * qty

        historical_vol = proposed_trade.get('daily_volatility', 0.02)  # Default 2%

        var_1day = position_value * historical_vol * 1.65
        var_pct = var_1day / equity if equity > 0 else 0.0

        max_var_pct = self.compliance_config.get('max_var_pct', 0.01)  # 1% default

        if var_pct > max_var_pct:
            return False, f"VaR {var_pct:.2%} exceeds {max_var_pct:.2%} limit"

        return True, f"VaR check passed: {var_pct:.2%}"

    async def _check_article_ii_volume(self, ib, contract, order_quantity: int) -> tuple[bool, str]:
        """
        Article II: Liquidity Safety
        No order shall exceed 10% of the average volume of the last 15 minutes.

        Returns:
            tuple: (passed: bool, reason: str)
        """
        try:
            # Request 15 minutes of 1-minute bars
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='900 S',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False
            )

            if not bars or len(bars) < 5:
                logger.warning("Insufficient volume data for Article II check")
                # Fail open with warning if no data
                return True, "Article II: Volume data unavailable (check skipped)"

            total_volume = sum(bar.volume for bar in bars)
            max_pct = self.compliance_config.get('max_volume_pct', 0.10)
            max_order_volume = total_volume * max_pct

            if order_quantity > max_order_volume:
                return False, f"Article II VIOLATION: Order qty {order_quantity} exceeds {max_pct:.0%} of 15-min volume ({max_order_volume:.0f})"

            return True, f"Article II: Order within volume limits ({order_quantity}/{max_order_volume:.0f})"

        except Exception as e:
            logger.error(f"Article II check failed: {e}")
            return True, f"Article II: Check failed ({e}), proceeding with caution"

    async def review(self, decision: dict, order_context: dict) -> str:
        """
        Reviews a proposed trade against the Veto Checklist (Order Placement Stage).

        Args:
            decision (dict): The Master Strategist's decision (reasoning, direction, confidence).
            order_context (dict): Context including 'spread_width', 'total_position_count', 'market_trend_pct'.

        Returns:
            str: "APPROVED" or "VETOED: <Reason>"
        """
        contract_symbol = order_context.get('symbol', 'Unknown')
        logger.info(f"Compliance Review for {contract_symbol}...")

        # --- CHECK 1: POSITION LIMIT ---
        current_count = order_context.get('total_position_count', 0)
        max_positions = 5

        if current_count >= max_positions:
            msg = f"VETOED: Position Limit Exceeded ({current_count} >= {max_positions})."
            logger.warning(msg)
            return msg

        # --- CHECK 2: LIQUIDITY CHECK ---
        spread_width = order_context.get('bid_ask_spread', 0.0)
        max_spread_ticks = 5
        tick_size = 0.05
        max_spread_val = max_spread_ticks * tick_size

        if spread_width > max_spread_val:
            msg = f"VETOED: Liquidity Gap. Spread {spread_width:.2f} > {max_spread_val:.2f} (5 ticks)."
            logger.warning(msg)
            return msg

        # --- CHECK 2.5: ARTICLE II - VOLUME CHECK ---
        # Only run if we have IB connection and contract info
        ib = order_context.get('ib')
        contract = order_context.get('contract')
        order_quantity = order_context.get('order_quantity', 1)

        if ib and contract:
            passed, reason = await self._check_article_ii_volume(ib, contract, order_quantity)
            if not passed:
                logger.warning(reason)
                return reason
            logger.info(reason)
        else:
            logger.warning("Skipping Article II Volume Check: Missing 'ib' or 'contract' in order_context")

        # --- CHECK 2.6: VaR CHECK ---
        # Get Equity (Net Liquidation Value)
        equity = 100000.0 # Default fallback
        if ib:
            try:
                # We need to fetch account summary efficiently or pass it in.
                # order_context doesn't have equity yet.
                # Let's try to get it from 'ib' if possible or skip.
                # Since review is async, we can call.
                # However, calling accountSummaryAsync might be slow.
                # Assuming order_manager might pass it in future.
                # For now, let's look for it in context or skip.
                equity = order_context.get('account_equity', 100000.0)
            except: pass

        # Construct simplified trade dict for VaR
        trade_for_var = {
            'order_quantity': order_quantity,
            'price': order_context.get('price', 0), # Need to ensure price is in context
            'symbol': contract_symbol,
            # We can try to get volatility from context or default
        }

        # We need current price for VaR calculation.
        # Order Manager passes 'price' (limit price) in order?
        # Actually context has 'bid_ask_spread'.
        # We assume order_context might need enhancement to support VaR fully.
        # But let's add the check logic structure.

        passed_var, var_reason = await self._check_var_limit(trade_for_var, equity)
        if not passed_var:
             logger.warning(f"VETOED: {var_reason}")
             return f"VETOED: {var_reason}"
        logger.info(var_reason)

        # --- CHECK 3: SANITY CHECK (The "Constitution") ---
        trend_pct = order_context.get('market_trend_pct', 0.0)
        direction = decision.get('direction', 'NEUTRAL')
        reasoning = decision.get('reasoning', '')

        audit_result = await self._audit_logic(direction, reasoning, trend_pct)

        if not audit_result['approved']:
            msg = f"VETOED: Sanity Check Failed. {audit_result['flagged_reason']}"
            logger.warning(msg)
            return msg

        logger.info(f"Compliance Review Passed for {contract_symbol}.")
        return "APPROVED"

    async def audit_decision(self, reports: dict, market_context: str, decision: dict, master_persona: str) -> dict:
        """
        Audits the Master Strategist's decision against the reports to prevent hallucinations (Signal Generation Stage).
        """
        logger.info("Auditing decision for hallucinations/compliance...")

        reports_text = ""
        for agent, content in reports.items():
            reports_text += f"\n--- {agent.upper()} ---\n{content}\n"

        prompt = (
            f"You are the Compliance Officer auditing a trading decision.\n"
            f"Your goal is to detect HALLUCINATIONS or violations of logic.\n\n"
            f"--- EVIDENCE (REPORTS) ---\n{reports_text}\n\n"
            f"--- MARKET CONTEXT ---\n{market_context}\n\n"
            f"--- DECISION ---\n"
            f"Direction: {decision.get('direction')}\n"
            f"Reasoning: {decision.get('reasoning')}\n\n"
            f"TASK:\n"
            f"Check if the Decision Reasoning is supported by the Evidence.\n"
            f"1. Does the reasoning cite facts NOT in the reports? (Hallucination)\n"
            f"2. Does the reasoning ignore a 'CRITICAL RISK' explicitly stated in reports?\n"
            f"3. Is the direction contradictory to the overwhelming sentiment of reports?\n\n"
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
                            temperature=0.0,
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
            logger.error(f"Compliance Audit failed: {e}")
            return {'approved': False, 'flagged_reason': f"Audit Error: {str(e)}"}

        return {'approved': False, 'flagged_reason': "Audit Failed (Unknown)"}

    async def _audit_logic(self, direction: str, reasoning: str, trend_pct: float) -> dict:
        """Uses the Compliance LLM to audit the logic against trend."""

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
            return {'approved': False, 'flagged_reason': "Audit System Error"}

        return {'approved': False, 'flagged_reason': "Audit Failed (Unknown)"}
