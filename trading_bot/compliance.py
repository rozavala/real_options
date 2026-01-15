import logging
import json
import os
import asyncio
from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole

logger = logging.getLogger(__name__)

class ComplianceGuardian:
    """Constitutional AI-based compliance checker with veto power."""

    CONSTITUTION = """
    Article I (Capital Preservation): No single position shall exceed {max_position_pct:.1%} of equity.
    Article II (Liquidity Safety): No order shall exceed {max_volume_pct:.1%} of 15-min avg volume.
    Article III (Stop-Loss Mandate): Every entry must have a defined stop-loss logic.
    Article IV (Concentration): Total Brazil exposure shall not exceed {max_brazil_pct:.0%}.
    Article V (VaR Limit): Daily VaR shall not exceed {var_limit_pct:.1%} of equity.
    """

    def __init__(self, config: dict):
        self.config = config
        self.limits = config.get('compliance', {})
        # Ensure defaults
        self.limits.setdefault('max_position_pct', 0.05)
        self.limits.setdefault('max_volume_pct', 0.10)
        self.limits.setdefault('max_brazil_pct', 0.30)
        self.limits.setdefault('var_limit_pct', 0.01)

        self.router = HeterogeneousRouter(config)

    async def _fetch_volume_stats(self, ib, contract) -> float:
        """Fetch 15-min average volume for Article II check."""
        try:
            if not ib or not contract:
                return 0.0

            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='900 S', # 15 mins
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False
            )
            if not bars:
                return 0.0

            total_vol = sum(b.volume for b in bars)
            return float(total_vol)
        except Exception as e:
            logger.warning(f"Failed to fetch volume stats: {e}")
            return 0.0

    async def review_order(self, order_context: dict) -> tuple[bool, str]:
        """
        Review proposed order against constitutional rules.
        Returns: (approved: bool, reason: str)
        """
        # Extract context
        ib = order_context.get('ib')
        contract = order_context.get('contract')
        symbol = order_context.get('symbol', 'Unknown')
        qty = order_context.get('order_quantity', 0)
        equity = order_context.get('account_equity', 100000.0)

        # 1. Gather Data for Constitution
        volume_15m = await self._fetch_volume_stats(ib, contract)

        # Prepare Review Packet
        review_packet = {
            "proposed_order": {
                "symbol": symbol,
                "quantity": qty,
                "estimated_notional": qty * 37500 * (order_context.get('price', 2.0)), # Estimate
            },
            "market_data": {
                "volume_15min_total": volume_15m,
                "volume_limit_check": f"{qty} vs {volume_15m * self.limits['max_volume_pct']:.1f}",
            },
            "portfolio_state": {
                "equity": equity,
                "current_positions": order_context.get('total_position_count', 0),
                "trend_24h": f"{order_context.get('market_trend_pct', 0):.2%}"
            }
        }

        prompt = f"""
        You are the Chief Compliance Officer. Your ONLY concern is capital preservation.

        CONSTITUTION:
        {self.CONSTITUTION.format(**self.limits)}

        REVIEW PACKET:
        {json.dumps(review_packet, indent=2)}

        TASK: Review this order against the Constitution.
        1. Check Article II: Is Quantity ({qty}) > {self.limits['max_volume_pct']:.0%} of Volume ({volume_15m})?
        2. Check Article I: Is Notional > {self.limits['max_position_pct']:.0%} of Equity ({equity})?

        OUTPUT JSON: {{"approved": bool, "reason": string}}
        """

        try:
            response = await self.router.route(
                AgentRole.COMPLIANCE_OFFICER,
                prompt,
                response_json=True
            )

            text = response.strip()
            # Clean potential markdown
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]

            result = json.loads(text)
            if not result.get('approved'):
                return False, result.get('reason', 'Vetoed')

            return True, "Approved"

        except Exception as e:
            logger.error(f"Compliance Guardian failed: {e}")
            # Fail closed
            return False, f"Compliance Error: {e}"

    async def audit_decision(self, reports: dict, market_context: str, decision: dict, master_persona: str) -> dict:
        """
        Audits the Master Strategist's decision against the reports to prevent hallucinations (Signal Generation Stage).
        """
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
            response = await self.router.route(
                AgentRole.COMPLIANCE_OFFICER,
                prompt,
                response_json=True
            )
            text = response.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            return json.loads(text)
        except Exception as e:
            logger.error(f"Compliance Audit failed: {e}")
            return {'approved': False, 'flagged_reason': f"Audit Error: {str(e)}"}
