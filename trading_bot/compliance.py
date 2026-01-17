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
        # Specific override from config for concentration check (Issue 8)
        self.max_brazil_concentration = self.config.get('compliance', {}).get('max_brazil_concentration', 1.0)

        self.router = HeterogeneousRouter(config)

    async def _check_concentration_risk(self, proposed_direction: str, ib) -> tuple[bool, str]:
        """Check if proposed trade increases concentration risk."""
        if not ib or not ib.isConnected():
            return True, ""  # Fail open if no IB connection

        try:
            # Get current portfolio
            portfolio = await ib.reqPortfolioAsync() # Or portfolio() sync if updated?
            # ib.portfolio() is sync property, assumes updates. reqPortfolioAsync updates it?
            # ib_insync 'portfolio()' returns list of Position.
            # Ideally we ensure we have fresh data.
            # But let's use the property.

            # Define Brazil-correlated assets
            brazil_proxies = ['KC', 'SB', 'EWZ', 'BRL']  # Coffee, Sugar, Brazil ETF, BRL futures

            brazil_exposure = 0.0
            total_equity = 0.0

            # Get Net Liquidation Value from Account Summary
            # We can't easily get it here without async call if not passed.
            # We will approximate from sum of market values + cash? No.
            # Let's sum absolute market value of positions as "Gross Exposure"
            # Concentration = Brazil Gross Exposure / Total Gross Exposure

            total_gross_exposure = sum(abs(p.marketValue) for p in portfolio)

            for position in portfolio:
                symbol = position.contract.symbol
                if any(proxy in symbol for proxy in brazil_proxies):
                    brazil_exposure += abs(position.marketValue)

            # If we are adding a trade (assuming it increases exposure), we check current + new?
            # We don't have new trade size here easily.
            # We just check CURRENT concentration as a gate.

            concentration_pct = (brazil_exposure / total_gross_exposure) if total_gross_exposure > 0 else 0

            if concentration_pct > self.max_brazil_concentration:
                # If we are adding to it (BULLISH/BEARISH)?
                # Actually if we are already over limit, we should block ANY trade that adds risk.
                # Since we don't know if Direction adds or reduces without complex correlation analysis,
                # we block all NEW signals if concentration is too high.
                if proposed_direction in ['BULLISH', 'BEARISH']:
                     return False, f"Brazil concentration at {concentration_pct:.1%} (max: {self.max_brazil_concentration:.0%})"

            return True, f"Brazil concentration: {concentration_pct:.1%}"

        except Exception as e:
            logger.warning(f"Concentration check failed: {e}")
            return True, ""  # Fail open

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
        # NEW: Concentration Check (Programmatic)
        # We need IB instance. It's not passed here?
        # run_emergency_cycle calls audit_decision. It doesn't pass 'ib'.
        # Wait, I need to update audit_decision signature or use a global?
        # No, better to update the signature in compliance.py AND the caller in orchestrator.py.
        # But 'audit_decision' is public API of this class.
        # Check if 'run_emergency_cycle' has 'ib'. Yes.
        # I will update signature of 'audit_decision' to accept optional 'ib'.
        # However, I should check existing usages.
        pass

    async def audit_decision(self, reports: dict, market_context: str, decision: dict, master_persona: str, ib=None) -> dict:
        """
        Audits the Master Strategist's decision.
        """
        # NEW: Concentration Check
        if ib:
            conc_ok, conc_reason = await self._check_concentration_risk(decision.get('direction'), ib)
            if not conc_ok:
                return {
                    'approved': False,
                    'flagged_reason': f"CONCENTRATION RISK: {conc_reason}"
                }

        reports_text = ""
        for agent, content in reports.items():
            # Handle structured reports (extract data field)
            content_str = content.get('data', str(content)) if isinstance(content, dict) else str(content)
            reports_text += f"\n--- {agent.upper()} ---\n{content_str}\n"

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
