import logging
import json
import os
import asyncio
import re
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
        """
        Fetch volume with IB primary, YFinance fallback, -1 for unknown.

        For FOP (options) and BAG (combos), fetches volume from the UNDERLYING
        FUTURES contract using the 'underConId' hard link from ContractDetails.
        """
        from ib_insync import Contract, Bag

        # === RESOLVE TARGET CONTRACT FOR VOLUME CHECK ===
        target_contract = contract
        underlying_resolved = False

        if contract and contract.secType in ('FOP', 'BAG'):
            logger.debug(f"Contract is {contract.secType}, resolving underlying futures for volume check")
            try:
                # 1. Identify the FOP contract to query details from
                if contract.secType == 'BAG' and contract.comboLegs:
                    # For BAG: use first leg (should be an FOP)
                    first_leg_conid = contract.comboLegs[0].conId
                    query_contract = Contract(conId=first_leg_conid)
                else:
                    # For direct FOP: use the contract itself
                    query_contract = Contract(conId=contract.conId) if contract.conId else contract

                # 2. Get ContractDetails to find the Hard Link
                details_list = await ib.reqContractDetailsAsync(query_contract)

                if details_list:
                    fop_details = details_list[0]

                    # 3. Follow the hard link to the underlying future
                    # NOTE: ib_insync uses 'underConId' (not 'underlyingConId')
                    under_id = getattr(fop_details, 'underConId', 0)

                    if under_id:
                        fut_contract = Contract(conId=under_id)
                        qualified = await ib.qualifyContractsAsync(fut_contract)

                        if qualified:
                            target_contract = qualified[0]
                            underlying_resolved = True
                            logger.info(f"Resolved underlying future: {target_contract.localSymbol} (conId: {target_contract.conId})")
                        else:
                            logger.warning(f"Failed to qualify underlying future conId {under_id}")
                    else:
                        logger.warning(f"No underConId found in ContractDetails for {query_contract}")
                else:
                    logger.warning(f"No ContractDetails returned for {query_contract}")

            except Exception as e:
                logger.warning(f"Failed to resolve underlying for volume: {e}")

        # === PRIMARY: IB ===
        if target_contract and ib and ib.isConnected():
            try:
                bars = await ib.reqHistoricalDataAsync(
                    target_contract,
                    endDateTime='',
                    durationStr='900 S',
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=False
                )
                if bars:
                    total_vol = sum(b.volume for b in bars)
                    if total_vol > 0:
                        source = f"IB ({target_contract.localSymbol})" if underlying_resolved else "IB"
                        logger.info(f"Volume from {source}: {total_vol}")
                        return float(total_vol)
            except Exception as e:
                logger.warning(f"IB volume fetch failed: {e}")

        # === SECONDARY: YFinance (fallback) ===
        try:
            from trading_bot.utils import get_market_data_cached
            symbol = target_contract.symbol if target_contract else 'KC'
            yf_ticker = f"{symbol}=F"

            data = get_market_data_cached([yf_ticker], period="1d")
            if data is not None and not data.empty and 'Volume' in data.columns:
                last_vol = data['Volume'].iloc[-1]
                # Handle numpy scalar vs native float (avoids Pandas FutureWarning)
                if hasattr(last_vol, 'item'):
                    volume = float(last_vol.item())
                else:
                    volume = float(last_vol)

                if volume > 0:
                    logger.info(f"Volume from YFinance ({yf_ticker}): {volume}")
                    return volume
        except Exception as e:
            logger.warning(f"YFinance fallback failed: {e}")

        logger.warning("Volume unknown from all sources. Returning -1.")
        return -1.0

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

        if volume_15m < 0:
            return False, "REVIEW REQUIRED - Volume data unavailable. Manual approval needed."
        elif volume_15m == 0:
            return False, f"REJECTED - Article II: Zero volume for {symbol}. Market illiquid."
        elif qty > volume_15m * self.limits['max_volume_pct']:
            return False, f"REJECTED - Article II: Order qty {qty} exceeds {self.limits['max_volume_pct']:.0%} of volume ({volume_15m:.0f})."

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

            # --- FIX: Validate raw response ---
            if not response or not response.strip():
                logger.error(f"Compliance LLM returned empty response for {symbol}")
                return False, "Compliance Error: Empty LLM response (fail-closed)"

            text = response.strip()

            # --- FIX: Regex extraction for chatty LLMs ---
            # Try to find JSON object even if surrounded by prose
            json_match = re.search(r'\{[^{}]*"approved"[^{}]*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
            else:
                # Fallback: try to find ANY JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    text = json_match.group(0)
            # --- END Regex extraction ---

            # Clean markdown code blocks (existing logic)
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            text = text.strip()

            # Validate cleaned text
            if not text:
                logger.error(f"Compliance response empty after cleaning for {symbol}. Raw: {response[:500]}")
                return False, "Compliance Error: Empty JSON after cleanup (fail-closed)"

            result = json.loads(text)
            if not result.get('approved'):
                return False, result.get('reason', 'Vetoed')

            return True, "Approved"

        except json.JSONDecodeError as e:
            logger.error(f"Compliance Guardian JSON parse failed: {e}. Raw response: {response[:500] if response else 'None'}")
            return False, f"Compliance Error: Invalid JSON response (fail-closed)"
        except Exception as e:
            logger.error(f"Compliance Guardian failed: {e}", exc_info=True)
            return False, f"Compliance Error: {e}"

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
