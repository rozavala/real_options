import logging
import json
import os
import asyncio
import re
from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole

logger = logging.getLogger(__name__)

# Module-level cache for ContractDetails (prevents redundant API calls during multi-signal bursts)
_CONTRACT_DETAILS_CACHE: dict = {}

async def calculate_spread_max_risk(ib, contract, order, config: dict) -> float:
    """
    Calculate the maximum capital at risk for an option spread.

    ARCHITECTURAL MANDATE (Flight Director):
    - NO PROXIES for wing width calculation
    - Must fetch actual ContractDetails for each leg
    - Calculate true strike differences
    - Use asyncio.gather() for parallel fetching (latency mitigation)
    - Cache ContractDetails to prevent redundant API calls

    ⚠️ SERIALIZATION WARNING (Flight Director Final Review):
    The asyncio.gather() below is ONLY safe because:
    1. The order_manager loop is sequential (enforced by _CAPITAL_LOCK)
    2. This function is called one order at a time
    DO NOT call this function concurrently from multiple tasks on the same `ib` connection.

    For DEBIT spreads (Bull Call, Bear Put): Max Risk = Premium Paid
    For CREDIT spreads (Iron Condor, Credit Spread): Max Risk = Wing Width - Credit

    Returns the dollar value of maximum loss.
    """
    from ib_insync import Bag, Contract

    multiplier = 37500  # Coffee options multiplier (lbs per contract)
    qty = order.totalQuantity
    net_price = abs(order.lmtPrice)  # In cents/lb for KC options

    # FIX: Convert multiplier to dollars (cents/lb -> dollars/contract)
    # 37,500 lbs * (1/100 dollars/cent) = 375 dollars per cent move
    dollar_multiplier = multiplier / 100

    # For non-BAG contracts, max risk = premium paid
    if not isinstance(contract, Bag) or not contract.comboLegs:
        return qty * net_price * dollar_multiplier

    # === FLIGHT DIRECTOR MANDATE: Calculate TRUE wing width from actual strikes ===
    try:
        # === LATENCY MITIGATION: Parallel fetch with caching ===
        async def get_leg_details(leg):
            """Fetch leg details with caching."""
            cache_key = leg.conId
            if cache_key in _CONTRACT_DETAILS_CACHE:
                logger.debug(f"Cache hit for conId {cache_key}")
                return _CONTRACT_DETAILS_CACHE[cache_key]

            details = await ib.reqContractDetailsAsync(Contract(conId=leg.conId))
            if details:
                leg_contract = details[0].contract
                result = {
                    'conId': leg.conId,
                    'strike': leg_contract.strike,
                    'right': leg_contract.right,  # 'C' or 'P'
                    'action': leg.action  # 'BUY' or 'SELL'
                }
                _CONTRACT_DETAILS_CACHE[cache_key] = result
                return result
            return None

        # Fetch ALL legs in parallel using asyncio.gather (Flight Director mandate)
        leg_details_raw = await asyncio.gather(
            *[get_leg_details(leg) for leg in contract.comboLegs],
            return_exceptions=True
        )

        # Filter out failures
        leg_details = [ld for ld in leg_details_raw if ld and not isinstance(ld, Exception)]

        if len(leg_details) != len(contract.comboLegs):
            failed_count = len(contract.comboLegs) - len(leg_details)
            logger.warning(f"Could not resolve {failed_count} leg(s). Using premium-based fallback.")
            return qty * net_price * dollar_multiplier * 2  # Conservative fallback

        # Separate legs by right (Calls vs Puts)
        calls = [l for l in leg_details if l['right'] == 'C']
        puts = [l for l in leg_details if l['right'] == 'P']

        # Calculate wing width for each side
        call_width = 0.0
        put_width = 0.0

        if len(calls) >= 2:
            call_strikes = sorted([l['strike'] for l in calls])
            call_width = call_strikes[-1] - call_strikes[0]
            logger.debug(f"Call wing width: {call_width} (strikes: {call_strikes})")

        if len(puts) >= 2:
            put_strikes = sorted([l['strike'] for l in puts])
            put_width = put_strikes[-1] - put_strikes[0]
            logger.debug(f"Put wing width: {put_width} (strikes: {put_strikes})")

        # Use the maximum wing width (for Iron Condors, both should be equal)
        wing_width = max(call_width, put_width)

        if wing_width == 0:
            # Straddle or single-strike strategy - risk is premium paid
            logger.info(f"Zero wing width detected (straddle?). Max risk = premium.")
            return qty * net_price * dollar_multiplier

        # Determine if DEBIT or CREDIT spread based on order action
        if order.action == 'BUY':
            # DEBIT SPREAD: Max loss = premium paid
            max_risk = qty * net_price * dollar_multiplier
            logger.info(f"DEBIT spread: Max risk = ${max_risk:,.2f} (premium paid)")
        else:
            # CREDIT SPREAD: Max loss = (wing_width - credit) * multiplier
            # wing_width is in price points (cents/lb for coffee)
            max_risk = qty * (wing_width - net_price) * dollar_multiplier
            # Floor at zero (can't have negative risk)
            max_risk = max(max_risk, 0)
            logger.info(f"CREDIT spread: Wing width={wing_width}, Credit={net_price}, Max risk=${max_risk:,.2f}")

        return max_risk

    except Exception as e:
        logger.error(f"Error calculating true wing width: {e}. Using conservative fallback.")
        # Conservative fallback: 2x premium for credit spreads, 1x for debit
        if order.action == 'SELL':
            return qty * net_price * dollar_multiplier * 3  # Very conservative for credits
        return qty * net_price * dollar_multiplier

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
                # Issue #5: Suppress "No data of type EODChart" for FOPs
                if "No data of type EODChart" in str(e):
                    logger.debug(f"IB EOD data unavailable for {target_contract.localSymbol}, attempting fallback.")
                else:
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
        # Calculate actual capital at risk for spreads (not fictitious futures notional)
        order_obj = order_context.get('order_object')
        ib = order_context.get('ib')

        if order_obj and ib:
            actual_capital_at_risk = await calculate_spread_max_risk(
                ib,
                contract,
                order_obj,
                self.config
            )
        else:
            # Fallback: Use conservative estimate with warning
            actual_capital_at_risk = qty * 37500 * abs(order_context.get('price', 0.01)) * 2
            logger.warning(f"No order object or IB connection passed to compliance - using conservative fallback")

        review_packet = {
            "proposed_order": {
                "symbol": symbol,
                "quantity": qty,
                "max_capital_at_risk": actual_capital_at_risk,
                "limit_price": order_context.get('price', 0.0),
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

        # --- DETERMINISTIC PRE-CHECK (Skip LLM for obvious violations) ---
        limit_pct = self.limits.get('max_position_pct', 0.40)

        # FLIGHT DIRECTOR FIX: Use higher limit for defined-risk strategies (straddles)
        # Heuristic: If risk per contract > $10,000, it's likely a straddle/premium trade
        if actual_capital_at_risk > 0 and qty > 0:
            risk_per_contract = actual_capital_at_risk / qty
            if risk_per_contract > 10000:  # Likely straddle or large premium
                limit_pct = self.limits.get('max_straddle_pct', 0.55)
                logger.info(f"Using straddle limit ({limit_pct:.0%}) for high-premium trade")

        max_allowed = equity * limit_pct

        # === FLIGHT DIRECTOR MANDATE: Regime Incompatibility Warning ===
        capital_consumption_pct = actual_capital_at_risk / equity if equity > 0 else 1.0
        if capital_consumption_pct > 0.25:
            logger.warning(
                f"⚠️ REGIME INCOMPATIBILITY WARNING: Trade for {symbol} consumes "
                f"{capital_consumption_pct:.1%} of equity (${actual_capital_at_risk:,.2f} / ${equity:,.2f}). "
                f"Account may be under-capitalized for this strategy."
            )

        if actual_capital_at_risk > max_allowed:
            return False, (
                f"REJECTED - Article I: Max capital at risk (${actual_capital_at_risk:,.2f}) "
                f"exceeds {limit_pct:.0%} of equity (${max_allowed:,.2f})."
            )

        # === FIX-003: Align LLM instructions with deterministic code logic ===
        # The limit_pct variable already contains the correct limit (40% standard or 55% for straddles)
        # We MUST pass this to the LLM so both "brains" use the same threshold

        limit_type_desc = "straddle/high-premium override" if limit_pct > self.limits['max_position_pct'] else "standard position limit"

        prompt = f"""
        You are the Chief Compliance Officer. Your ONLY concern is capital preservation.

        CONSTITUTION:
        {self.CONSTITUTION.format(**self.limits)}

        REVIEW PACKET:
        {json.dumps(review_packet, indent=2)}

        APPLICABLE LIMIT FOR THIS TRADE: {limit_pct:.0%} of equity ({limit_type_desc})
        - Maximum Allowed Capital at Risk: ${max_allowed:,.2f}

        TASK: Review this order against the Constitution.
        1. Check Article II (Liquidity): Is Quantity ({qty}) > {self.limits['max_volume_pct']:.0%} of Volume ({volume_15m})?
        2. Check Article I (Capital Preservation): Is Max Capital at Risk (${actual_capital_at_risk:,.2f}) > {limit_pct:.0%} of Equity (${equity:,.2f})?
           - This trade consumes {capital_consumption_pct:.1%} of account equity
           - The applicable limit is {limit_pct:.0%} (NOT the base {self.limits['max_position_pct']:.0%})

        IMPORTANT: Use {limit_pct:.0%} as the threshold for Article I, not {self.limits['max_position_pct']:.0%}.

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
            logger.error(
                f"Compliance Guardian JSON parse failed for {symbol}. "
                f"Error at position {e.pos}: {e.msg}. "
                f"Response preview: {response[:500] if response else 'None'}..."
            )
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
