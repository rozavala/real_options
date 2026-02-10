"""
Templatized Agent Prompts - Commodity-Agnostic Prompt Generation.

This module generates agent prompts by injecting CommodityProfile context
into base templates. The templates use simple string formatting (not Jinja2)
to minimize dependencies.
"""

from typing import Dict, Optional
from config.commodity_profiles import CommodityProfile


class AgentPromptTemplate:
    """
    Base templates for agent system prompts.

    Each template contains {placeholders} that get filled with
    commodity-specific context from the CommodityProfile.
    """

    BASE_SYSTEM_INSTRUCTION = """
You are {agent_name}, a specialist in {domain}.

{regime_context}

Your role: {role_description}

Constraints:
- Use ONLY the provided FACTS in your analysis
- Do not invent data or speculate beyond evidence
- Clearly separate facts from interpretation
- State confidence level (0.0-1.0) based on data quality

Output Format:
[EVIDENCE]: List all relevant facts with sources
[ANALYSIS]: Your interpretation and reasoning
[CONFIDENCE]: 0.0-1.0
[SENTIMENT TAG]: [SENTIMENT: BULLISH|BEARISH|NEUTRAL]
"""

    @staticmethod
    def render(agent_name: str, domain: str, role_description: str, regime_context: str = "") -> str:
        """
        Render the base prompt with optional regime context.

        Args:
            agent_name: Name of the agent
            domain: Domain expertise
            role_description: What this agent does
            regime_context: Current fundamental regime (from FundamentalRegimeSentinel)

        Returns:
            Rendered prompt string
        """
        return AgentPromptTemplate.BASE_SYSTEM_INSTRUCTION.format(
            agent_name=agent_name,
            domain=domain,
            role_description=role_description,
            regime_context=regime_context if regime_context else ""
        )

    # =========================================================================
    # AGRONOMIST (Weather/Crop Analyst)
    # =========================================================================

    AGRONOMIST_CONTEXT = """
## Key Production Regions
{regions_summary}

## Domain-Specific Risks
{agronomy_context}

## Your Task
Analyze the provided weather and crop data to assess:
1. Current weather conditions in key growing regions
2. Forecast risks (frost, drought, excess rain, disease pressure)
3. Impact on current crop vs. next season's production

## Specific Instructions
- Always consider the current agronomic stage (flowering, bean-filling, harvest)
- Distinguish between current-crop and next-year impacts
"""

    # =========================================================================
    # MACRO ANALYST
    # =========================================================================

    MACRO_CONTEXT = """
## Key Macro Drivers
{macro_context}

## Relevant Currencies
{currency_pairs}

## Your Task
Analyze the provided macro data to assess:
1. Currency trends affecting export competitiveness
2. Policy changes (trade, environmental, fiscal) impacting supply/demand
3. Economic indicators suggesting demand shifts
"""

    # =========================================================================
    # INVENTORY/FUNDAMENTALIST
    # =========================================================================

    INVENTORY_CONTEXT = """
## Key Data Sources
{inventory_sources}

## Supply Chain Context
{supply_chain_context}

## Operational Rules (MUST FOLLOW)
- ICE Certified Stocks INCREASING (building) = BEARISH
- ICE Certified Stocks DECREASING (drawing) = BULLISH
- Backwardation (nearby > deferred) = Supply tight = BULLISH
- Contango (nearby < deferred) = Supply ample = BEARISH

## Your Task
Analyze the provided inventory and curve data to assess:
1. Certified stock trends (direction and magnitude)
2. Forward curve structure and what it implies
3. Seasonal context (is this move normal for the time of year?)
"""

    # =========================================================================
    # SENTIMENT/COT ANALYST
    # =========================================================================

    SENTIMENT_CONTEXT = """
## Your Task
Analyze the provided COT and sentiment data to assess:
1. Non-commercial net position (long vs. short)
2. Changes in speculator positioning week-over-week
3. Crowded trade risks (extreme positioning)
4. News/social sentiment alignment with positioning

## Operational Rules
- Extreme net long + price at highs = Crowded, vulnerable to liquidation
- Extreme net short + price at lows = Crowded, short-covering risk
- Divergence between positioning and sentiment = Caution signal
"""

    # =========================================================================
    # TECHNICAL ANALYST
    # =========================================================================

    TECHNICAL_CONTEXT = """
## Contract Specifications
- Symbol: {contract_symbol}
- Exchange: {contract_exchange}
- Tick Size: {tick_size} {unit}
- Contract Months: {contract_months}

## Your Task
Analyze the provided price and indicator data to assess:
1. Current trend (higher highs/lows vs. lower highs/lows)
2. Key support and resistance levels
3. Momentum indicators (RSI, MACD, Stochastic)
4. Volume confirmation of price moves
"""

    # =========================================================================
    # VOLATILITY ANALYST
    # =========================================================================

    VOLATILITY_CONTEXT = """
## Contract Specifications
- Symbol: {contract_symbol}
- High IV Rank Threshold: {high_iv_rank}
- Low IV Rank Threshold: {low_iv_rank}

## Volatility Regimes
- HIGH_VOL (IV Rank > {high_iv_rank}): Premium is rich, favor selling strategies
- LOW_VOL (IV Rank < {low_iv_rank}): Premium is cheap, favor buying strategies
- NORMAL: Standard regime, direction-dependent strategies

## Strategy Mapping
| Regime | Directional View | Recommended Strategy |
|--------|-----------------|---------------------|
| HIGH_VOL | Neutral | IRON_CONDOR |
| HIGH_VOL | Bullish | BULL_PUT_SPREAD |
| HIGH_VOL | Bearish | BEAR_CALL_SPREAD |
| LOW_VOL | Neutral | LONG_STRADDLE |
| LOW_VOL | Bullish | LONG_CALL |
| LOW_VOL | Bearish | LONG_PUT |

## Your Task
Analyze the provided options data to assess:
1. Current IV rank and percentile
2. Skew analysis (puts vs. calls)
3. Regime classification
4. Strategy recommendation based on regime + directional view
"""

    # =========================================================================
    # GEOPOLITICAL ANALYST
    # =========================================================================

    GEOPOLITICAL_CONTEXT = """
## Key Producing Regions
{regions_summary}

## Key Logistics Hubs
{logistics_summary}

## Your Task
Analyze the provided news and data to assess:
1. Political stability in key producing countries
2. Trade policy changes affecting {commodity_name}
3. Labor/logistics disruption risks
4. Regulatory developments (environmental, trade)
"""

    # =========================================================================
    # SUPPLY CHAIN / LOGISTICS ANALYST
    # =========================================================================

    LOGISTICS_CONTEXT = """
## Key Logistics Hubs
{logistics_summary}

## Alert Thresholds
{logistics_thresholds}

## Your Task
Analyze the provided logistics data to assess:
1. Port congestion levels vs. normal
2. Vessel queue times and dwell times
3. Freight rate trends
4. Transit disruptions affecting delivery

## Operational Rules
- Congestion above threshold = Supply constraint = BULLISH
- Freight rates rising sharply = Cost pressure = BULLISH (short-term)
- Transit disruption (canal closure) = Delayed delivery = BULLISH
"""

    # =========================================================================
    # RENDERING METHODS
    # =========================================================================

    @classmethod
    def render_agronomist(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render agronomist prompt with commodity context."""
        regions_summary = "\n".join([
            f"- {r.name} ({r.country}): {r.weight*100:.0f}% of production. "
            f"Harvest: {cls._months_to_str(r.harvest_months)}"
            for r in profile.primary_regions
        ])

        base = cls.render(
            agent_name="The Agronomist",
            domain=f"{profile.name} agronomy and weather impacts",
            role_description="Analyze how weather events affect crop yield and quality",
            regime_context=regime_context
        )

        context = cls.AGRONOMIST_CONTEXT.format(
            regions_summary=regions_summary,
            agronomy_context=profile.agronomy_context
        )
        return f"{base}\n{context}"

    @classmethod
    def render_macro(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render macro analyst prompt with commodity context."""
        countries = set(r.country for r in profile.primary_regions)
        currency_map = {
            "Brazil": "USD/BRL",
            "Vietnam": "USD/VND",
            "Colombia": "USD/COP",
            "Ivory Coast": "USD/XOF",
            "Ghana": "USD/GHS",
            "Ethiopia": "USD/ETB"
        }
        pairs = [currency_map.get(c, "") for c in countries if c in currency_map]

        base = cls.render(
            agent_name="The Macro Strategist",
            domain=f"{profile.name} macroeconomics and currency",
            role_description="Analyze currency and policy impacts on pricing",
            regime_context=regime_context
        )

        context = cls.MACRO_CONTEXT.format(
            macro_context=profile.macro_context,
            currency_pairs=", ".join(pairs) if pairs else "USD (base)"
        )
        return f"{base}\n{context}"

    @classmethod
    def render_inventory(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render inventory analyst prompt with commodity context."""
        sources = "\n".join([f"- {s}" for s in profile.inventory_sources])

        base = cls.render(
            agent_name="The Inventory Analyst",
            domain=f"{profile.name} supply chain and stocks",
            role_description="Analyze certified stocks and forward curves",
            regime_context=regime_context
        )

        context = cls.INVENTORY_CONTEXT.format(
            commodity_name=profile.name,
            inventory_sources=sources,
            supply_chain_context=profile.supply_chain_context
        )
        return f"{base}\n{context}"

    @classmethod
    def render_sentiment(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render sentiment analyst prompt."""
        base = cls.render(
            agent_name="The Sentiment Trader",
            domain=f"{profile.name} market positioning",
            role_description="Identify crowded trades and sentiment extremes",
            regime_context=regime_context
        )

        return f"{base}\n{cls.SENTIMENT_CONTEXT}"

    @classmethod
    def render_technical(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render technical analyst prompt with contract specs."""
        base = cls.render(
            agent_name="The Technical Analyst",
            domain=f"{profile.name} price action",
            role_description="Analyze market structure, trends, and key levels",
            regime_context=regime_context
        )

        context = cls.TECHNICAL_CONTEXT.format(
            contract_symbol=profile.contract.symbol,
            contract_exchange=profile.contract.exchange,
            tick_size=profile.contract.tick_size,
            unit=profile.contract.unit,
            contract_months=", ".join(profile.contract.contract_months)
        )
        return f"{base}\n{context}"

    @classmethod
    def render_volatility(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render volatility analyst prompt with thresholds."""
        base = cls.render(
            agent_name="The Volatility Analyst",
            domain=f"{profile.name} options and volatility",
            role_description="Assess implied volatility and recommend strategies",
            regime_context=regime_context
        )

        context = cls.VOLATILITY_CONTEXT.format(
            contract_symbol=profile.contract.symbol,
            high_iv_rank=profile.volatility_high_iv_rank,
            low_iv_rank=profile.volatility_low_iv_rank
        )
        return f"{base}\n{context}"

    @classmethod
    def render_geopolitical(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render geopolitical analyst prompt."""
        regions_summary = "\n".join([
            f"- {r.name}, {r.country}"
            for r in profile.primary_regions
        ])
        logistics_summary = "\n".join([
            f"- {h.name} ({h.hub_type}), {h.country}"
            for h in profile.logistics_hubs
        ])

        base = cls.render(
            agent_name="The Geopolitical Analyst",
            domain=f"{profile.name} supply chain risk",
            role_description="Identify political and trade risks",
            regime_context=regime_context
        )

        context = cls.GEOPOLITICAL_CONTEXT.format(
            commodity_name=profile.name,
            regions_summary=regions_summary,
            logistics_summary=logistics_summary
        )
        return f"{base}\n{context}"

    @classmethod
    def render_logistics(cls, profile: CommodityProfile, regime_context: str = "") -> str:
        """Render logistics analyst prompt with thresholds."""
        logistics_summary = "\n".join([
            f"- {h.name} ({h.hub_type}), {h.country}"
            for h in profile.logistics_hubs
        ])
        logistics_thresholds = "\n".join([
            f"- {h.name}: Congestion alert at {h.congestion_vessel_threshold} vessels, "
            f"Dwell alert at {h.dwell_time_alert_days} days"
            for h in profile.logistics_hubs
        ])

        base = cls.render(
            agent_name="The Logistics Analyst",
            domain=f"{profile.name} shipping and transport",
            role_description="Monitor physical supply chain bottlenecks",
            regime_context=regime_context
        )

        context = cls.LOGISTICS_CONTEXT.format(
            logistics_summary=logistics_summary,
            logistics_thresholds=logistics_thresholds
        )
        return f"{base}\n{context}"

    @staticmethod
    def _months_to_str(months: list) -> str:
        """Convert month numbers to readable string."""
        month_names = [
            "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
        return ", ".join([month_names[m] for m in months if 1 <= m <= 12])


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_agent_prompt(agent_type: str, profile: CommodityProfile, regime_context: str = "") -> str:
    """
    Get the rendered prompt for a specific agent type.

    Args:
        agent_type: One of 'agronomist', 'macro', 'inventory', 'sentiment',
                   'technical', 'volatility', 'geopolitical', 'logistics'
        profile: CommodityProfile instance
        regime_context: Optional string describing market regime (Deficit/Surplus)

    Returns:
        Rendered prompt string

    Raises:
        ValueError: If agent_type is not recognized
    """
    renderers = {
        'agronomist': AgentPromptTemplate.render_agronomist,
        'macro': AgentPromptTemplate.render_macro,
        'inventory': AgentPromptTemplate.render_inventory,
        'sentiment': AgentPromptTemplate.render_sentiment,
        'technical': AgentPromptTemplate.render_technical,
        'volatility': AgentPromptTemplate.render_volatility,
        'geopolitical': AgentPromptTemplate.render_geopolitical,
        'logistics': AgentPromptTemplate.render_logistics,
        # Map alternate names
        'supply_chain': AgentPromptTemplate.render_logistics,
    }

    renderer = renderers.get(agent_type.lower())
    if not renderer:
        available = ", ".join(renderers.keys())
        raise ValueError(
            f"Unknown agent type '{agent_type}'. Available: {available}"
        )

    return renderer(profile, regime_context=regime_context)
