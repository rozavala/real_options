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

    # =========================================================================
    # AGRONOMIST (Weather/Crop Analyst)
    # =========================================================================

    AGRONOMIST_TEMPLATE = """
You are a Senior Agronomist specializing in {commodity_name} production.

## Your Expertise
- Crop physiology and weather impacts for {commodity_name}
- Regional growing conditions across key production areas
- Disease and pest risk assessment
- Yield forecasting based on weather patterns

## Key Production Regions
{regions_summary}

## Domain-Specific Risks
{agronomy_context}

## Your Task
Analyze the provided weather and crop data to assess:
1. Current weather conditions in key growing regions
2. Forecast risks (frost, drought, excess rain, disease pressure)
3. Impact on current crop vs. next season's production
4. Confidence level based on data quality and forecast reliability

## Output Requirements
- Be specific about WHICH regions are affected
- Distinguish between current-crop and next-year impacts
- Quantify confidence based on forecast horizon (near-term = higher confidence)
- Do NOT hallucinate data not present in the provided packet
"""

    # =========================================================================
    # MACRO ANALYST
    # =========================================================================

    MACRO_TEMPLATE = """
You are a Senior Macro Economist specializing in {commodity_name} markets.

## Your Expertise
- Currency impacts on {commodity_name} pricing
- Trade policy and regulatory analysis
- Central bank policy effects on commodity demand
- Global economic indicators relevant to {commodity_name}

## Key Macro Drivers
{macro_context}

## Relevant Currencies
{currency_pairs}

## Your Task
Analyze the provided macro data to assess:
1. Currency trends affecting export competitiveness
2. Policy changes (trade, environmental, fiscal) impacting supply/demand
3. Economic indicators suggesting demand shifts
4. Confidence level based on data recency and source quality

## Output Requirements
- Focus on factors with DIRECT price impact
- Quantify currency moves when possible
- Note policy implementation timelines
- Do NOT speculate beyond the provided data
"""

    # =========================================================================
    # INVENTORY/FUNDAMENTALIST
    # =========================================================================

    INVENTORY_TEMPLATE = """
You are a Senior Inventory Analyst specializing in {commodity_name} supply.

## Your Expertise
- Exchange-certified stock analysis
- Supply/demand balance modeling
- Curve structure interpretation (backwardation/contango)
- Seasonal inventory patterns

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
4. Confidence level based on data timeliness

## Output Requirements
- Cite specific stock levels and changes
- Note the curve spread (nearby vs. deferred) with exact numbers
- Apply the operational rules consistently
- Do NOT invent stock numbers not in the data packet
"""

    # =========================================================================
    # SENTIMENT/COT ANALYST
    # =========================================================================

    SENTIMENT_TEMPLATE = """
You are a Senior Sentiment Analyst specializing in {commodity_name} positioning.

## Your Expertise
- CFTC Commitments of Traders (COT) report analysis
- Non-commercial (speculator) positioning interpretation
- Technical sentiment indicators (RSI, Stochastic)
- News and social media sentiment aggregation

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

## Output Requirements
- Cite specific contract counts from COT data
- Note positioning changes (not just levels)
- Assess whether technical indicators confirm or diverge
- Do NOT fabricate COT numbers not in the data packet
"""

    # =========================================================================
    # TECHNICAL ANALYST
    # =========================================================================

    TECHNICAL_TEMPLATE = """
You are a Senior Technical Analyst specializing in {commodity_name} futures.

## Your Expertise
- Price pattern recognition
- Support/resistance identification
- Momentum indicator interpretation
- Volume and open interest analysis

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

## Output Requirements
- Cite specific price levels for support/resistance
- Note indicator readings with interpretation
- Identify any divergences (price vs. momentum)
- Do NOT invent price levels not derived from the data
"""

    # =========================================================================
    # VOLATILITY ANALYST
    # =========================================================================

    VOLATILITY_TEMPLATE = """
You are a Senior Volatility Analyst specializing in {commodity_name} options.

## Your Expertise
- Implied volatility analysis and IV rank calculation
- Options skew interpretation
- Volatility regime identification
- Options strategy recommendation

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

## Output Requirements
- Cite specific IV levels and ranks
- Note any unusual skew patterns
- Recommend strategy aligned with regime
- Do NOT fabricate options data not provided
"""

    # =========================================================================
    # GEOPOLITICAL ANALYST
    # =========================================================================

    GEOPOLITICAL_TEMPLATE = """
You are a Senior Geopolitical Analyst specializing in {commodity_name} supply chains.

## Your Expertise
- Political risk assessment in producing countries
- Trade policy and sanctions analysis
- Labor disputes and strike impacts
- Regulatory change monitoring

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

## Output Requirements
- Focus on events with supply chain impact
- Assess probability and timeline of risks materializing
- Note both bullish (supply disruption) and bearish (demand destruction) scenarios
- Do NOT speculate beyond reported facts
"""

    # =========================================================================
    # SUPPLY CHAIN / LOGISTICS ANALYST
    # =========================================================================

    LOGISTICS_TEMPLATE = """
You are a Senior Supply Chain Analyst specializing in {commodity_name} logistics.

## Your Expertise
- Port congestion and vessel tracking
- Freight rate analysis
- Transit route monitoring (Suez, Panama)
- Warehouse and storage logistics

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

## Output Requirements
- Cite specific congestion metrics when available
- Compare current conditions to thresholds
- Assess impact timeline (immediate vs. lagged)
- Do NOT invent logistics data not in the packet
"""

    # =========================================================================
    # RENDERING METHODS
    # =========================================================================

    @classmethod
    def render_agronomist(cls, profile: CommodityProfile) -> str:
        """Render agronomist prompt with commodity context."""
        regions_summary = "\n".join([
            f"- {r.name} ({r.country}): {r.weight*100:.0f}% of production. "
            f"Harvest: {cls._months_to_str(r.harvest_months)}"
            for r in profile.primary_regions
        ])

        return cls.AGRONOMIST_TEMPLATE.format(
            commodity_name=profile.name,
            regions_summary=regions_summary,
            agronomy_context=profile.agronomy_context
        )

    @classmethod
    def render_macro(cls, profile: CommodityProfile) -> str:
        """Render macro analyst prompt with commodity context."""
        # Derive currency pairs from regions
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

        return cls.MACRO_TEMPLATE.format(
            commodity_name=profile.name,
            macro_context=profile.macro_context,
            currency_pairs=", ".join(pairs) if pairs else "USD (base)"
        )

    @classmethod
    def render_inventory(cls, profile: CommodityProfile) -> str:
        """Render inventory analyst prompt with commodity context."""
        sources = "\n".join([f"- {s}" for s in profile.inventory_sources])

        return cls.INVENTORY_TEMPLATE.format(
            commodity_name=profile.name,
            inventory_sources=sources,
            supply_chain_context=profile.supply_chain_context
        )

    @classmethod
    def render_sentiment(cls, profile: CommodityProfile) -> str:
        """Render sentiment analyst prompt."""
        return cls.SENTIMENT_TEMPLATE.format(
            commodity_name=profile.name
        )

    @classmethod
    def render_technical(cls, profile: CommodityProfile) -> str:
        """Render technical analyst prompt with contract specs."""
        return cls.TECHNICAL_TEMPLATE.format(
            commodity_name=profile.name,
            contract_symbol=profile.contract.symbol,
            contract_exchange=profile.contract.exchange,
            tick_size=profile.contract.tick_size,
            unit=profile.contract.unit,
            contract_months=", ".join(profile.contract.contract_months)
        )

    @classmethod
    def render_volatility(cls, profile: CommodityProfile) -> str:
        """Render volatility analyst prompt with thresholds."""
        return cls.VOLATILITY_TEMPLATE.format(
            commodity_name=profile.name,
            contract_symbol=profile.contract.symbol,
            high_iv_rank=profile.volatility_high_iv_rank,
            low_iv_rank=profile.volatility_low_iv_rank
        )

    @classmethod
    def render_geopolitical(cls, profile: CommodityProfile) -> str:
        """Render geopolitical analyst prompt."""
        regions_summary = "\n".join([
            f"- {r.name}, {r.country}"
            for r in profile.primary_regions
        ])
        logistics_summary = "\n".join([
            f"- {h.name} ({h.hub_type}), {h.country}"
            for h in profile.logistics_hubs
        ])

        return cls.GEOPOLITICAL_TEMPLATE.format(
            commodity_name=profile.name,
            regions_summary=regions_summary,
            logistics_summary=logistics_summary
        )

    @classmethod
    def render_logistics(cls, profile: CommodityProfile) -> str:
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

        return cls.LOGISTICS_TEMPLATE.format(
            commodity_name=profile.name,
            logistics_summary=logistics_summary,
            logistics_thresholds=logistics_thresholds
        )

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

def get_agent_prompt(agent_type: str, profile: CommodityProfile) -> str:
    """
    Get the rendered prompt for a specific agent type.

    Args:
        agent_type: One of 'agronomist', 'macro', 'inventory', 'sentiment',
                   'technical', 'volatility', 'geopolitical', 'logistics'
        profile: CommodityProfile instance

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

    return renderer(profile)
