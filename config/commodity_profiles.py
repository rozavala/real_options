"""
Commodity Profile System - Decouples trading logic from commodity specifics.

This module defines the "Fuel" that powers the "Engine". The Engine (orchestrator,
agents, TMS) remains identical; only the Profile changes.
"""

from dataclasses import dataclass, field
from datetime import time
from typing import List, Dict, Optional, Tuple
from enum import Enum
import os
import logging
import json

logger = logging.getLogger(__name__)


def parse_trading_hours(hours_str: str) -> tuple:
    """Parse 'HH:MM-HH:MM' to (open_time, close_time) as ET.

    Returns:
        Tuple of (datetime.time, datetime.time) representing market open and close.
    """
    open_str, close_str = hours_str.split('-')
    oh, om = map(int, open_str.split(':'))
    ch, cm = map(int, close_str.split(':'))
    return time(oh, om), time(ch, cm)


class CommodityType(Enum):
    """Broad commodity categories for routing and context."""
    SOFT = "soft"           # Coffee, Cocoa, Sugar, Cotton
    ENERGY = "energy"       # Crude Oil, Natural Gas
    METAL = "metal"         # Gold, Silver, Copper
    GRAIN = "grain"         # Wheat, Corn, Soybeans


@dataclass
class GrowingRegion:
    """Physical characteristics of a commodity growing region."""
    name: str
    country: str
    latitude_range: Tuple[float, float]
    longitude_range: Tuple[float, float]
    production_share: float  # % of global production

    # NEW FIELDS (Flight Director Amendment - Data Locality)
    historical_weekly_precip_mm: float = 60.0  # Normal rainfall per week during growing season
    drought_threshold_mm: float = 30.0  # Below this = drought risk
    flood_threshold_mm: float = 150.0  # Above this = flood risk

    # Agronomic calendar (NEW)
    flowering_months: List[int] = field(default_factory=list)  # Months when flowering occurs (1-12)
    harvest_months: List[int] = field(default_factory=list)  # Months when harvest occurs (1-12)
    bean_filling_months: List[int] = field(default_factory=list)  # Critical bean development period

    # Legacy/Optional fields for compatibility
    planting_months: List[int] = field(default_factory=list)
    frost_threshold_celsius: Optional[float] = None
    drought_soil_moisture_pct: Optional[float] = None
    flood_precip_mm_24h: Optional[float] = None

    @property
    def latitude(self) -> float:
        """Center latitude for API queries."""
        return (self.latitude_range[0] + self.latitude_range[1]) / 2.0

    @property
    def longitude(self) -> float:
        """Center longitude for API queries."""
        return (self.longitude_range[0] + self.longitude_range[1]) / 2.0

    @property
    def weight(self) -> float:
        """Alias for production_share."""
        return self.production_share


@dataclass
class LogisticsHub:
    """Key logistics points for supply chain monitoring."""
    name: str
    hub_type: str  # "port", "rail", "warehouse", "refinery"
    country: str
    latitude: float
    longitude: float

    # Monitoring thresholds
    congestion_vessel_threshold: int = 20
    dwell_time_alert_days: int = 5


@dataclass
class ContractSpec:
    """Exchange contract specifications."""
    symbol: str                    # e.g., "KC", "CC", "CL"
    exchange: str                  # e.g., "ICE", "NYMEX"
    contract_months: List[str]     # e.g., ["H", "K", "N", "U", "Z"]
    tick_size: float               # Minimum price movement
    tick_value: float              # Dollar value per tick
    contract_size: float           # Units per contract
    unit: str                      # e.g., "cents/lb", "$/barrel"
    trading_hours_et: str          # e.g., "04:15-13:30" (Eastern Time)

    # Position limits
    spot_month_limit: int = 0
    all_months_limit: int = 0


@dataclass
class MarketStatesConfig:
    """Three-tier market state configuration per commodity.

    Defines Active/Passive/Sleeping windows and emergency trigger parameters.
    Source: CME Globex — NG trades Sun–Fri 18:00–17:00 ET, daily maint. halt 17:00–18:00 ET.
    """
    active: str                           # e.g., "09:00-14:30"
    passive: List[str]                    # e.g., ["18:00-09:00", "14:30-17:00"]
    maintenance_breaks: List[str]         # e.g., ["17:00-18:00"]
    emergency_trigger_pct: float          # e.g., 7.0 (0 = disabled)
    emergency_cooldown_seconds: int       # e.g., 1800
    emergency_council_timeout_seconds: int  # e.g., 300
    passive_new_positions_only: bool      # True = no adds to existing theses
    emergency_dry_run: bool = True        # True = log only, False = live execution
    blackout_windows: List[Dict] = field(default_factory=list)


@dataclass
class CommodityProfile:
    """
    Complete commodity configuration for the trading system.

    This is the "Fuel" - the Engine loads this profile and adapts all
    agent prompts, sentinel thresholds, and analysis logic accordingly.
    """
    # Identity
    name: str                           # e.g., "Coffee Arabica"
    ticker: str                         # e.g., "KC"
    commodity_type: CommodityType

    # Contract specifications
    contract: ContractSpec

    # Geography
    primary_regions: List[GrowingRegion] = field(default_factory=list)
    logistics_hubs: List[LogisticsHub] = field(default_factory=list)

    # Domain-specific context for LLM prompts
    agronomy_context: str = ""          # Weather/crop-specific risks
    macro_context: str = ""             # Currency, trade policy drivers
    supply_chain_context: str = ""      # Logistics, seasonal patterns

    # Key data sources
    inventory_sources: List[str] = field(default_factory=list)
    weather_apis: List[str] = field(default_factory=list)
    news_keywords: List[str] = field(default_factory=list)
    social_accounts: List[str] = field(default_factory=list)
    sentiment_search_queries: List[str] = field(default_factory=list)  # Added for XSentimentSentinel
    legitimate_data_sources: List[str] = field(default_factory=list)   # Added for Observability (Issue 3)

    # Risk thresholds
    volatility_high_iv_rank: float = 0.7
    volatility_low_iv_rank: float = 0.3
    price_move_alert_pct: float = 2.0
    straddle_risk_threshold: float = 10000.0  # Max straddle risk per contract (v3.1)

    # Liquidity filter: hybrid tick/percentage model
    # Spread must be ≤ max(tick_allowance, pct_allowance)
    # Tick floor protects cheap OTM options from ratio inflation
    # Percentage ceiling caps slippage on expensive ITM options
    max_liquidity_spread_pct: float = 0.75       # 75% of abs(theoretical price)
    max_liquidity_spread_ticks: int = 60          # Absolute floor in tick units

    # M1, M3, M7, E4 FIXES:
    fallback_iv: float = 0.35  # Default 35% (commodity-specific)
    risk_free_rate: float = 0.04  # M3 fix
    default_starting_capital: float = 50000.0  # E4 fix
    min_dte: int = 45  # M7 fix: minimum days to expiry
    max_dte: int = 365  # M7 fix: maximum days to expiry

    # Price validation (from config.json commodity_profile — used by order manager)
    stop_parse_range: List[float] = field(default_factory=lambda: [0.0, 9999.0])
    typical_price_range: List[float] = field(default_factory=lambda: [0.0, 9999.0])

    # Research prompts for signal_generator (per-agent search guidance)
    research_prompts: Dict[str, str] = field(default_factory=dict)

    # Compliance: geographic concentration check
    concentration_proxies: List[str] = field(default_factory=list)  # e.g., ['KC', 'SB', 'EWZ', 'BRL']
    concentration_label: str = ""  # e.g., "Brazil", "West Africa"

    # yfinance ticker for VaR historical data (e.g., "KC=F", "CC=F")
    yfinance_ticker: str = ""

    # Cross-commodity correlation basket for MacroContagionSentinel
    cross_commodity_basket: Dict[str, str] = field(default_factory=dict)  # e.g., {'gold': 'GC=F', ...}

    # Scoring: minimum price move to resolve as directional (below = NEUTRAL)
    # Calibrated from annualized IV: threshold ≈ 0.3 × (IV / sqrt(252))
    neutral_move_threshold_pct: float = 0.008  # Default 0.8%

    # Three-tier market state configuration (Active/Passive/Sleeping)
    market_states: Optional[MarketStatesConfig] = None

    # TMS Temporal Decay Rates (lambda values for exponential decay)
    # Higher lambda = faster decay = shorter useful life
    # relevance = base_score × exp(-lambda × age_days)
    # Half-life formula: t½ = ln(2) / lambda ≈ 0.693 / lambda
    tms_decay_rates: Dict[str, float] = field(default_factory=lambda: {
        'weather': 0.15,        # Half-life ≈ 4.6 days — weather is very perishable
        'logistics': 0.10,      # Half-life ≈ 6.9 days — port/shipping disruptions
        'news': 0.08,           # Half-life ≈ 8.7 days — news cycle
        'sentiment': 0.08,      # Half-life ≈ 8.7 days — market sentiment shifts
        'price': 0.20,          # Half-life ≈ 3.5 days — price data is very time-sensitive
        'microstructure': 0.25, # Half-life ≈ 2.8 days — order book data is extremely perishable
        'technical': 0.05,      # Half-life ≈ 13.9 days — technical patterns persist longer
        'macro': 0.02,          # Half-life ≈ 34.7 days — macro trends are slow-moving
        'geopolitical': 0.03,   # Half-life ≈ 23.1 days — geopolitical shifts persist
        'inventory': 0.04,      # Half-life ≈ 17.3 days — warehouse reports update weekly
        'supply_chain': 0.05,   # Half-life ≈ 13.9 days
        'research': 0.005,      # Half-life ≈ 138.6 days — academic findings persist very long
        'trade_journal': 0.01,  # Half-life ≈ 69.3 days — trade lessons persist
        'default': 0.05         # Default decay if type not specified
    })

    def get_region_coords(self) -> List[Dict]:
        """Return lat/lon for weather API queries."""
        return [
            {
                "name": r.name,
                "lat": r.latitude,
                "lon": r.longitude,
                "weight": r.weight
            }
            for r in self.primary_regions
        ]

    def get_harvest_calendar(self) -> Dict[str, List[int]]:
        """Return harvest months by region."""
        return {
            r.name: r.harvest_months
            for r in self.primary_regions
        }


# =============================================================================
# PROFILE FACTORY — All profiles loaded from JSON (config/profiles/*.json)
# =============================================================================



def get_commodity_profile(ticker: str) -> CommodityProfile:
    """
    Load a commodity profile by ticker symbol from JSON.

    All profiles are loaded from config/profiles/{ticker}.json.

    Args:
        ticker: Exchange ticker symbol (e.g., "KC", "CC", "NG")

    Returns:
        CommodityProfile instance

    Raises:
        ValueError: If no profile exists for the ticker
    """
    ticker = ticker.upper()

    profile_path = f"config/profiles/{ticker.lower()}.json"
    if os.path.exists(profile_path):
        try:
            return _load_profile_from_json(profile_path)
        except Exception as e:
            logger.error(f"Failed to load profile {profile_path}: {e}")
            raise ValueError(f"Invalid profile for '{ticker}': {e}")

    raise ValueError(
        f"No commodity profile for '{ticker}'. "
        f"Create a profile at: {profile_path}"
    )


def _parse_market_states(raw: Optional[dict]) -> Optional[MarketStatesConfig]:
    """Parse market_states dict from JSON into MarketStatesConfig, or None."""
    if not raw:
        return None
    return MarketStatesConfig(
        active=raw['active'],
        passive=raw.get('passive', []),
        maintenance_breaks=raw.get('maintenance_breaks', []),
        emergency_trigger_pct=raw.get('emergency_trigger_pct', 0),
        emergency_cooldown_seconds=raw.get('emergency_cooldown_seconds', 1800),
        emergency_council_timeout_seconds=raw.get('emergency_council_timeout_seconds', 300),
        passive_new_positions_only=raw.get('passive_new_positions_only', True),
        emergency_dry_run=raw.get('emergency_dry_run', True),
        blackout_windows=raw.get('blackout_windows', []),
    )


def _load_profile_from_json(path: str) -> CommodityProfile:
    """
    Load a CommodityProfile from a JSON file.

    This enables users to add custom commodities without modifying Python code.
    See config/profiles/template.json for the expected format.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Build nested objects from JSON
    # FIX (Flight Director Amendment): Added new fields with defaults
    regions = [
        GrowingRegion(
            name=r['name'],
            country=r['country'],
            # Convert JSON list/tuple to tuple for range
            latitude_range=tuple(r.get('latitude_range', (0.0, 0.0))),
            longitude_range=tuple(r.get('longitude_range', (0.0, 0.0))),
            production_share=r.get('production_share', r.get('weight', 0.0)),

            # New Fields
            historical_weekly_precip_mm=r.get('historical_weekly_precip_mm', 60.0),
            drought_threshold_mm=r.get('drought_threshold_mm', 30.0),
            flood_threshold_mm=r.get('flood_threshold_mm', 150.0),
            flowering_months=r.get('flowering_months', []),
            harvest_months=r.get('harvest_months', []),
            bean_filling_months=r.get('bean_filling_months', []),

            # Legacy/Optional
            planting_months=r.get('planting_months', []),
            frost_threshold_celsius=r.get('frost_threshold_celsius'),
            drought_soil_moisture_pct=r.get('drought_soil_moisture_pct'),
            flood_precip_mm_24h=r.get('flood_precip_mm_24h'),
        )
        for r in data.get('primary_regions', [])
    ]

    hubs = [
        LogisticsHub(
            name=h['name'],
            hub_type=h.get('hub_type', 'port'),
            country=h['country'],
            latitude=h.get('latitude', 0.0),
            longitude=h.get('longitude', 0.0),
            # V3 FIX: Monitoring thresholds (defaults match dataclass)
            congestion_vessel_threshold=h.get('congestion_vessel_threshold', 20),
            dwell_time_alert_days=h.get('dwell_time_alert_days', 5),
        )
        for h in data.get('logistics_hubs', [])
    ]

    contract = ContractSpec(
        # FIX (Final Review): Aligned fields with ContractSpec dataclass definition
        exchange=data['contract']['exchange'],
        symbol=data['contract']['symbol'],
        contract_months=data['contract']['contract_months'],
        tick_size=data['contract']['tick_size'],
        tick_value=data['contract']['tick_value'],  # Was 'point_value' - FIXED
        contract_size=data['contract']['contract_size'],  # Was missing - ADDED
        unit=data['contract']['unit'],  # Was missing - ADDED
        trading_hours_et=data['contract'].get('trading_hours_et',
                         data['contract'].get('trading_hours_utc', 'See exchange')),
        # V3 FIX: Position limits (defaults match dataclass)
        spot_month_limit=data['contract'].get('spot_month_limit', 0),
        all_months_limit=data['contract'].get('all_months_limit', 0),
    )

    return CommodityProfile(
        name=data['name'],
        ticker=data['ticker'],
        commodity_type=CommodityType(data.get('commodity_type', 'soft')),
        contract=contract,
        primary_regions=regions,
        logistics_hubs=hubs,
        inventory_sources=data.get('inventory_sources', []),
        news_keywords=data.get('news_keywords', []),
        agronomy_context=data.get('agronomy_context', ''),
        macro_context=data.get('macro_context', ''),
        supply_chain_context=data.get('supply_chain_context', ''),
        # V3 FIX: Load remaining configurable fields (defaults match dataclass)
        social_accounts=data.get('social_accounts', []),
        sentiment_search_queries=data.get('sentiment_search_queries', []),
        legitimate_data_sources=data.get('legitimate_data_sources', []),
        weather_apis=data.get('weather_apis', []),
        volatility_high_iv_rank=data.get('volatility_high_iv_rank', 0.7),
        volatility_low_iv_rank=data.get('volatility_low_iv_rank', 0.3),
        price_move_alert_pct=data.get('price_move_alert_pct', 2.0),
        # V3 FIX: Load TMS decay rates if present
        tms_decay_rates=data.get('tms_decay_rates', {
            'weather': 0.15, 'logistics': 0.10, 'news': 0.08, 'sentiment': 0.08,
            'price': 0.20, 'microstructure': 0.25, 'technical': 0.05,
            'macro': 0.02, 'geopolitical': 0.03, 'inventory': 0.04,
            'supply_chain': 0.05, 'research': 0.005, 'trade_journal': 0.01,
            'default': 0.05
        }),
        # Risk/pricing fields (must match dataclass — JSON profiles need these)
        straddle_risk_threshold=data.get('straddle_risk_threshold', 10000.0),
        max_liquidity_spread_pct=data.get('max_liquidity_spread_pct', 0.75),
        max_liquidity_spread_ticks=data.get('max_liquidity_spread_ticks', 60),
        fallback_iv=data.get('fallback_iv', 0.35),
        risk_free_rate=data.get('risk_free_rate', 0.04),
        default_starting_capital=data.get('default_starting_capital', 50000.0),
        min_dte=data.get('min_dte', 45),
        max_dte=data.get('max_dte', 365),
        stop_parse_range=data.get('stop_parse_range', [0.0, 9999.0]),
        typical_price_range=data.get('typical_price_range', [0.0, 9999.0]),
        # Commodity-agnostic fields
        yfinance_ticker=data.get('yfinance_ticker', ''),
        research_prompts=data.get('research_prompts', {}),
        concentration_proxies=data.get('concentration_proxies', []),
        concentration_label=data.get('concentration_label', ''),
        cross_commodity_basket=data.get('cross_commodity_basket', {}),
        neutral_move_threshold_pct=data.get('neutral_move_threshold_pct', 0.008),
        market_states=_parse_market_states(data.get('market_states')),
    )


def list_available_profiles() -> List[str]:
    """Return list of available commodity tickers from JSON profiles."""
    import glob

    available = set()
    custom_dir = "config/profiles"
    if os.path.exists(custom_dir):
        for json_file in glob.glob(f"{custom_dir}/*.json"):
            basename = os.path.basename(json_file).replace('.json', '').upper()
            if basename != 'TEMPLATE':
                available.add(basename)

    return sorted(available)


def get_active_profile(config: dict) -> CommodityProfile:
    """
    Get the active commodity profile based on application config.

    Args:
        config: Application config dictionary

    Returns:
        CommodityProfile for the configured symbol
    """
    symbol = config.get('symbol', 'KC')
    # Handle commodity dict if present (v2 format)
    if isinstance(symbol, dict):
        symbol = symbol.get('ticker', 'KC')

    return get_commodity_profile(symbol)
