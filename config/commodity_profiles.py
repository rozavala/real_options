"""
Commodity Profile System - Decouples trading logic from commodity specifics.

This module defines the "Fuel" that powers the "Engine". The Engine (orchestrator,
agents, TMS) remains identical; only the Profile changes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import os
import logging
import json

logger = logging.getLogger(__name__)


class CommodityType(Enum):
    """Broad commodity categories for routing and context."""
    SOFT = "soft"           # Coffee, Cocoa, Sugar, Cotton
    ENERGY = "energy"       # Crude Oil, Natural Gas
    METAL = "metal"         # Gold, Silver, Copper
    GRAIN = "grain"         # Wheat, Corn, Soybeans


@dataclass
class GrowingRegion:
    """Geographic region with weather monitoring parameters."""
    name: str
    country: str
    latitude: float
    longitude: float
    weight: float  # Production weight (0.0-1.0)

    # Weather thresholds (commodity-specific)
    frost_threshold_celsius: Optional[float] = None
    drought_soil_moisture_pct: Optional[float] = None
    flood_precip_mm_24h: Optional[float] = None

    # Seasonality
    harvest_months: List[int] = field(default_factory=list)
    planting_months: List[int] = field(default_factory=list)


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
    trading_hours_utc: str         # e.g., "04:15-13:30"

    # Position limits
    spot_month_limit: int = 0
    all_months_limit: int = 0


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

    # Risk thresholds
    volatility_high_iv_rank: float = 0.7
    volatility_low_iv_rank: float = 0.3
    price_move_alert_pct: float = 2.0

    # Price validation (from config.json commodity_profile — used by order manager)
    stop_parse_range: List[float] = field(default_factory=lambda: [0.0, 9999.0])
    typical_price_range: List[float] = field(default_factory=lambda: [0.0, 9999.0])

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
# PREDEFINED PROFILES
# =============================================================================

COFFEE_ARABICA_PROFILE = CommodityProfile(
    name="Coffee Arabica",
    ticker="KC",
    commodity_type=CommodityType.SOFT,

    contract=ContractSpec(
        symbol="KC",
        exchange="ICE",
        contract_months=["H", "K", "N", "U", "Z"],  # Mar, May, Jul, Sep, Dec
        tick_size=0.05,
        tick_value=18.75,
        contract_size=37500,  # lbs
        unit="cents/lb",
        trading_hours_utc="04:15-13:30",
        spot_month_limit=500,
        all_months_limit=5000
    ),

    primary_regions=[
        GrowingRegion(
            name="Minas Gerais",
            country="Brazil",
            latitude=-19.9,
            longitude=-43.9,
            weight=0.35,
            frost_threshold_celsius=2.0,
            drought_soil_moisture_pct=10.0,
            harvest_months=[5, 6, 7, 8, 9],  # May-Sep
            planting_months=[10, 11, 12]
        ),
        GrowingRegion(
            name="São Paulo",
            country="Brazil",
            latitude=-23.5,
            longitude=-46.6,
            weight=0.15,
            frost_threshold_celsius=2.0,
            drought_soil_moisture_pct=10.0,
            harvest_months=[5, 6, 7, 8],
            planting_months=[10, 11]
        ),
        GrowingRegion(
            name="Dak Lak",
            country="Vietnam",
            latitude=12.7,
            longitude=108.0,
            weight=0.15,
            drought_soil_moisture_pct=15.0,
            harvest_months=[10, 11, 12, 1],  # Oct-Jan
            planting_months=[5, 6]
        ),
        GrowingRegion(
            name="Central Highlands",
            country="Vietnam",
            latitude=14.3,
            longitude=108.2,
            weight=0.05,
            drought_soil_moisture_pct=15.0,
            harvest_months=[10, 11, 12, 1],  # Oct-Jan
            planting_months=[5, 6]
        ),
        GrowingRegion(
            name="Colombia Huila",
            country="Colombia",
            latitude=2.5,
            longitude=-75.5,
            weight=0.10,
            flood_precip_mm_24h=100.0,
            harvest_months=[4, 5, 6, 10, 11, 12],  # Two harvests
            planting_months=[3, 9]
        ),
        GrowingRegion(
            name="Honduras Copan",
            country="Honduras",
            latitude=14.8,
            longitude=-89.1,
            weight=0.05,
            drought_soil_moisture_pct=12.0,
            harvest_months=[11, 12, 1, 2, 3],  # Nov-Mar
            planting_months=[5, 6]
        ),
        GrowingRegion(
            name="Sumatra",
            country="Indonesia",
            latitude=3.6,
            longitude=98.7,
            weight=0.05,
            flood_precip_mm_24h=120.0,
            harvest_months=[3, 4, 5, 6, 9, 10, 11, 12],  # Two harvests (fly crop + main)
            planting_months=[1, 2]
        ),
        # NOTE: Ethiopia (Sidamo/Yirgacheffe) is not currently monitored in production
        # but is a significant Arabica origin. Add to config.json sentinel locations
        # before enabling here. Weight set to 0.0 to prevent accidental activation.
        GrowingRegion(
            name="Sidamo/Yirgacheffe",
            country="Ethiopia",
            latitude=6.0,
            longitude=38.5,
            weight=0.0,  # NOT YET IN PRODUCTION — set to 0 until sentinel configured
            drought_soil_moisture_pct=12.0,
            harvest_months=[10, 11, 12],
            planting_months=[4, 5]
        ),
    ],

    logistics_hubs=[
        LogisticsHub(
            name="Port of Santos",
            hub_type="port",
            country="Brazil",
            latitude=-23.95,
            longitude=-46.30,
            congestion_vessel_threshold=20,
            dwell_time_alert_days=5
        ),
        LogisticsHub(
            name="Port of Ho Chi Minh",
            hub_type="port",
            country="Vietnam",
            latitude=10.8,
            longitude=106.7,
            congestion_vessel_threshold=15,
            dwell_time_alert_days=4
        ),
        LogisticsHub(
            name="Suez Canal",
            hub_type="transit",
            country="Egypt",
            latitude=30.0,
            longitude=32.5,
            congestion_vessel_threshold=50,
            dwell_time_alert_days=2
        ),
    ],

    agronomy_context="""
    Critical weather risks for Coffee Arabica:
    - FROST (Brazil, Jun-Aug): Temperatures below 2°C damage leaves and cherries.
      Severe frost (<-2°C) destroys trees, affecting NEXT YEAR's crop.
    - DROUGHT (Brazil, Oct-Nov): Soil moisture <10% during flowering reduces yield.
    - EXCESS RAIN (Colombia): >100mm/24h causes cherry rot and landslides.
    - Coffee Leaf Rust (Hemileia vastatrix): Fungal disease favored by humid conditions.

    Seasonality: Brazil harvest May-Sep (70% of crop). Vietnam Oct-Jan (Robusta focus).
    Flowering in Brazil: Sep-Oct, critical for next year's production.
    """,

    macro_context="""
    Key macro drivers for Coffee:
    - USD/BRL exchange rate: Weaker Real = cheaper Brazilian exports = bearish.
    - EUDR (EU Deforestation Regulation): Compliance costs, supply chain disruption.
    - Interest rates: Higher rates strengthen USD, bearish for coffee priced in USD.
    - Vietnam Dong: Secondary currency exposure for Robusta.
    """,

    supply_chain_context="""
    Key supply chain factors:
    - Port of Santos: Brazil's primary coffee export hub. Congestion = bullish.
    - Suez/Panama canals: Transit disruptions extend delivery times to EU.
    - ICE Certified Stocks: Visible inventory. Drawing = bullish, Building = bearish.
    - GCA (Green Coffee Association): US warehouse stocks (monthly, delayed).
    - Backwardation: Nearby > deferred = tight supply, bullish.
    """,

    inventory_sources=[
        "ICE Arabica Certified Stocks",
        "GCA Green Coffee Stocks",
        "USDA World Markets and Trade",
        "CONAB Brazil Crop Estimates"
    ],

    weather_apis=[
        "open-meteo",
        "meteomatics"
    ],

    news_keywords=[
        "coffee", "arabica", "robusta", "café",
        "frost brazil", "coffee rust", "santos port",
        "ICE coffee", "KC futures", "EUDR coffee"
    ],

    social_accounts=[
        # MUST MATCH config.json → sentinels.x_sentiment.from_handles
        # These are the production-curated X handles monitored by XSentimentSentinel
        "SpillingTheBean",    # Coffee-specific commodity analyst
        "optima_t",           # Coffee market intelligence
        "Reuters",            # Major wire service
        "ICOcoffeeorg",       # International Coffee Organization
        "zerohedge",          # Macro/markets commentary
        "Barchart",           # Commodity data & charts
        "WSJ"                 # Wall Street Journal markets
    ],

    volatility_high_iv_rank=0.70,
    volatility_low_iv_rank=0.30,
    price_move_alert_pct=2.0,

    # Price validation — MUST match config.json → commodity_profile.KC
    stop_parse_range=[80.0, 800.0],       # Valid stop-loss price range (cents/lb)
    typical_price_range=[100.0, 600.0]    # Sanity check range for predictions
)


COCOA_PROFILE = CommodityProfile(
    name="Cocoa",
    ticker="CC",
    commodity_type=CommodityType.SOFT,

    contract=ContractSpec(
        symbol="CC",
        exchange="ICE",
        contract_months=["H", "K", "N", "U", "Z"],
        tick_size=1.0,
        tick_value=10.0,
        contract_size=10,  # metric tons
        unit="$/metric ton",
        trading_hours_utc="04:45-13:30",
        spot_month_limit=1000,
        all_months_limit=10000
    ),

    primary_regions=[
        GrowingRegion(
            name="Côte d'Ivoire",
            country="Ivory Coast",
            latitude=6.8,
            longitude=-5.3,
            weight=0.40,
            drought_soil_moisture_pct=15.0,
            harvest_months=[10, 11, 12, 1, 2],  # Main crop Oct-Feb
            planting_months=[5, 6]
        ),
        GrowingRegion(
            name="Ghana",
            country="Ghana",
            latitude=6.7,
            longitude=-1.6,
            weight=0.20,
            drought_soil_moisture_pct=15.0,
            harvest_months=[10, 11, 12, 1],
            planting_months=[5, 6]
        ),
        GrowingRegion(
            name="Ecuador",
            country="Ecuador",
            latitude=-1.8,
            longitude=-79.5,
            weight=0.08,
            flood_precip_mm_24h=80.0,
            harvest_months=[3, 4, 5, 6],
            planting_months=[11, 12]
        ),
    ],

    logistics_hubs=[
        LogisticsHub(
            name="Port of Abidjan",
            hub_type="port",
            country="Ivory Coast",
            latitude=5.3,
            longitude=-4.0,
            congestion_vessel_threshold=15,
            dwell_time_alert_days=4
        ),
        LogisticsHub(
            name="Port of Tema",
            hub_type="port",
            country="Ghana",
            latitude=5.6,
            longitude=0.0,
            congestion_vessel_threshold=10,
            dwell_time_alert_days=5
        ),
    ],

    agronomy_context="""
    Critical weather risks for Cocoa:
    - HARMATTAN (Dec-Feb): Dry, dusty winds from Sahara stress trees.
    - DROUGHT: Soil moisture <15% reduces pod development.
    - BLACK POD DISEASE (Phytophthora): Fungal disease favored by excess humidity.
    - SWOLLEN SHOOT VIRUS: Spread by mealybugs, requires tree removal.

    Seasonality: Main crop Oct-Feb (65%), Mid-crop May-Aug (35%).
    """,

    macro_context="""
    Key macro drivers for Cocoa:
    - EUDR (EU Deforestation Regulation): Major compliance challenge.
    - Côte d'Ivoire/Ghana minimum price: Government price floors.
    - Chocolate demand (Europe/US): Consumer spending sensitivity.
    - GBP/USD: UK pricing exposure.
    """,

    supply_chain_context="""
    Key supply chain factors:
    - ICCO (International Cocoa Organization) stocks
    - ICE Certified Cocoa Stocks
    - Port of Abidjan: Primary West African export hub
    - Grinding statistics: Proxy for demand (Europe, Asia, Americas)
    """,

    inventory_sources=[
        "ICE Cocoa Certified Stocks",
        "ICCO Quarterly Bulletin",
        "European Cocoa Association Grindings"
    ],

    weather_apis=["open-meteo"],

    news_keywords=[
        "cocoa", "cacao", "chocolate",
        "ivory coast cocoa", "ghana cocoa",
        "black pod", "harmattan", "ICCO"
    ],

    social_accounts=[
        "@CocoaBarometer", "@ICCOorg"
    ],

    volatility_high_iv_rank=0.65,
    volatility_low_iv_rank=0.25,
    price_move_alert_pct=3.0
)


# =============================================================================
# ⚠️  CONFIG.JSON INTEGRATION — CRITICAL OPERATIONAL NOTE
# =============================================================================
#
# CommodityProfile is SUPPLEMENTARY to config.json, NOT a replacement.
#
# RUNTIME AUTHORITY (config.json — sentinels read from HERE):
#   - sentinels.x_sentiment.from_handles  → X accounts to monitor
#   - sentinels.x_sentiment.search_queries → search terms
#   - sentinels.x_sentiment.exclude_keywords → spam filters
#   - sentinels.weather.locations          → lat/lon for weather API calls
#   - sentinels.weather.triggers           → frost_temp_c=4.0 (ALERT threshold)
#   - sentinels.logistics.rss_urls         → RSS feed URLs
#   - sentinels.news.rss_urls              → news RSS feed URLs
#   - sentinels.microstructure.*           → order book thresholds
#   - sentinels.prediction_markets.*       → Polymarket topics
#   - fred_series, yf_series_map           → data pipeline tickers
#   - commodity_profile.KC.stop_parse_range → [80, 800] for stop-loss validation
#   - commodity_profile.KC.typical_price_range → [100, 600] for sanity checks
#
# THIS FILE PROVIDES (for agent prompts & profile metadata):
#   - agronomy_context, macro_context, supply_chain_context → injected into prompts
#   - GrowingRegion data → used in prompt templates & future sentinel routing
#   - ContractSpec → tick size, contract months, position limits
#   - social_accounts → REFERENCE copy (must stay synced with config.json)
#
# THRESHOLD NOTE: GrowingRegion.frost_threshold_celsius=2.0 is the AGRONOMIC
# damage threshold (crop damage occurs here). Config.json sentinel
# frost_temp_c=4.0 is the ALERT threshold (fires early as early warning).
# Both are correct — they serve different purposes.
#
# FUTURE (Phase 3+): Sentinels will read from CommodityProfile directly,
# making config.json sentinel sections auto-generated from profiles.
# Until then, changes to social_accounts/regions must be synced in BOTH places.
# =============================================================================


# =============================================================================
# PROFILE FACTORY
# =============================================================================

_PROFILES = {
    "KC": COFFEE_ARABICA_PROFILE,
    "CC": COCOA_PROFILE,
    # Add more profiles as needed:
    # "CL": CRUDE_OIL_PROFILE,
    # "GC": GOLD_PROFILE,
}


def get_commodity_profile(ticker: str) -> CommodityProfile:
    """
    Load a commodity profile by ticker symbol.

    FIX (MECE 8.1): Supports both hardcoded and JSON-file profiles.
    This enables adding custom commodities without modifying Python code.

    Lookup order:
    1. Hardcoded profiles (fast path, most common)
    2. JSON file at config/profiles/{ticker}.json (extensibility)

    Args:
        ticker: Exchange ticker symbol (e.g., "KC", "CC")

    Returns:
        CommodityProfile instance

    Raises:
        ValueError: If no profile exists for the ticker
    """
    ticker = ticker.upper()

    # Fast path: check hardcoded profiles first
    if ticker in _PROFILES:
        return _PROFILES[ticker]

    # Extensibility: check for custom JSON profile
    custom_path = f"config/profiles/{ticker.lower()}.json"
    if os.path.exists(custom_path):
        try:
            return _load_profile_from_json(custom_path)
        except Exception as e:
            logger.error(f"Failed to load custom profile {custom_path}: {e}")
            raise ValueError(f"Invalid custom profile for '{ticker}': {e}")

    available = ", ".join(_PROFILES.keys())
    raise ValueError(
        f"No commodity profile for '{ticker}'. "
        f"Available: {available}. "
        f"Or create custom profile at: {custom_path}"
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
    # FIX (Flight Director Review): Aligned with GrowingRegion dataclass exactly.
    # - Removed 'climate_risks' (not a field on GrowingRegion)
    # - Added required 'weight' field
    # - Added optional weather thresholds and planting_months
    # - harvest_months MUST be List[int] (1=Jan, 12=Dec), NOT strings
    regions = [
        GrowingRegion(
            name=r['name'],
            country=r['country'],
            latitude=r.get('latitude', 0.0),
            longitude=r.get('longitude', 0.0),
            weight=r.get('weight', 0.0),
            # Weather thresholds (optional, commodity-specific)
            frost_threshold_celsius=r.get('frost_threshold_celsius'),
            drought_soil_moisture_pct=r.get('drought_soil_moisture_pct'),
            flood_precip_mm_24h=r.get('flood_precip_mm_24h'),
            # Seasonality (integers: 1=Jan, 12=Dec)
            harvest_months=r.get('harvest_months', []),
            planting_months=r.get('planting_months', [])
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
        trading_hours_utc=data['contract'].get('trading_hours_utc', 'See exchange'),  # Was 'trading_hours' - FIXED
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
        weather_apis=data.get('weather_apis', []),
        volatility_high_iv_rank=data.get('volatility_high_iv_rank', 0.7),
        volatility_low_iv_rank=data.get('volatility_low_iv_rank', 0.3),
        price_move_alert_pct=data.get('price_move_alert_pct', 2.0),
    )


def list_available_profiles() -> List[str]:
    """
    Return list of available commodity tickers.

    Includes both hardcoded and custom JSON profiles.
    """
    import glob  # Only glob needs local import (not used elsewhere)

    available = set(_PROFILES.keys())

    # Add custom profiles
    custom_dir = "config/profiles"
    if os.path.exists(custom_dir):
        for json_file in glob.glob(f"{custom_dir}/*.json"):
            ticker = os.path.basename(json_file).replace('.json', '').upper()
            available.add(ticker)

    return sorted(available)
