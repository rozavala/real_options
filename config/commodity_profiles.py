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
# PREDEFINED PROFILES
# =============================================================================

COFFEE_ARABICA_PROFILE = CommodityProfile(
    name="Coffee (Arabica)",
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
        trading_hours_et="04:15-13:30",
        spot_month_limit=500,
        all_months_limit=5000
    ),

    primary_regions=[
        GrowingRegion(
            name="Minas Gerais",
            country="Brazil",
            latitude_range=(-22.0, -14.0),
            longitude_range=(-51.0, -39.0),
            production_share=0.30,
            historical_weekly_precip_mm=60.0,  # ~240mm/month during growing season
            drought_threshold_mm=30.0,  # <30mm/week = drought
            flood_threshold_mm=150.0,  # >150mm/week = flood
            flowering_months=[9, 10, 11],  # Sep-Nov (Southern Hemisphere spring)
            harvest_months=[5, 6, 7, 8],  # May-Aug (dry season)
            bean_filling_months=[12, 1, 2, 3],  # Dec-Mar (rainy season)
            planting_months=[10, 11, 12],
            frost_threshold_celsius=2.0,
            drought_soil_moisture_pct=10.0
        ),
        GrowingRegion(
            name="Espirito Santo",
            country="Brazil",
            latitude_range=(-21.0, -17.5),
            longitude_range=(-41.5, -39.5),
            production_share=0.15,
            historical_weekly_precip_mm=55.0,
            drought_threshold_mm=25.0,
            flood_threshold_mm=140.0,
            flowering_months=[9, 10, 11],
            harvest_months=[5, 6, 7, 8],
            bean_filling_months=[12, 1, 2, 3],
            planting_months=[10, 11],
            frost_threshold_celsius=2.0,
            drought_soil_moisture_pct=10.0
        ),
        GrowingRegion(
            name="Central Highlands",
            country="Vietnam",
            latitude_range=(11.0, 14.0),
            longitude_range=(107.0, 109.0),
            production_share=0.18,
            historical_weekly_precip_mm=70.0,  # Higher rainfall in Vietnam
            drought_threshold_mm=35.0,
            flood_threshold_mm=180.0,
            flowering_months=[1, 2, 3],  # Jan-Mar (tropical dry season)
            harvest_months=[10, 11, 12],  # Oct-Dec
            bean_filling_months=[4, 5, 6, 7, 8, 9],  # Apr-Sep (monsoon)
            planting_months=[5, 6],
            drought_soil_moisture_pct=15.0
        ),
        GrowingRegion(
            name="Copan",
            country="Honduras",
            latitude_range=(14.5, 15.5),
            longitude_range=(-89.0, -88.0),
            production_share=0.05,
            historical_weekly_precip_mm=50.0,
            drought_threshold_mm=20.0,
            flood_threshold_mm=130.0,
            flowering_months=[3, 4, 5],  # Mar-May
            harvest_months=[11, 12, 1, 2],  # Nov-Feb
            bean_filling_months=[6, 7, 8, 9, 10],
            planting_months=[5, 6],
            drought_soil_moisture_pct=12.0
        ),
        # Legacy/Extra regions mapped to new structure
        GrowingRegion(
            name="São Paulo",
            country="Brazil",
            latitude_range=(-24.0, -23.0),
            longitude_range=(-47.0, -46.0),
            production_share=0.15,
            harvest_months=[5, 6, 7, 8],
            planting_months=[10, 11],
            frost_threshold_celsius=2.0,
            drought_soil_moisture_pct=10.0
        ),
        GrowingRegion(
            name="Colombia Huila",
            country="Colombia",
            latitude_range=(2.0, 3.0),
            longitude_range=(-76.0, -75.0),
            production_share=0.10,
            harvest_months=[4, 5, 6, 10, 11, 12],
            planting_months=[3, 9],
            flood_precip_mm_24h=100.0
        ),
        GrowingRegion(
            name="Sumatra",
            country="Indonesia",
            latitude_range=(3.0, 4.0),
            longitude_range=(98.0, 99.0),
            production_share=0.05,
            harvest_months=[3, 4, 5, 6, 9, 10, 11, 12],
            planting_months=[1, 2],
            flood_precip_mm_24h=120.0
        ),
        GrowingRegion(
            name="Sidamo/Yirgacheffe",
            country="Ethiopia",
            latitude_range=(5.5, 6.5),
            longitude_range=(38.0, 39.0),
            production_share=0.0,
            harvest_months=[10, 11, 12],
            planting_months=[4, 5],
            drought_soil_moisture_pct=12.0
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
        "SpillingTheBean",    # Coffee-specific commodity analyst
        "optima_t",           # Coffee market intelligence
        "Reuters",            # Major wire service
        "ICOcoffeeorg",       # International Coffee Organization
        "zerohedge",          # Macro/markets commentary
        "Barchart",           # Commodity data & charts
        "WSJ"                 # Wall Street Journal markets
    ],

    sentiment_search_queries=[
        "coffee futures",
        "arabica prices",
        "KC futures",
        "robusta market",
        "coffee supply",
        "Brazil coffee harvest",
        "coffee supply chain",
        "US coffee tariffs",
    ],

    legitimate_data_sources=[
        'USDA', 'ICE', 'ICE Exchange', 'CONAB', 'CECAFE',
        'ICO', 'Green Coffee Association', 'NOAA',
        # v7.1: Sources discovered via agent grounded search (false positive fixes)
        'Banco Central do Brasil', 'DatamarNews', 'Seatrade Maritime',
        'StoneX', 'Saxo Bank', 'CocoaIntel', 'Drewry',
        'FX Leaders', 'MarketScreener', 'Somar',
    ],

    volatility_high_iv_rank=0.70,
    volatility_low_iv_rank=0.30,
    price_move_alert_pct=2.0,
    straddle_risk_threshold=10000.0,
    fallback_iv=0.35,
    risk_free_rate=0.04,
    default_starting_capital=50000.0,
    min_dte=45,
    max_dte=365,

    # Price validation — MUST match config.json → commodity_profile.KC
    stop_parse_range=[80.0, 800.0],       # Valid stop-loss price range (cents/lb)
    typical_price_range=[100.0, 600.0],   # Sanity check range for predictions

    research_prompts={
        "agronomist": "Search for 'current 10-day weather forecast Minas Gerais coffee zone' and 'NOAA Brazil precipitation anomaly'. Analyze if recent rains are beneficial for flowering or excessive.",
        "macro": "Search for 'USD BRL exchange rate forecast' and 'Brazil Central Bank Selic rate outlook'. Determine if the BRL is trending to encourage farmer selling.",
        "geopolitical": "Search for 'Red Sea shipping coffee delays', 'Brazil port of Santos wait times', and 'EUDR regulation delay latest news'. Determine if there are logistical bottlenecks.",
        "supply_chain": "Search for 'Cecafé Brazil coffee export report latest', 'Global container freight index rates', and 'Green coffee shipping manifest trends'. Analyze flow volume vs port capacity.",
        "inventory": (
            "Search for 'ICE coffee certified stocks level news 2026' and 'ICO monthly coffee market report global supply'. "
            "Look for recent specific numbers in bags (e.g., 'ICE stocks rose to X bags'). "
            "Search for 'coffee forward curve structure' to detect 'Backwardation' or 'Contango'."
        ),
        "sentiment": "Search for 'Coffee COT report non-commercial net length'. Determine if market is overbought.",
        "technical": "Search for 'Coffee futures technical analysis {contract}' and '{contract} support resistance levels'. Look for 'RSI divergence' or 'Moving Average crossover'. IMPORTANT: You MUST find and explicitly state the current value of the '200-day Simple Moving Average (SMA)'.",
        "volatility": "Search for 'Coffee Futures Implied Volatility Rank current' and '{contract} option volatility skew'. Determine if option premiums are cheap or expensive relative to historical volatility.",
    },
    yfinance_ticker="KC=F",
    concentration_proxies=['KC', 'SB', 'EWZ', 'BRL'],
    concentration_label="Brazil",
    cross_commodity_basket={
        'gold': 'GC=F',
        'silver': 'SI=F',
        'wheat': 'ZW=F',
        'soybeans': 'ZS=F',
    },
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
        trading_hours_et="04:45-13:30",
        spot_month_limit=1000,
        all_months_limit=10000
    ),

    primary_regions=[
        GrowingRegion(
            name="Côte d'Ivoire",
            country="Ivory Coast",
            latitude_range=(6.0, 7.6), # Approx center 6.8
            longitude_range=(-6.0, -4.6), # Approx center -5.3
            production_share=0.40,
            drought_soil_moisture_pct=15.0,
            harvest_months=[10, 11, 12, 1, 2],  # Main crop Oct-Feb
            planting_months=[5, 6]
        ),
        GrowingRegion(
            name="Ghana",
            country="Ghana",
            latitude_range=(6.0, 7.5), # Approx center 6.7
            longitude_range=(-2.5, -0.5), # Approx center -1.6
            production_share=0.20,
            drought_soil_moisture_pct=15.0,
            harvest_months=[10, 11, 12, 1],
            planting_months=[5, 6]
        ),
        GrowingRegion(
            name="Ecuador",
            country="Ecuador",
            latitude_range=(-2.5, -1.0), # Approx center -1.8
            longitude_range=(-80.0, -79.0), # Approx center -79.5
            production_share=0.08,
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

    sentiment_search_queries=[
        "cocoa futures",
        "cocoa prices",
        "CC futures",
        "chocolate demand",
        "ivory coast cocoa",
        "ghana cocoa",
        "cocoa supply chain",
        "ICCO cocoa",
    ],

    volatility_high_iv_rank=0.65,
    volatility_low_iv_rank=0.25,
    price_move_alert_pct=3.0,
    straddle_risk_threshold=8000.0,

    research_prompts={
        "agronomist": "Search for 'Côte d Ivoire cocoa harvest forecast' and 'Ghana cocoa rainfall anomaly'. Analyze if conditions favor or threaten the main crop.",
        "macro": "Search for 'EUR USD exchange rate forecast' and 'West Africa CFA franc outlook'. Determine currency impact on cocoa pricing.",
        "geopolitical": "Search for 'Côte d Ivoire cocoa regulation' and 'Ghana COCOBOD policy'. Analyze supply-side policy risks.",
        "supply_chain": "Search for 'Abidjan port cocoa shipments' and 'cocoa grinding data Europe'. Analyze processing demand vs origin supply.",
        "inventory": "Search for 'ICE Cocoa Certified Stocks' and 'European cocoa warehouse stocks'. Look for inventory trends.",
        "sentiment": "Search for 'Cocoa COT report non-commercial net length'. Determine if market is overbought.",
        "technical": "Search for 'Cocoa futures technical analysis {contract}' and '{contract} support resistance levels'. Look for 'RSI divergence' or 'Moving Average crossover'. IMPORTANT: You MUST find the current '200-day SMA'.",
        "volatility": "Search for 'Cocoa Futures Implied Volatility Rank current' and '{contract} option volatility skew'. Determine if premiums are cheap or expensive.",
    },
    yfinance_ticker="CC=F",
    concentration_proxies=['CC', 'SB', 'EWZ', 'NGN=X'],
    concentration_label="West Africa",
    cross_commodity_basket={
        'gold': 'GC=F',
        'sugar': 'SB=F',
        'wheat': 'ZW=F',
        'coffee': 'KC=F',
    },
)


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
            flowering_months=r.get('flowering_months', [9, 10, 11]),
            harvest_months=r.get('harvest_months', [5, 6, 7, 8]),
            bean_filling_months=r.get('bean_filling_months', [12, 1, 2, 3]),

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
