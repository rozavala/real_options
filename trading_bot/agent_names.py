"""
Central agent name normalization.

Different parts of the system use different names for the same agents.
This module provides canonical names and conversion utilities.
"""

# Canonical agent names (lowercase, underscore-separated)
CANONICAL_AGENTS = [
    'agronomist',      # Weather/Meteorologist
    'macro',           # Macro Economist
    'geopolitical',    # Geopolitical Risk
    'supply_chain',    # Supply Chain/Logistics
    'inventory',       # Inventory/Fundamentalist
    'sentiment',       # Sentiment/News/COT
    'technical',       # Technical Analyst
    'volatility',      # Volatility Analyst
    'master_decision', # Master Strategist
    # 'ml_model' removed in v4.0 — ML pipeline archived
]

# Map various names to canonical names
NAME_ALIASES = {
    # Agronomist / Weather
    'agronomist': 'agronomist',
    'meteorologist': 'agronomist',
    'meteorologist_sentiment': 'agronomist',
    'weather': 'agronomist',

    # Macro
    'macro': 'macro',
    'macro_economist': 'macro',
    'macro_sentiment': 'macro',

    # Geopolitical
    'geopolitical': 'geopolitical',
    'geopolitical_risk': 'geopolitical',
    'geopolitical_sentiment': 'geopolitical',

    # Supply Chain
    'supply_chain': 'supply_chain',
    'logistics': 'supply_chain',

    # Inventory / Fundamentalist
    'inventory': 'inventory',
    'fundamentalist': 'inventory',
    'fundamentalist_sentiment': 'inventory',
    'inventory_analyst': 'inventory',

    # Sentiment
    'sentiment': 'sentiment',
    'sentiment_cot': 'sentiment',
    'sentiment_sentiment': 'sentiment',
    'news': 'sentiment',

    # Technical
    'technical': 'technical',
    'technical_sentiment': 'technical',

    # Volatility
    'volatility': 'volatility',
    'volatility_sentiment': 'volatility',

    # Master
    'master': 'master_decision',
    'master_decision': 'master_decision',
    'master_strategist': 'master_decision',

    # ML (archived v4.0 — aliases kept for backward compat with historical data)
    'ml_model': 'ml_model',
    'ml': 'ml_model',
    'ml_signal': 'ml_model',
}

def normalize_agent_name(name: str) -> str:
    """
    Convert any agent name variant to canonical form.

    Args:
        name: Any agent name (case-insensitive)

    Returns:
        Canonical lowercase agent name, or original if not found
    """
    if not name:
        return 'unknown'

    # Lowercase and strip
    normalized = name.lower().strip()

    # Remove common suffixes
    for suffix in ['_sentiment', '_summary', '_signal']:
        if normalized.endswith(suffix):
            normalized = normalized.replace(suffix, '')
            break

    # Look up in aliases
    return NAME_ALIASES.get(normalized, normalized)


def get_display_name(canonical_name: str) -> str:
    """Get pretty display name for dashboards."""
    DISPLAY_NAMES = {
        'agronomist': '🌦️ Meteorologist',
        'macro': '💵 Macro Economist',
        'geopolitical': '🌍 Geopolitical',
        'supply_chain': '🚢 Supply Chain',
        'inventory': '📦 Fundamentalist',
        'sentiment': '🧠 Sentiment/COT',
        'technical': '📉 Technical',
        'volatility': '⚡ Volatility',
        'master_decision': '👑 Master Strategist',
        'ml_model': '🤖 ML Model',
    }
    return DISPLAY_NAMES.get(canonical_name, canonical_name.title())
