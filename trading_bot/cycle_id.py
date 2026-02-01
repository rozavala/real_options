"""
Cycle ID generation for deterministic prediction-to-outcome matching.

DESIGN DECISIONS:
- Prefixed with commodity code for multi-commodity support (KC-xxx, CT-xxx)
- 8-char hex suffix for uniqueness (4 billion possibilities per commodity)
- Human-readable format for debugging
- NO timestamp in the ID (that's what the timestamp column is for)

USAGE:
    from trading_bot.cycle_id import generate_cycle_id
    cid = generate_cycle_id("KC")  # → "KC-a1b2c3d4"
"""

import uuid
import logging

logger = logging.getLogger(__name__)

# Default commodity when none specified (backward compat)
DEFAULT_COMMODITY = "KC"


def generate_cycle_id(commodity: str = DEFAULT_COMMODITY) -> str:
    """
    Generate a unique, commodity-namespaced cycle ID.

    Args:
        commodity: Commodity code (KC, CT, SB, etc.)

    Returns:
        String like "KC-a1b2c3d4"
    """
    suffix = uuid.uuid4().hex[:8]
    cycle_id = f"{commodity.upper()}-{suffix}"
    logger.debug(f"Generated cycle_id: {cycle_id}")
    return cycle_id


def parse_cycle_id(cycle_id: str) -> dict:
    """
    Parse a cycle_id into its components.

    Args:
        cycle_id: String like "KC-a1b2c3d4"

    Returns:
        {"commodity": "KC", "suffix": "a1b2c3d4", "valid": True}
    """
    if not cycle_id or not isinstance(cycle_id, str):
        return {"commodity": None, "suffix": None, "valid": False}

    parts = cycle_id.split("-", 1)
    if len(parts) != 2 or len(parts[1]) < 4:
        return {"commodity": None, "suffix": None, "valid": False}

    return {
        "commodity": parts[0],
        "suffix": parts[1],
        "valid": True
    }


def is_valid_cycle_id(cycle_id) -> bool:
    """Check if a value is a valid cycle_id (not None, NaN, empty)."""
    if cycle_id is None:
        return False
    if not isinstance(cycle_id, str):
        return False
    if cycle_id.strip() == '' or cycle_id == 'nan':
        return False
    return parse_cycle_id(cycle_id)["valid"]
