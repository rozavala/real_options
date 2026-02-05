"""
Shared confidence parsing utilities.

v5.3.1: Canonical source of truth for confidence band mappings.
All modules that need to convert LLM confidence output (band strings
or numeric values) to floats should import from here.

Commodity-agnostic: No commodity-specific logic.
"""

import logging

logger = logging.getLogger(__name__)

# === CONFIDENCE BAND MAPPING ===
# v7.0 philosophy: LLMs reason in categories, Python translates to numbers.
# These values represent the midpoint of each band's semantic range.
# If adding new bands, update ONLY this dict — all consumers import it.
CONFIDENCE_BANDS = {
    'LOW': 0.55,
    'MODERATE': 0.65,
    'HIGH': 0.80,
    'EXTREME': 0.90,
}


def parse_confidence(raw_value, default: float = 0.5) -> float:
    """
    Parse confidence from LLM output — handles bands, numbers, and edge cases.

    Supports:
        - Band strings: 'HIGH', 'moderate', ' Low ' (case/whitespace insensitive)
        - Numeric strings: '0.73', '0.5'
        - Actual numbers: 0.73, 1
        - Graceful fallback to `default` for anything unparseable

    Args:
        raw_value: The raw confidence value from LLM output or report dict.
        default: Fallback value if parsing fails. Default 0.5 (neutral).

    Returns:
        float: Parsed confidence value, clamped to [0.0, 1.0].

    Examples:
        >>> parse_confidence('HIGH')
        0.80
        >>> parse_confidence(0.73)
        0.73
        >>> parse_confidence('GARBAGE')
        0.5
    """
    if isinstance(raw_value, (int, float)):
        return max(0.0, min(1.0, float(raw_value)))
    if isinstance(raw_value, str):
        upper = raw_value.strip().upper()
        if upper in CONFIDENCE_BANDS:
            return CONFIDENCE_BANDS[upper]
        try:
            return max(0.0, min(1.0, float(raw_value)))
        except ValueError:
            logger.warning(
                f"Unparseable confidence value: '{raw_value}'. Using default {default}."
            )
            return default
    # None, list, dict, or other unexpected types
    if raw_value is not None:
        logger.warning(
            f"Unexpected confidence type: {type(raw_value).__name__} = {raw_value!r}. "
            f"Using default {default}."
        )
    return default
