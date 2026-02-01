"""
Coffee Bot Configuration Package.

This package contains commodity profiles and system configuration.
"""

from .commodity_profiles import (
    get_commodity_profile,
    CommodityProfile,
    CommodityType,
    ContractSpec,
    GrowingRegion,
    LogisticsHub,
    COFFEE_ARABICA_PROFILE,
)

__all__ = [
    'get_commodity_profile',
    'CommodityProfile',
    'CommodityType',
    'ContractSpec',
    'GrowingRegion',
    'LogisticsHub',
    'COFFEE_ARABICA_PROFILE',
]
