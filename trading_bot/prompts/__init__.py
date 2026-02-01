"""
Agent Prompts Package.

This package contains templatized prompts for all agents,
supporting commodity-agnostic operations.

FIX (MECE V2 #2): Templates are class attributes on AgentPromptTemplate,
not module-level constants. We re-export them for convenience.
"""

from .base_prompts import (
    get_agent_prompt,
    AgentPromptTemplate,  # Export the class
)

# Re-export class attributes as module-level constants for convenience
# This allows: from trading_bot.prompts import AGRONOMIST_TEMPLATE
AGRONOMIST_TEMPLATE = AgentPromptTemplate.AGRONOMIST_TEMPLATE
MACRO_TEMPLATE = AgentPromptTemplate.MACRO_TEMPLATE
INVENTORY_TEMPLATE = AgentPromptTemplate.INVENTORY_TEMPLATE
SENTIMENT_TEMPLATE = AgentPromptTemplate.SENTIMENT_TEMPLATE
TECHNICAL_TEMPLATE = AgentPromptTemplate.TECHNICAL_TEMPLATE
VOLATILITY_TEMPLATE = AgentPromptTemplate.VOLATILITY_TEMPLATE
GEOPOLITICAL_TEMPLATE = AgentPromptTemplate.GEOPOLITICAL_TEMPLATE
LOGISTICS_TEMPLATE = AgentPromptTemplate.LOGISTICS_TEMPLATE

__all__ = [
    'get_agent_prompt',
    'AgentPromptTemplate',
    'AGRONOMIST_TEMPLATE',
    'MACRO_TEMPLATE',
    'INVENTORY_TEMPLATE',
    'SENTIMENT_TEMPLATE',
    'TECHNICAL_TEMPLATE',
    'VOLATILITY_TEMPLATE',
    'GEOPOLITICAL_TEMPLATE',
    'LOGISTICS_TEMPLATE',
]
