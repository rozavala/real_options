"""
Agent Prompts Package.

This package contains templatized prompts for all agents,
supporting commodity-agnostic operations.
"""

from .base_prompts import (
    get_agent_prompt,
    AgentPromptTemplate,  # Export the class
)

__all__ = [
    'get_agent_prompt',
    'AgentPromptTemplate',
]
