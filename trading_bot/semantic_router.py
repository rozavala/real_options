from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RouteDecision:
    primary_agent: str
    secondary_agents: List[str]
    confidence: float
    reasoning: str

class SemanticRouter:
    """Routes sentinel triggers to appropriate agents using lightweight classification."""

    ROUTE_MATRIX = {
        # (trigger_source, keyword_hints) -> (primary, secondaries)
        ("WeatherSentinel", "frost"): ("agronomist", ["supply_chain"]),
        ("WeatherSentinel", "drought"): ("agronomist", ["inventory"]),
        ("WeatherSentinel", "flood"): ("agronomist", ["supply_chain", "geopolitical"]),
        ("PriceSentinel", "spike"): ("technical", ["sentiment", "macro"]),
        ("PriceSentinel", "crash"): ("technical", ["macro", "geopolitical"]),
        ("LogisticsSentinel", "strike"): ("supply_chain", ["geopolitical"]),
        ("LogisticsSentinel", "port"): ("supply_chain", ["inventory"]),
        ("NewsSentinel", "default"): ("sentiment", ["macro", "geopolitical"]),
        ("PredictionMarketSentinel", "fed"): ("macro", ["sentiment", "technical"]),
        ("PredictionMarketSentinel", "brazil"): ("macro", ["agronomist", "supply_chain"]),
        ("PredictionMarketSentinel", "election"): ("geopolitical", ["macro", "sentiment"]),
        ("PredictionMarketSentinel", "tariff"): ("geopolitical", ["macro", "supply_chain"]),
        ("PredictionMarketSentinel", "default"): ("macro", ["geopolitical", "sentiment"]),
    }

    def __init__(self, config: dict):
        self.config = config
        # Optional: Initialize lightweight classifier here

    def route(self, trigger) -> RouteDecision:
        reason_lower = trigger.reason.lower()

        # Keyword matching (fast path)
        for (source, hint), (primary, secondaries) in self.ROUTE_MATRIX.items():
            if trigger.source == source and hint in reason_lower:
                return RouteDecision(
                    primary_agent=primary,
                    secondary_agents=secondaries,
                    confidence=0.8,
                    reasoning=f"Matched: {source}/{hint}"
                )

        # Default fallback by source
        defaults = {
            "WeatherSentinel": "agronomist",
            "LogisticsSentinel": "supply_chain",
            "PriceSentinel": "technical",
            "NewsSentinel": "sentiment",
            "PredictionMarketSentinel": "macro"
        }
        return RouteDecision(
            primary_agent=defaults.get(trigger.source, "macro"),
            secondary_agents=["macro"],
            confidence=0.5,
            reasoning="Default routing"
        )
