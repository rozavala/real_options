"""
Observability Hub with Hallucination Detection.

This module provides:
- Agent trace logging (inputs, reasoning, outputs)
- Hallucination detection via source grounding
- Citation verification
- Automatic agent quarantine for repeated violations
"""

import re
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)

# FIX (MECE V2 #4): Explicit exports for clean imports
__all__ = [
    'HallucinationSeverity',
    'HallucinationFlag',
    'AgentTrace',
    'HallucinationDetector',
    'ObservabilityHub',
]


class HallucinationSeverity(Enum):
    """Severity levels for hallucination flags."""
    LOW = "LOW"          # Minor inconsistency
    MEDIUM = "MEDIUM"    # Unverified claim
    HIGH = "HIGH"        # Fabricated source
    CRITICAL = "CRITICAL"  # Fabricated data with trade impact


@dataclass
class HallucinationFlag:
    """Record of a detected hallucination."""
    timestamp: datetime
    agent: str
    severity: HallucinationSeverity
    description: str
    claim: str
    evidence: Optional[str] = None


@dataclass
class AgentTrace:
    """Complete trace of an agent's execution."""
    agent: str
    timestamp: datetime

    # Input
    query: str
    retrieved_documents: List[str] = field(default_factory=list)

    # Processing
    reasoning_steps: List[str] = field(default_factory=list)

    # Output
    output_text: str = ""
    sentiment: str = ""
    confidence: float = 0.0

    # Validation
    hallucination_flags: List[HallucinationFlag] = field(default_factory=list)
    is_valid: bool = True

    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = ""


class HallucinationDetector:
    """
    Detects hallucinations in agent outputs by cross-referencing claims
    against retrieved documents and known facts.

    IMPORTANT: This class is COMMODITY-AGNOSTIC. Known facts are derived
    dynamically from the CommodityProfile, not hardcoded.
    """

    # Patterns for extracting claims (commodity-agnostic)
    NUMBER_PATTERN = re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(bags|contracts|tons|cents|%|percent|barrels|bushels|ounces)\b', re.IGNORECASE)
    SOURCE_PATTERN = re.compile(r'\[Source:\s*([^\]]+)\]|\(source:\s*([^)]+)\)', re.IGNORECASE)
    DATE_PATTERN = re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)

    def __init__(
        self,
        profile,  # 'CommodityProfile' - type hint omitted to avoid circular import if needed
        quarantine_threshold: int = 5
    ):
        """
        Initialize detector with commodity profile.

        Args:
            profile: CommodityProfile instance for commodity-specific facts
            quarantine_threshold: Number of flags before quarantine
        """
        self.profile = profile
        self.quarantine_threshold = quarantine_threshold
        self.agent_flags: Dict[str, List[HallucinationFlag]] = {}
        self.quarantined_agents: Set[str] = set()

        # DYNAMICALLY build known facts from the profile
        # This ensures commodity-agnostic operation

        # FIX: Tokenize hub names to prevent false positives
        # "Port of Santos" → {"Port of Santos", "Port", "of", "Santos"}
        # This allows partial matches like "congestion at Santos"
        port_tokens = set()
        for h in profile.logistics_hubs:
            port_tokens.add(h.name)  # Full name: "Port of Santos"
            port_tokens.update(h.name.split())  # Tokens: "Port", "of", "Santos"

        # Same for producer regions - tokenize country and region names
        producer_tokens = set()
        for r in profile.primary_regions:
            producer_tokens.add(r.country)
            producer_tokens.add(r.name)
            producer_tokens.update(r.name.split())  # "Minas Gerais" → "Minas", "Gerais"

        self.known_facts = {
            # Contract months from profile
            'contract_months': set(profile.contract.contract_months),

            # Ports/logistics hubs - TOKENIZED to prevent false positives
            'ports': port_tokens,

            # Producing countries/regions - TOKENIZED
            'producers': producer_tokens,

            # Data sources/organizations from profile
            'organizations': set(profile.inventory_sources),

            # Commodity-specific keywords (derived from news_keywords)
            'keywords': set(profile.news_keywords) if profile.news_keywords else set(),
        }

        # Remove common words that would cause noise
        noise_words = {'of', 'the', 'and', 'in', 'at', 'to', 'for', 'a', 'an', 'Port', 'City'}
        self.known_facts['ports'] -= noise_words
        self.known_facts['producers'] -= noise_words

        logger.info(
            f"HallucinationDetector initialized for {profile.name}: "
            f"{len(self.known_facts['ports'])} port tokens, "
            f"{len(self.known_facts['producers'])} producer tokens"
        )

    def check_output(
        self,
        agent: str,
        output_text: str,
        retrieved_docs: List[str],
        grounded_data: Optional[str] = None
    ) -> List[HallucinationFlag]:
        """
        Check agent output for hallucinations.

        Args:
            agent: Agent name
            output_text: The agent's output text
            retrieved_docs: Documents retrieved by RAG
            grounded_data: Raw grounded data packet

        Returns:
            List of hallucination flags (empty if clean)
        """
        flags = []

        # Combine all source material
        source_material = "\n".join(retrieved_docs)
        if grounded_data:
            source_material += "\n" + grounded_data

        # Check 1: Citation verification
        citation_flags = self._check_citations(output_text, source_material)
        flags.extend(citation_flags)

        # Check 2: Number grounding
        number_flags = self._check_numbers(output_text, source_material)
        flags.extend(number_flags)

        # Check 3: Factual verification
        fact_flags = self._check_facts(output_text)
        flags.extend(fact_flags)

        # Record flags for this agent
        if agent not in self.agent_flags:
            self.agent_flags[agent] = []

        for flag in flags:
            flag.agent = agent
            flag.timestamp = datetime.now(timezone.utc)
            self.agent_flags[agent].append(flag)

            logger.warning(f"Hallucination detected [{flag.severity.value}]: {agent} - {flag.description}")

        # Check for quarantine
        recent_flags = [
            f for f in self.agent_flags[agent]
            if (datetime.now(timezone.utc) - f.timestamp).days < 7
        ]

        if len(recent_flags) >= self.quarantine_threshold:
            self.quarantined_agents.add(agent)
            logger.error(f"Agent {agent} QUARANTINED: {len(recent_flags)} hallucination flags in 7 days")

        return flags

    def is_quarantined(self, agent: str) -> bool:
        """Check if agent is quarantined."""
        return agent in self.quarantined_agents

    def release_from_quarantine(self, agent: str) -> None:
        """Manually release an agent from quarantine."""
        self.quarantined_agents.discard(agent)
        logger.info(f"Agent {agent} released from quarantine")

    def _check_citations(self, output: str, sources: str) -> List[HallucinationFlag]:
        """Verify that cited sources exist in retrieved documents."""
        flags = []

        # Extract citations from output
        citations = self.SOURCE_PATTERN.findall(output)
        citations = [c[0] or c[1] for c in citations if c[0] or c[1]]

        for citation in citations:
            # Check if citation appears in sources
            if citation.lower() not in sources.lower():
                flags.append(HallucinationFlag(
                    timestamp=datetime.now(timezone.utc),
                    agent="",
                    severity=HallucinationSeverity.HIGH,
                    description=f"Cited source not found in retrieved documents",
                    claim=f"Source: {citation}",
                    evidence="Source not in RAG results"
                ))

        return flags

    def _check_numbers(self, output: str, sources: str) -> List[HallucinationFlag]:
        """Verify that specific numbers are grounded in sources."""
        flags = []

        # Extract numbers with units from output
        numbers = self.NUMBER_PATTERN.findall(output)

        for num, unit in numbers:
            # Check if this number appears in sources
            num_clean = num.replace(',', '')

            # Allow some variance (within 10%)
            try:
                num_float = float(num_clean)
                found = False

                for source_num in self.NUMBER_PATTERN.findall(sources):
                    try:
                        source_float = float(source_num[0].replace(',', ''))
                        if abs(source_float - num_float) / max(num_float, 1) < 0.10:
                            found = True
                            break
                    except ValueError:
                        continue

                if not found and num_float > 100:  # Only flag significant numbers
                    flags.append(HallucinationFlag(
                        timestamp=datetime.now(timezone.utc),
                        agent="",
                        severity=HallucinationSeverity.MEDIUM,
                        description=f"Number not found in source documents",
                        claim=f"{num} {unit}",
                        evidence="Number not grounded in RAG"
                    ))
            except ValueError:
                continue

        return flags

    def _check_facts(self, output: str) -> List[HallucinationFlag]:
        """Check for factual errors against known facts from CommodityProfile."""
        flags = []
        output_lower = output.lower()

        # Check for invalid contract months (using profile-derived months)
        # Pattern matches: H26, K25, N24, etc.
        month_pattern = re.compile(r'\b([A-Z])\d{2}\b')  # e.g., "H26", "K25"

        # Skip common FINANCIAL abbreviations that look like contract codes:
        # - Q = Quarter (Q1, Q2, Q3, Q4 → Q25 = "in 2025")
        # - Y = Year (FY25, CY25 → Y25 would match)
        # - S = Semester (S1, S2 → S25 would match)
        # - FY = Fiscal Year
        # - CY = Calendar Year
        SKIP_FINANCIAL_ABBREVS = {'Q', 'Y', 'S', 'FY', 'CY'}  # NOT including H!

        for match in month_pattern.findall(output):
            if match in SKIP_FINANCIAL_ABBREVS:
                continue

            if match not in self.known_facts['contract_months']:
                flags.append(HallucinationFlag(
                    timestamp=datetime.now(timezone.utc),
                    agent="",
                    severity=HallucinationSeverity.LOW,
                    description=f"Invalid contract month code for {self.profile.name}",
                    claim=f"Contract month: {match}",
                    evidence=f"Valid months for {self.profile.ticker}: {self.known_facts['contract_months']}"
                ))

        # Check for mentions of ports not in the profile
        # (Only flag if a port is explicitly claimed as a major hub)
        port_claim_pattern = re.compile(r'(?:port of|major hub|key port)\s+(\w+)', re.IGNORECASE)
        for match in port_claim_pattern.findall(output):
            # FIX: Case-insensitive comparison to prevent false positives
            match_lower = match.lower()
            known_ports_lower = {p.lower() for p in self.known_facts['ports']}

            if match_lower not in known_ports_lower and match_lower not in {'the', 'a', 'an'}:
                # Only flag as LOW - could be a valid secondary port
                flags.append(HallucinationFlag(
                    timestamp=datetime.now(timezone.utc),
                    agent="",
                    severity=HallucinationSeverity.LOW,
                    description=f"Port not in {self.profile.name} logistics hubs",
                    claim=f"Port: {match}",
                    evidence=f"Known hubs: {self.known_facts['ports']}"
                ))

        return flags

    def get_agent_stats(self, agent: str) -> Dict:
        """Get hallucination statistics for an agent."""
        flags = self.agent_flags.get(agent, [])

        recent = [f for f in flags if (datetime.now(timezone.utc) - f.timestamp).days < 7]

        severity_counts = {s.value: 0 for s in HallucinationSeverity}
        for f in recent:
            severity_counts[f.severity.value] += 1

        return {
            'total_flags': len(flags),
            'recent_flags': len(recent),
            'is_quarantined': agent in self.quarantined_agents,
            'severity_breakdown': severity_counts
        }


class ObservabilityHub:
    """
    Central hub for agent observability.

    Collects traces, detects hallucinations, tracks costs.

    IMPORTANT: Must be initialized with a CommodityProfile to ensure
    commodity-agnostic hallucination detection.
    """

    def __init__(self, profile):
        """
        Initialize hub with commodity profile.

        Args:
            profile: CommodityProfile for commodity-specific fact checking
        """
        self.profile = profile
        self.traces: List[AgentTrace] = []
        self.hallucination_detector = HallucinationDetector(profile)
        self.cost_tracker: Dict[str, float] = {}

        logger.info(f"ObservabilityHub initialized for {profile.name}")

    def record_trace(self, trace: AgentTrace) -> None:
        """Record an agent execution trace."""
        # Run hallucination detection
        flags = self.hallucination_detector.check_output(
            agent=trace.agent,
            output_text=trace.output_text,
            retrieved_docs=trace.retrieved_documents
        )

        trace.hallucination_flags = flags
        trace.is_valid = len([f for f in flags if f.severity in [HallucinationSeverity.HIGH, HallucinationSeverity.CRITICAL]]) == 0

        self.traces.append(trace)

        # Track costs
        cost = self._estimate_cost(trace)
        if trace.agent not in self.cost_tracker:
            self.cost_tracker[trace.agent] = 0.0
        self.cost_tracker[trace.agent] += cost

    def is_agent_valid(self, agent: str) -> bool:
        """Check if agent is currently valid (not quarantined)."""
        return not self.hallucination_detector.is_quarantined(agent)

    def get_cost_summary(self) -> Dict:
        """Get cost summary by agent."""
        return {
            'by_agent': self.cost_tracker.copy(),
            'total': sum(self.cost_tracker.values())
        }

    def _estimate_cost(self, trace: AgentTrace) -> float:
        """
        Estimate API cost for a trace.

        FIX (MECE 1.5): Load costs from config file instead of hardcoding.
        This allows easy updates when API pricing changes.
        """
        # Try to load from config file (updated manually when prices change)
        costs = self._load_cost_config()

        model_key = 'default'
        model_lower = trace.model_name.lower()

        # Match model to cost config
        for key in costs:
            if key in model_lower:
                model_key = key
                break

        # Get cost per 1K tokens (average of input/output if available)
        model_cost = costs.get(model_key, costs.get('default', 0.001))
        if isinstance(model_cost, dict):
            # Has input/output breakdown
            input_cost = model_cost.get('input', 0.001)
            output_cost = model_cost.get('output', 0.002)
            cost = (trace.input_tokens / 1000) * input_cost + (trace.output_tokens / 1000) * output_cost
        else:
            # Single rate
            total_tokens = trace.input_tokens + trace.output_tokens
            cost = (total_tokens / 1000) * model_cost

        return cost

    def _load_cost_config(self) -> Dict:
        """Load API costs from config file, with fallback to defaults."""
        config_path = "config/api_costs.json"

        # Default costs (fallback if config file missing)
        DEFAULT_COSTS = {
            'gemini-2.0-flash': {'input': 0.00010, 'output': 0.00040},
            'gemini-2.0-pro': {'input': 0.00125, 'output': 0.00500},
            'gpt-4o': {'input': 0.00250, 'output': 0.01000},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.00060},
            'claude-sonnet': {'input': 0.00300, 'output': 0.01500},
            'claude-opus': {'input': 0.01500, 'output': 0.07500},
            'grok': {'input': 0.00500, 'output': 0.01000},
            'default': {'input': 0.00100, 'output': 0.00200},
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    import json
                    config = json.load(f)
                    return config.get('costs_per_1k_tokens', DEFAULT_COSTS)
        except Exception as e:
            logger.warning(f"Failed to load API cost config: {e}. Using defaults.")

        return DEFAULT_COSTS
