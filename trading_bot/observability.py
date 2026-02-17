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
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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
    'count_directional_evidence',
]

BULLISH_WORDS = {'increase', 'rise', 'surge', 'shortage', 'deficit', 'drought', 'frost', 'bullish', 'strong', 'growth', 'up', 'rose', 'gained', 'rally', 'congestion', 'bottleneck', 'backwardation', 'tightness', 'disruption', 'delay', 'restricted', 'drawdown', 'depletion', 'thinning', 'rationing', 'hawkish', 'hoarding', 'scarcity'}
BEARISH_WORDS = {'decrease', 'fall', 'surplus', 'bumper', 'oversupply', 'bearish', 'weak', 'decline', 'down', 'fell', 'lost', 'crash', 'plunge', 'contango', 'glut', 'abundance', 'oversupplied', 'liquidation', 'selloff', 'selling', 'overproduction', 'ample', 'easing', 'normalizing', 'weakening', 'buildup', 'accumulation', 'stockpile', 'plentiful', 'resolved', 'resolution', 'deleverage'}
NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nor', 'rejected', 'failed',
                  'despite', 'unlikely', 'against', 'overcame', 'ignored', 'dismissed',
                  'without', 'lack', 'absence', 'declining', 'decreased'}

def count_directional_evidence(text: str) -> tuple:
    """Count bullish vs bearish evidence words with negation awareness. Commodity-agnostic."""
    words = text.lower().split()
    bullish_count = 0
    bearish_count = 0

    for i, word in enumerate(words):
        # Check for negation in preceding 3 words
        context_start = max(0, i - 3)
        preceding = set(words[context_start:i])
        is_negated = bool(preceding & NEGATION_WORDS)

        # Simple stemming/matching (could be improved)
        # Check if word contains any of the target roots?
        # The constants above are full words. Let's do exact match for now as per guide snippet style.
        # But 'increase' vs 'increased' vs 'increasing'.
        # The guide snippet showed simple "word in BULLISH_WORDS".
        # I'll stick to exact match or simple contains if robust.
        # Original code was: word in ['INCREASE', ...] if word in grounded_data.upper().
        # That was substring search in the whole text.
        # Here we are iterating words.
        # Let's check for containment to handle suffixes.

        is_bullish = any(bw in word for bw in BULLISH_WORDS)
        is_bearish = any(bw in word for bw in BEARISH_WORDS)

        if is_bullish:
            if is_negated:
                bearish_count += 1  # "not bullish" → bearish evidence
            else:
                bullish_count += 1
        elif is_bearish:
            if is_negated:
                bullish_count += 1  # "not bearish" → bullish evidence
            else:
                bearish_count += 1

    return bullish_count, bearish_count


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
    grounded_data: str = ""  # Fix B3: Capture grounded search results

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
        quarantine_threshold: int = 5,
        data_dir: str = None
    ):
        """
        Initialize detector with commodity profile.

        Args:
            profile: CommodityProfile instance for commodity-specific facts
            quarantine_threshold: Number of flags before quarantine
            data_dir: Commodity-specific data directory (e.g. data/KC)
        """
        self.profile = profile
        self.quarantine_threshold = quarantine_threshold
        self.agent_flags: Dict[str, List[HallucinationFlag]] = {}
        self.quarantined_agents: Set[str] = set()
        import os as _os
        self._state_file = _os.path.join(data_dir, "quarantine_state.json") if data_dir else "data/KC/quarantine_state.json"
        self._load_state()

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

            # Legitimate data sources from profile (Issue 3)
            'legitimate_data_sources': set(profile.legitimate_data_sources),

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
        """Check output with per-cycle deduplication, memory pruning, and auto-release logic."""
        flags = []

        source_material = "\n".join(retrieved_docs)
        if grounded_data:
            source_material += "\n" + grounded_data

        # Run Checks
        # Skip citation check when agent used grounded search — the agent cites
        # real sources from Google results whose names may not appear verbatim in
        # the raw summary text, causing persistent false positives & quarantine.
        if not grounded_data:
            flags.extend(self._check_citations(output_text, source_material))
        flags.extend(self._check_numbers(output_text, source_material))
        flags.extend(self._check_facts(output_text))

        # --- PER-CYCLE DEDUPLICATION ---
        # Only count one flag per severity/type per cycle to prevent instant quarantine
        deduplicated_flags = {}
        for flag in flags:
            # Key by severity + generic description type (before the colon)
            key = (flag.severity, flag.description.split(':')[0])
            if key not in deduplicated_flags:
                deduplicated_flags[key] = flag

        final_flags = list(deduplicated_flags.values())

        # Record flags
        if agent not in self.agent_flags:
            self.agent_flags[agent] = []

        for flag in final_flags:
            flag.agent = agent
            flag.timestamp = datetime.now(timezone.utc)
            self.agent_flags[agent].append(flag)
            logger.warning(f"Hallucination detected [{flag.severity.value}]: {agent} - {flag.description}")

        # --- MEMORY SAFETY PRUNING (AMENDMENT 1) ---
        # Keep only last 30 days of flags to prevent memory leak / OOM crash.
        # On a 4GB droplet, unbounded growth could reach 100MB+ after 30 days.
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self.agent_flags[agent] = [f for f in self.agent_flags[agent] if f.timestamp > cutoff]

        # --- QUARANTINE LOGIC (Count Cycles, Not Flags) ---
        recent_flags = [
            f for f in self.agent_flags[agent]
            if (datetime.now(timezone.utc) - f.timestamp).days < 7
        ]

        # Group by 5-second buckets to identify distinct flawed cycles
        cycles_with_flags = set()
        for f in recent_flags:
            bucket = f.timestamp.replace(second=(f.timestamp.second // 5) * 5, microsecond=0)
            cycles_with_flags.add(bucket)

        if len(cycles_with_flags) >= self.quarantine_threshold:
            if agent not in self.quarantined_agents:
                self.quarantined_agents.add(agent)
                logger.error(f"Agent {agent} QUARANTINED: {len(cycles_with_flags)} flawed cycles in 7 days.")

        # --- AUTO-RELEASE LOGIC (AMENDMENT 2: handles empty recent_flags) ---
        if agent in self.quarantined_agents:
            if not recent_flags:
                # No flags at all in 7-day window → definitely release
                self.quarantined_agents.discard(agent)
                logger.info(f"Agent {agent} AUTO-RELEASED from quarantine (no flags in 7-day window)")
            else:
                most_recent = max(f.timestamp for f in recent_flags)
                hours_since = (datetime.now(timezone.utc) - most_recent).total_seconds() / 3600
                if hours_since > 48:
                    self.quarantined_agents.discard(agent)
                    logger.info(f"Agent {agent} AUTO-RELEASED from quarantine (clean for {hours_since:.1f}h)")

        return final_flags

    def is_quarantined(self, agent: str) -> bool:
        """Check if agent is quarantined."""
        return agent in self.quarantined_agents

    def release_from_quarantine(self, agent: str) -> None:
        """Manually release an agent from quarantine."""
        self.quarantined_agents.discard(agent)
        logger.info(f"Agent {agent} released from quarantine")
        self._save_state()

    def _save_state(self):
        """Persist quarantine state to survive restarts."""
        try:
            state = {
                'quarantined_agents': list(self.quarantined_agents),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            # Create data dir if needed
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save quarantine state: {e}")

    def _load_state(self):
        """Load quarantine state from disk."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                self.quarantined_agents = set(state.get('quarantined_agents', []))
                if self.quarantined_agents:
                    logger.info(f"Restored quarantine state: {self.quarantined_agents}")
            except Exception as e:
                logger.warning(f"Could not load quarantine state: {e}")

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip Unicode accents for robust comparison (e.g. Cecafé → cecafe)."""
        return unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode('ascii')

    def _check_citations(self, output: str, sources: str) -> List[HallucinationFlag]:
        """Verify citations using 3-tier fuzzy matching."""
        flags = []
        citations = self.SOURCE_PATTERN.findall(output)
        citations = [c[0] or c[1] for c in citations if c[0] or c[1]]

        source_lower = self._normalize(sources)
        # Tokenize sources for overlap check (use normalized text for accent-safe matching)
        source_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', source_lower))
        known_orgs = {self._normalize(s) for s in self.known_facts.get('organizations', set())}

        # === v5.2 FIX: Tier 0 — Known legitimate data sources (never flag) ===
        # Build from profile if available, with universal financial sources as baseline.
        # These are standard research sources agents reference from prompt context,
        # not from RAG retrieval. Commodity-agnostic: universal sources always included.
        UNIVERSAL_FINANCIAL_SOURCES = {
            'barchart', 'reuters', 'bloomberg', 'trading economics',
            'world bank', 'imf', 'cftc', 'commitments of traders',
            'cot report', 'raw research',
            # === FIX A4: Expanded known sources ===
            # Agents discover these via Phase 1 Gemini Search (AFC).
            # General financial sources (commodity-agnostic):
            'tradingview', 'trading view', 'investing.com', 'marketwatch',
            'cnbc', 'financial times', 'wsj', 'wall street journal',
            'yahoo finance', 'seeking alpha', 'refinitiv',
            'cme group', 'ice futures', 'intercontinental exchange',
            # Government/institutional sources:
            'usda', 'noaa', 'accuweather', 'weather.com',
        }
        # Profile-specific sources (if available) — normalized for accent-safe matching
        profile_sources = {
            self._normalize(s) for s in self.known_facts.get('legitimate_data_sources', set())
        }
        known_legitimate = UNIVERSAL_FINANCIAL_SOURCES | profile_sources

        for citation in citations:
            cit_norm = self._normalize(citation.strip())

            # Tier 0: Known legitimate source (NEVER flag)
            if any(known_src in cit_norm for known_src in known_legitimate):
                continue

            # Tier 1: Exact Match
            if cit_norm in source_lower:
                continue

            # Tier 2: Known Organization
            if any(org in cit_norm for org in known_orgs):
                continue

            # Tier 3: Token Overlap (>60%)
            cit_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', cit_norm))
            if cit_words:
                overlap = len(cit_words & source_words) / len(cit_words)
                if overlap >= 0.6:
                    continue

            flags.append(HallucinationFlag(
                timestamp=datetime.now(timezone.utc),
                agent="",
                severity=HallucinationSeverity.MEDIUM,  # Downgraded from HIGH
                description="Cited source not found in retrieved documents",
                claim=f"Source: {citation}",
                evidence="Low semantic overlap with RAG results"
            ))
        return flags

    def _check_numbers(self, output: str, sources: str) -> List[HallucinationFlag]:
        """Verify numbers using aggregate checking and widened tolerance."""
        flags = []
        numbers = self.NUMBER_PATTERN.findall(output)
        if not numbers:
            return flags

        # Extract all source numbers once
        source_nums = []
        for s in re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', sources):
            try:
                source_nums.append(float(s.replace(',', '')))
            except (ValueError, OverflowError):
                continue

        # SAFETY GUARD: If sources contain no numbers, we can't ground anything.
        # Skip the check rather than flagging all agent numbers as ungrounded.
        if not source_nums:
            return flags

        ungrounded_count = 0
        total_significant = 0
        examples = []

        for num, unit in numbers:
            try:
                val = float(num.replace(',', ''))
                if val < 100:
                    continue  # Skip small numbers/percentages
                total_significant += 1

                # Check 15% tolerance (widened from 10% to handle rounding/conversion)
                match = any(abs(s - val) / max(val, 1) < 0.15 for s in source_nums)
                if not match:
                    ungrounded_count += 1
                    if len(examples) < 3:
                        examples.append(f"{num} {unit}")
            except (ValueError, OverflowError):
                continue

        # AGGREGATE flagging: only flag if >50% of significant numbers are ungrounded
        # AND at least 3 ungrounded (safety rail against small-sample false positives)
        if total_significant > 0 and ungrounded_count >= 3:
            if (ungrounded_count / total_significant) > 0.5:
                flags.append(HallucinationFlag(
                    timestamp=datetime.now(timezone.utc),
                    agent="",
                    severity=HallucinationSeverity.MEDIUM,
                    description=f"{ungrounded_count}/{total_significant} significant numbers not grounded",
                    claim=f"Examples: {', '.join(examples)}",
                    evidence="Numbers not found in RAG source text"
                ))
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

    def __init__(self, profile, data_dir: str = None):
        """
        Initialize hub with commodity profile.

        Args:
            profile: CommodityProfile for commodity-specific fact checking
            data_dir: Commodity-specific data directory (e.g. data/KC)
        """
        self.profile = profile
        self.traces: List[AgentTrace] = []
        self.hallucination_detector = HallucinationDetector(profile, data_dir=data_dir)
        self.cost_tracker: Dict[str, float] = {}

        logger.info(f"ObservabilityHub initialized for {profile.name}")

    def record_trace(self, trace: AgentTrace) -> None:
        """Record an agent execution trace."""
        # Run hallucination detection
        flags = self.hallucination_detector.check_output(
            agent=trace.agent,
            output_text=trace.output_text,
            retrieved_docs=trace.retrieved_documents,
            grounded_data=trace.grounded_data  # Fix B3: Pass grounded data
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
