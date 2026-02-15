"""The Trading Council Module (Updated for google-genai SDK).

This module implements the '5-headed beast' agent system using Google Gemini models.
It includes specialized research agents, a Dialectical Debate (Permabear/Permabull),
and a Master Strategist that synthesizes reports to make final trading decisions.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from google import genai
from google.genai import types
from trading_bot.confidence_utils import parse_confidence
from trading_bot.state_manager import StateManager
from trading_bot.sentinels import SentinelTrigger
from trading_bot.semantic_router import SemanticRouter
from trading_bot.market_data_provider import format_market_context_for_prompt
from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole, get_router, CriticalRPCError
from trading_bot.budget_guard import BudgetThrottledError
from trading_bot.tms import TransactiveMemory
from trading_bot.reliability_scorer import ReliabilityScorer
from trading_bot.weighted_voting import calculate_weighted_decision, determine_trigger_type
from trading_bot.enhanced_brier import EnhancedBrierTracker
from trading_bot.observability import ObservabilityHub, AgentTrace

logger = logging.getLogger(__name__)

# === NEW IMPORTS FOR COMMODITY-AGNOSTIC SUPPORT ===
# Added for HRO Enhancement - does not affect existing functionality
# FIX (MECE 1.2): More explicit error handling to prevent silent failures
_HAS_COMMODITY_PROFILES = False
_PROFILE_IMPORT_ERROR = None  # Track why import failed for debugging

try:
    from config.commodity_profiles import get_commodity_profile, CommodityProfile
    from trading_bot.prompts.base_prompts import get_agent_prompt
    _HAS_COMMODITY_PROFILES = True
except ImportError as e:
    _PROFILE_IMPORT_ERROR = f"ImportError: {e}"
    logger.warning(f"Commodity profiles not available: {e}. Using legacy prompts.")
except Exception as e:
    _PROFILE_IMPORT_ERROR = f"Unexpected: {e}"
    logger.error(f"Unexpected error loading commodity profiles: {e}. Using legacy prompts.")

# Log at module load time for debugging
if not _HAS_COMMODITY_PROFILES:
    logger.info(f"Profile system disabled. Reason: {_PROFILE_IMPORT_ERROR or 'Unknown'}")

# Setup Provenance Logger
provenance_logger = logging.getLogger("provenance")
provenance_logger.setLevel(logging.INFO)
provenance_logger.propagate = False
if not provenance_logger.handlers:
    from logging.handlers import RotatingFileHandler
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(
        os.path.join(log_dir, 'research_provenance.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=3
    )
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    provenance_logger.addHandler(fh)

@dataclass
class GroundedDataPacket:
    """Container for grounded research data with provenance."""
    search_query: str
    gathered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_findings: str = ""
    extracted_facts: List[Dict[str, str]] = field(default_factory=list)  # [{fact, date, source}]
    source_urls: List[str] = field(default_factory=list)
    data_freshness_hours: int = 4  # CHANGED from 48 → 4 (soft penalty zone default)

    def to_context_block(self) -> str:
        """Format as context block for analyst prompts."""
        facts_str = "\n".join([
            f"- [{f.get('date', 'undated')}] {f.get('fact')} (source: {f.get('source', 'unknown')})"
            for f in self.extracted_facts
        ]) or "No specific dated facts extracted."

        return f"""
=== GROUNDED DATA PACKET ===
Query: {self.search_query}
Gathered: {self.gathered_at.isoformat()}
Data Freshness Target: Last {self.data_freshness_hours} hours

EXTRACTED FACTS:
{facts_str}

RAW RESEARCH FINDINGS:
{self.raw_findings[:2000]}  # Truncate for context limits

SOURCES CONSULTED:
{chr(10).join(self.source_urls[:5]) or 'No URLs captured'}
=== END GROUNDED DATA ===
"""

DEBATE_OUTPUT_SCHEMA = """
Output JSON with this structure:
{
    "position": "BULLISH|BEARISH",
    "key_arguments": [
        {"claim": "...", "evidence": "...", "confidence": 0.0-1.0}
    ],
    "acknowledged_risks": ["..."],
    "market_data_alignment": "AGREES|DISAGREES|NEUTRAL"
}
"""

class ResponseTimeTracker:
    def __init__(self):
        self.latencies: Dict[str, List[float]] = {}

    def record(self, provider: str, latency: float):
        if provider not in self.latencies:
            self.latencies[provider] = []
        self.latencies[provider].append(latency)
        # Keep last 100
        self.latencies[provider] = self.latencies[provider][-100:]

    def get_avg_latency(self, provider: str) -> float:
        if provider not in self.latencies or not self.latencies[provider]:
            return 0.0
        return sum(self.latencies[provider]) / len(self.latencies[provider])

class TradingCouncil:
    """Manages the tiered multi-model agent system for commodity futures trading."""

    def __init__(self, config: dict):
        """
        Initializes the Trading Council with configuration.

        Args:
            config: The full application configuration dictionary.
        """
        self.full_config = config
        self.config = config.get('gemini', {})
        self.personas = self.config.get('personas', {})
        self.semantic_router = SemanticRouter(config)

        # 1. Setup API Key
        api_key = self.config.get('api_key')
        # If config key is placeholder or empty, try standard env var
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            api_key = os.environ.get('GEMINI_API_KEY')

        if not api_key:
            raise ValueError("Gemini API Key not found. Please set GEMINI_API_KEY env var or update config.json.")

        # 2. Initialize the New Client
        self.client = genai.Client(api_key=api_key)

        # 3. Model Names
        self.agent_model_name = self.config.get('agent_model', 'gemini-1.5-flash')
        self.master_model_name = self.config.get('master_model', 'gemini-1.5-pro')

        # Semaphore for grounded data calls (rate limiting)
        self._grounded_data_semaphore = asyncio.Semaphore(3)

        # 5. Initialize Heterogeneous Router (for multi-model support)
        self.heterogeneous_router = None
        try:
            self.heterogeneous_router = HeterogeneousRouter(config)
            self.use_heterogeneous = len(self.heterogeneous_router.available_providers) > 1
            if self.use_heterogeneous:
                logger.info(f"Heterogeneous routing enabled. Diversity: {self.heterogeneous_router.get_diversity_report()}")
        except Exception as e:
            logger.warning(f"Heterogeneous router init failed, using Gemini only: {e}")
            self.heterogeneous_router = None
            self.use_heterogeneous = False

        # 4. Safety Settings (Block Only High)
        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]

        # 6. Initialize Cognitive Infrastructure
        self.tms = TransactiveMemory()
        self.scorer = ReliabilityScorer()
        self.response_tracker = ResponseTimeTracker()
        self.brier_tracker = EnhancedBrierTracker()
        self._search_cache = {}  # Cache for grounded data

        # Rate limiter state for grounded data gathering (Gemini with tools)
        # Semaphore initialized lazily in _gather_grounded_data to ensure event loop safety.
        # See Flag 1 in Flight Director review: asyncio.Semaphore binds to the running
        # event loop at creation time; lazy init guarantees we're inside an async context.
        self._grounded_data_min_interval = 2.5  # seconds between calls (25 RPM = 2.4s, with margin)
        self._last_grounded_call_time = 0

        # ============================================================
        # NEW: Commodity Profile Integration (ADDITIVE)
        # This block is OPTIONAL - system works without it
        # ============================================================
        self.commodity_profile = None
        self.observability = None

        if _HAS_COMMODITY_PROFILES:
            try:
                ticker = config.get('commodity', {}).get('ticker', 'KC')
                self.commodity_profile = get_commodity_profile(ticker)
                logger.info(f"Loaded commodity profile: {self.commodity_profile.name}")

                # Initialize Observability (requires profile)
                try:
                    self.observability = ObservabilityHub(self.commodity_profile)
                    logger.info("Observability Hub initialized.")
                except Exception as e:
                    logger.warning(f"Observability init failed: {e}")

            except Exception as e:
                logger.warning(f"Commodity profile load failed, using legacy prompts: {e}")
                self.commodity_profile = None

        commodity_label = self.commodity_profile.name if self.commodity_profile else "General"
        logger.info(f"Trading Council initialized for {commodity_label}. Agents: {self.agent_model_name}, Master: {self.master_model_name}")

        # DSPy optimized prompt cache (loaded lazily, no dspy import)
        self._optimized_prompt_cache: dict = {}

        # Load TMS decay rates from commodity profile
        self._tms_decay_rates = getattr(
            self.commodity_profile, 'tms_decay_rates', {'default': 0.05}
        ) if self.commodity_profile else {'default': 0.05}

    def invalidate_grounded_data_cache(self):
        """Force-clear the grounded data cache.

        Call this before each order generation cycle to ensure agents
        perform fresh searches rather than reusing data from a prior
        cycle. Without this, a sentinel cycle's cached data can serve
        the order generation cycle hours later, causing freshness gate
        failures and citation mismatches.
        """
        if hasattr(self, '_search_cache'):
            cache_size = len(self._search_cache)
            self._search_cache.clear()
            logger.info(
                f"Grounded data cache invalidated ({cache_size} entries cleared). "
                f"Next cycle will perform fresh searches."
            )

    def _load_optimized_prompt(self, ticker: str, persona_key: str) -> dict | None:
        """Load DSPy-optimized prompt from disk, with in-memory cache."""
        cache_key = f"{ticker}/{persona_key}"
        if cache_key in self._optimized_prompt_cache:
            return self._optimized_prompt_cache[cache_key]

        dspy_config = self.full_config.get("dspy", {})
        prompts_dir = dspy_config.get("optimized_prompts_dir", "data/dspy_optimized")
        if not os.path.isabs(prompts_dir):
            prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), prompts_dir)

        prompt_path = os.path.join(prompts_dir, ticker, persona_key, "prompt.json")
        if not os.path.exists(prompt_path):
            logger.debug(f"No optimized prompt at {prompt_path}, using default")
            self._optimized_prompt_cache[cache_key] = None
            return None

        try:
            with open(prompt_path, "r") as f:
                data = json.load(f)
            self._optimized_prompt_cache[cache_key] = data
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load optimized prompt {prompt_path}: {e}")
            self._optimized_prompt_cache[cache_key] = None
            return None

    @staticmethod
    def _format_demos(demos: list[dict]) -> str:
        """Format few-shot demonstration examples as readable text."""
        parts = []
        for i, demo in enumerate(demos, 1):
            inp = demo.get("input", {})
            out = demo.get("output", {})
            parts.append(
                f"--- Example {i} ---\n"
                f"Context: {inp.get('market_context', 'N/A')}\n"
                f"Direction: {out.get('direction', 'N/A')}\n"
                f"Confidence: {out.get('confidence', 'N/A')}\n"
                f"Analysis: {out.get('analysis', 'N/A')}"
            )
        return "\n\n".join(parts)

    def _validate_agent_output(self, agent_name: str, analysis: str, grounded_data: str) -> tuple:
        """
        Rule-based sanity check on agent output.

        Returns (is_valid, issues, confidence_adjustment).
        confidence_adjustment: If not None, the confidence should be clamped to this value.
        """
        issues = []
        confidence_adjustment = None

        analysis_upper = analysis.upper()

        # Check 1: Direction-evidence consistency (ENHANCED)
        from trading_bot.observability import count_directional_evidence
        bullish_evidence, bearish_evidence = count_directional_evidence(grounded_data)

        # Only flag significant mismatches (requires sufficient evidence words):
        total_directional = bullish_evidence + bearish_evidence

        if total_directional >= 5:
            if 'BULLISH' in analysis_upper and bearish_evidence > bullish_evidence + 2:
                issues.append(f"Direction-evidence mismatch: BULLISH call but {bearish_evidence} bearish vs {bullish_evidence} bullish evidence words")
                # Penalize confidence proportional to mismatch severity
                mismatch_ratio = bearish_evidence / max(total_directional, 1)
                mismatch_adj = max(0.3, 1.0 - mismatch_ratio)  # Floor at 0.3
                # Use min() to not overwrite a stricter penalty from another check
                confidence_adjustment = min(confidence_adjustment, mismatch_adj) if confidence_adjustment is not None else mismatch_adj

            if 'BEARISH' in analysis_upper and bullish_evidence > bearish_evidence + 2:
                issues.append(f"Direction-evidence mismatch: BEARISH call but {bullish_evidence} bullish vs {bearish_evidence} bearish evidence words")
                mismatch_ratio = bullish_evidence / max(total_directional, 1)
                mismatch_adj = max(0.3, 1.0 - mismatch_ratio)  # Floor at 0.3
                confidence_adjustment = min(confidence_adjustment, mismatch_adj) if confidence_adjustment is not None else mismatch_adj
        else:
            logger.debug(f"[{agent_name}] Low directional word count ({total_directional}), skipping mismatch check")

        # Check 2: Over-confidence (ENHANCED with correction)
        import re
        conf_match = re.search(r'"confidence"\s*:\s*(0\.\d+|1\.0)', analysis)
        if conf_match:
            conf = float(conf_match.group(1))
            total_evidence = bullish_evidence + bearish_evidence
            if conf > 0.9 and total_evidence < 3:
                issues.append(f"Over-confidence: {conf} with low evidence")
                # Clamp confidence proportional to evidence
                # 0 evidence → max 0.55, 1 → 0.65, 2 → 0.75
                max_allowed = 0.55 + (total_evidence * 0.10)
                overconf_adj = min(conf, max_allowed)
                # Use min() to not overwrite a stricter penalty from Check 1
                confidence_adjustment = min(confidence_adjustment, overconf_adj) if confidence_adjustment is not None else overconf_adj

        # Check 3: Hallucination flag — numbers not in grounded data
        # v5.2 FIX: Context-aware thresholds and price-level whitelisting
        import re as re_mod
        output_numbers = set(re_mod.findall(r'\b\d{3,}\b', analysis))
        source_numbers = set(re_mod.findall(r'\b\d{3,}\b', grounded_data))
        fabricated = output_numbers - source_numbers

        # Whitelist price levels near the underlying price for technical/volatility/macro agents
        # These agents CALCULATE support/resistance levels that won't appear in input data.
        if agent_name in ('technical', 'volatility', 'macro'):
            try:
                price_numbers = [float(n) for n in source_numbers if 50 < float(n) < 1000]
                if price_numbers:
                    ref_price = max(price_numbers)  # Use highest as reference
                    fabricated = {
                        n for n in fabricated
                        if not (ref_price * 0.7 < float(n) < ref_price * 1.3)
                    }
            except (ValueError, ZeroDivisionError):
                pass  # If parsing fails, keep original check

        # Agents with web search capability get a higher threshold
        # (their LLM retrieved numbers via AFC that aren't in our local data)
        if agent_name in ('sentiment', 'geopolitical'):
            threshold = 5
        else:
            threshold = 3

        if len(fabricated) > threshold:
            issues.append(
                f"Possible hallucination: {len(fabricated)} numbers not in grounded data: "
                f"{list(fabricated)[:3]}"
            )

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning(f"[{agent_name}] Output validation issues: {issues}")

        return is_valid, issues, confidence_adjustment

    def _compute_data_freshness(self, dated_facts: list, freshness_text: str) -> int:
        """
        Compute actual data freshness in hours from extracted facts.

        Strategy:
        1. Parse dates from dated_facts → compute hours since newest fact
        2. If no parseable dates, use text heuristics on the freshness_text
        3. Default to 4 hours (within soft penalty range, but NOT hard gate)

        Returns:
            Integer hours representing how old the data is.
            Lower = fresher. 0-4 = fresh, 4-24 = moderately stale, >24 = critically stale.
        """
        now = datetime.now(timezone.utc)

        # === STRATEGY 1: Parse actual dates from facts ===
        if dated_facts:
            newest_age_hours = None
            for fact in dated_facts:
                date_str = fact.get('date', '') if isinstance(fact, dict) else ''
                date_str_lower = str(date_str).lower().strip()

                try:
                    if date_str_lower in ('today', 'now', 'current'):
                        newest_age_hours = 0
                        break  # Can't get fresher than today
                    elif date_str_lower == 'yesterday':
                        newest_age_hours = min(newest_age_hours or 999, 24)
                    elif date_str_lower == 'this week':
                        newest_age_hours = min(newest_age_hours or 999, 72)
                    else:
                        # Try parsing YYYY-MM-DD or other date formats
                        from dateutil import parser as date_parser
                        parsed_date = date_parser.parse(date_str)
                        if parsed_date.tzinfo is None:
                            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                        age = (now - parsed_date).total_seconds() / 3600
                        if age >= 0:  # Ignore future dates (hallucination guard — Flag 3)
                            newest_age_hours = min(newest_age_hours or 999, age)
                except (ValueError, TypeError, OverflowError):
                    continue

            if newest_age_hours is not None:
                return max(0, int(newest_age_hours))

        # === STRATEGY 2: Text heuristics on freshness_text ===
        ft_lower = str(freshness_text).lower()
        if any(kw in ft_lower for kw in ['today', 'last hour', 'minutes ago', 'just now', 'real-time']):
            return 1
        elif any(kw in ft_lower for kw in ['yesterday', 'last 24', '24 hour', 'past day']):
            return 18  # Conservative — "yesterday" could be up to ~36h ago
        elif any(kw in ft_lower for kw in ['last 48', '2 day', 'two day', 'past 48']):
            return 36
        elif any(kw in ft_lower for kw in ['this week', 'last week', 'past week', 'several day']):
            return 96
        elif 'no recent' in ft_lower or 'unavailable' in ft_lower:
            return 999  # Genuinely stale — should trip the gate

        # === STRATEGY 3: Default ===
        # CRITICAL: Default must NOT auto-trip the 24h hard gate.
        # 4 hours places it in the "soft penalty" zone, which is the
        # correct conservative-but-not-blocking default.
        logger.info(f"Data freshness unparseable ('{freshness_text}'). Defaulting to 4h (soft penalty zone).")
        return 4

    def _clean_json_text(self, text: str) -> str:
        """Helper to strip markdown code blocks from JSON strings."""
        if not text:
            return "{}"  # Safe default for empty/None responses
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    async def _route_call(self, role: AgentRole, prompt: str, system_prompt: str = None, response_json: bool = False) -> str:
        """
        Route call to appropriate model based on role.
        Falls back to Gemini if router unavailable.
        """
        start_time = time.time()
        try:
            if self.use_heterogeneous and self.heterogeneous_router:
                try:
                    result = await self.heterogeneous_router.route(role, prompt, system_prompt, response_json)
                    return result
                except BudgetThrottledError:
                    raise  # Don't fall back to Gemini — budget applies to all providers
                except Exception as e:
                    logger.warning(f"Router failed for {role.value}, falling back to Gemini: {e}")

            # Fallback to existing Gemini implementation
            model_to_use = self.master_model_name if role == AgentRole.MASTER_STRATEGIST else self.agent_model_name
            return await self._call_model(model_to_use, prompt, response_json=response_json)
        finally:
            latency = time.time() - start_time
            self.response_tracker.record(role.value, latency)

    async def _call_model(self, model_name: str, prompt: str, use_tools: bool = False, response_json: bool = False) -> str:
        """Helper to call Gemini with retries."""
        retries = 3
        base_delay = 5

        config_args = {
            "temperature": 0.0 if use_tools else 0.1,
            "safety_settings": self.safety_settings
        }

        if use_tools:
            config_args["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        if response_json:
            config_args["response_mime_type"] = "application/json"

        for attempt in range(retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_args)
                )
                return response.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                logger.error(f"Model call failed: {e}")
                raise e
        return ""

    async def _gather_grounded_data(self, search_instruction: str, persona_key: str) -> GroundedDataPacket:
        """
        Phase 1: Use Gemini with Google Search to gather current data.
        This method ALWAYS uses tools, regardless of heterogeneous routing settings.
        Includes 10-minute caching and Sentinel fallback.
        """
        # 1. Check Cache
        now = datetime.now(timezone.utc)
        if search_instruction in self._search_cache:
            entry = self._search_cache[search_instruction]
            if (now - entry['timestamp']).total_seconds() < 600:
                logger.info(f"[{persona_key}] Using cached grounded data.")
                return entry['packet']

        # 2. Craft Data-Gathering Prompt
        gathering_prompt = f"""You are a Research Data Gatherer. Your ONLY job is to find CURRENT information.

TASK: {search_instruction}

INSTRUCTIONS:
1. USE GOOGLE SEARCH to find information from the LAST 24-48 HOURS.
2. Focus on: dates, numbers, quotes from officials, specific events.
3. DO NOT ANALYZE OR GIVE OPINIONS. Just report what you find.
4. IF YOU CANNOT FIND NEWS from the last 72 hours, explicitly state 'NO RECENT NEWS FOUND'.

OUTPUT FORMAT (JSON):
{{
    "raw_summary": "Brief summary of what you found (2-3 paragraphs)",
    "dated_facts": [
        {{"date": "YYYY-MM-DD or 'today'/'yesterday'", "fact": "specific finding", "source": "publication name"}}
    ],
    "search_queries_used": ["query1", "query2"],
    "data_quality": "HIGH|MEDIUM|LOW",
    "data_freshness": "Data appears to be from [timeframe]"
}}
"""

        try:
            # Rate-limited Gemini call to prevent 429 errors
            # === FIX A2: Retry wrapper for transient 503 timeouts ===
            max_attempts = 2
            last_error = None
            response = None

            async with self._grounded_data_semaphore:
                for attempt in range(max_attempts):
                    try:
                        # Enforce minimum interval between calls
                        import time as _time
                        elapsed = _time.monotonic() - self._last_grounded_call_time
                        if elapsed < self._grounded_data_min_interval:
                            await asyncio.sleep(self._grounded_data_min_interval - elapsed)

                        response = await self._call_model(
                            self.master_model_name,
                            gathering_prompt,
                            use_tools=True,
                            response_json=True
                        )
                        self._last_grounded_call_time = _time.monotonic()
                        last_error = None
                        break  # Success — exit retry loop

                    except Exception as e:
                        last_error = e
                        self._last_grounded_call_time = _time.monotonic() # Update time even on fail
                        if '503' in str(e) and attempt < max_attempts - 1:
                            backoff_seconds = 5 * (attempt + 1)  # 5s first retry
                            logger.warning(
                                f"[{persona_key}] Gemini 503 (attempt {attempt + 1}/{max_attempts}) "
                                f"— retrying after {backoff_seconds}s backoff..."
                            )
                            await asyncio.sleep(backoff_seconds)
                            continue
                        else:
                            break  # Non-503 error or final attempt

                if last_error:
                    raise last_error

            # Parse response
            data = json.loads(self._clean_json_text(response))

            # Type guard: LLM may return a JSON array instead of object
            if isinstance(data, list):
                data = {
                    'raw_summary': str(data),
                    'dated_facts': [item for item in data if isinstance(item, dict)],
                    'data_freshness': '',
                    'search_queries_used': []
                }
            elif not isinstance(data, dict):
                data = {'raw_summary': str(data), 'dated_facts': [], 'data_freshness': '', 'search_queries_used': []}

            # === COMPUTE ACTUAL DATA FRESHNESS ===
            freshness_hours = self._compute_data_freshness(
                data.get('dated_facts', []),
                data.get('data_freshness', '')
            )

            packet = GroundedDataPacket(
                search_query=search_instruction,
                raw_findings=data.get('raw_summary', ''),
                extracted_facts=data.get('dated_facts', []),
                source_urls=data.get('search_queries_used', []),
                data_freshness_hours=freshness_hours
            )

            # Check for freshness/empty results
            if "NO RECENT NEWS FOUND" in packet.raw_findings or not packet.extracted_facts:
                 logger.warning(f"[{persona_key}] Grounded search yielded no recent results.")

            # Update Cache
            self._search_cache[search_instruction] = {'timestamp': now, 'packet': packet}

            # Log Provenance
            provenance_logger.info(f"QUERY: {search_instruction} | FACTS: {len(packet.extracted_facts)}")

            return packet

        except Exception as e:
            logger.error(f"Grounded data gathering failed: {e}")

            # FALLBACK: Inject Sentinel History
            history = StateManager.load_state("sentinel_history")
            fallback_events = []
            if 'events' in history:
                data = history['events']
                if isinstance(data, str) and data.startswith("STALE"):
                     fallback_events = [data]
                elif isinstance(data, list):
                    fallback_events = data

            fallback_text = "\n".join([str(evt) for evt in fallback_events])

            packet = GroundedDataPacket(
                search_query=search_instruction,
                raw_findings=f"SEARCH FAILED. FALLBACK DATA (SENTINEL ALERTS):\n{fallback_text}",
                data_freshness_hours=999
            )
            provenance_logger.warning(f"QUERY: {search_instruction} | FALLBACK TRIGGERED")
            return packet

    async def _apply_reflexion(self, agent_name: str, base_prompt: str,
                                contract_name: str = "") -> str:
        """
        Reflexion: Query TMS for past mistakes relevant to current analysis.
        Returns enhanced prompt with past lessons injected.

        Fail-safe: Returns base_prompt unchanged if TMS unavailable or query fails.
        """
        try:
            if not self.tms or not self.tms.collection:
                return base_prompt

            # Query TMS for similar past situations where this agent was wrong
            query = f"{agent_name} mistake prediction wrong {contract_name}"
            results = self.tms.collection.query(
                query_texts=[query],
                n_results=3,
                where={"agent": agent_name} if agent_name else None,
            )

            if not results or not results.get('documents') or not results['documents'][0]:
                return base_prompt

            lessons = results['documents'][0]

            reflexion_block = (
                "\n\n--- SELF-CRITIQUE (from your past mistakes) ---\n"
                "Review these lessons from previous analyses where you were wrong:\n"
            )
            for i, lesson in enumerate(lessons[:3], 1):
                reflexion_block += f"{i}. {lesson[:300]}\n"
            reflexion_block += (
                "Incorporate these lessons. Avoid repeating the same errors.\n"
                "--- END SELF-CRITIQUE ---\n"
            )

            return base_prompt + reflexion_block

        except Exception as e:
            logger.warning(f"Reflexion query failed for {agent_name}: {e}")
            return base_prompt  # Fail-safe: no reflexion is always better than crashing

    async def research_topic(self, persona_key: str, search_instruction: str, regime_context: str = "") -> str:
        """
        Conducts deep-dive research using a specialized agent persona.

        ARCHITECTURE:
        - Phase 1: Gemini with Google Search gathers GROUNDED current data
        - Phase 2: Routed model analyzes the grounded data with diverse perspective
        """

        # Retrieve relevant context from TMS
        # Retrieve ONLY recent insights (last 14 days) to prevent temporal confusion
        # This prevents agents from citing old frost/weather events as current
        relevant_context = self.tms.retrieve(
            search_instruction,
            n_results=3,
            max_age_days=14,
            decay_rates=self._tms_decay_rates
        )
        if relevant_context:
            logger.info(f"Retrieved {len(relevant_context)} TMS insights for {persona_key}.")

        if relevant_context:
            # Add temporal warning to prevent confusion
            context_str = (
                "PRIOR INSIGHTS (from last 14 days only - do NOT cite older events):\n"
                + "\n".join(relevant_context)
            )
        else:
            context_str = "No recent prior context available."

        # === PHASE 1: GROUNDED DATA GATHERING ===
        logger.info(f"[{persona_key}] Phase 1: Gathering grounded data...")
        grounded_data = await self._gather_grounded_data(search_instruction, persona_key)

        if "FAILED" in grounded_data.raw_findings:
            logger.warning(f"[{persona_key}] Data gathering failed, using fallback data.")

        # Check quarantine status
        if self.observability and not self.observability.is_agent_valid(persona_key):
            logger.warning(f"Agent {persona_key} is quarantined! Using fallback/skipping.")
            # We could return a failure here, but for now we proceed with a warning or maybe set a flag?
            # Ideally we should fallback to another model or return error.
            # Failing open for now as per "Observability Must Never Kill Trading" principle,
            # but logging CRITICAL warning.

        # === PHASE 2: HETEROGENEOUS ANALYSIS ===
        persona_prompt = self.personas.get(persona_key, "You are a helpful research assistant.")

        # ENHANCED: Try commodity-aware prompt first, fall back to legacy
        if self.commodity_profile and _HAS_COMMODITY_PROFILES:
            try:
                # Use commodity-specific prompt template
                persona_prompt = get_agent_prompt(persona_key, self.commodity_profile, regime_context=regime_context)
                logger.debug(f"Using commodity-aware prompt for {persona_key}")
            except (ValueError, KeyError) as e:
                # Fall back to legacy persona prompt (no change in behavior)
                logger.debug(f"No commodity template for {persona_key}, using legacy: {e}")
                pass  # persona_prompt already set above

        # DSPy optimized prompt override (per-commodity toggle)
        commodity_ticker = self.full_config.get("symbol", "KC")
        dspy_config = self.full_config.get("dspy", {})
        enabled_map = dspy_config.get("use_optimized_prompts", {})
        if enabled_map.get(commodity_ticker, False):
            optimized = self._load_optimized_prompt(commodity_ticker, persona_key)
            if optimized:
                persona_prompt = optimized["instruction"]
                if optimized.get("demos"):
                    demo_text = self._format_demos(optimized["demos"])
                    persona_prompt = f"{persona_prompt}\n\nEXAMPLES OF GOOD ANALYSIS:\n{demo_text}"
                logger.debug(f"[{persona_key}] Using DSPy-optimized prompt ({len(optimized.get('demos', []))} demos)")

        # Inject grounded data into prompt
        full_prompt = (
            f"{persona_prompt}\n\n"
            f"PREVIOUS INSIGHTS (Do not repeat if still valid):\n{context_str}\n\n"
            f"{grounded_data.to_context_block()}\n\n"
            f"YOUR TASK: Analyze the GROUNDED DATA above and provide your expert assessment regarding: {search_instruction}\n"
            f"CRITICAL: Base your analysis ONLY on the data provided above. Do NOT hallucinate or invent "
            f"information not present in the Grounded Data Packet. If the data is insufficient, say so.\n\n"
            f"OUTPUT FORMAT (JSON ONLY):\n"
            f"{{ \n"
            f"  'evidence': 'Cite 3-5 specific data points from the Grounded Data Packet above.',\n"
            f"  'analysis': 'Explain what this data implies for Coffee supply/demand.',\n"
            f"  'confidence': 0.0-1.0 (Float based on data quality),\n"
            f"  'sentiment': 'BULLISH'|'BEARISH'|'NEUTRAL'\n"
            f"}}"
        )

        # Apply Reflexion: inject past mistake lessons (fail-safe, zero API cost)
        full_prompt = await self._apply_reflexion(persona_key, full_prompt)

        # Determine Role for routing
        role_map = {
            'agronomist': AgentRole.AGRONOMIST,
            'macro': AgentRole.MACRO_ANALYST,
            'geopolitical': AgentRole.GEOPOLITICAL_ANALYST,
            'sentiment': AgentRole.SENTIMENT_ANALYST,
            'technical': AgentRole.TECHNICAL_ANALYST,
            'volatility': AgentRole.VOLATILITY_ANALYST,
            'inventory': AgentRole.INVENTORY_ANALYST,
            'supply_chain': AgentRole.SUPPLY_CHAIN_ANALYST,
        }
        role = role_map.get(persona_key, AgentRole.AGRONOMIST)

        try:
            # Phase 2 always uses heterogeneous routing for diverse analysis if enabled
            if self.use_heterogeneous:
                result_raw = await self.heterogeneous_router.route(role, full_prompt, response_json=True)
            else:
                # If no heterogeneous, fallback to agent model (Gemini) without tools
                result_raw = await self._call_model(self.agent_model_name, full_prompt, use_tools=False, response_json=True)

            # Validate Output (New 3-tuple Unpacking)
            is_valid, issues, conf_adj = self._validate_agent_output(
                persona_key, result_raw, grounded_data.raw_findings
            )
            if not is_valid:
                logger.warning(f"[{persona_key}] Validation issues: {issues}")

            # Parse JSON
            try:
                data = json.loads(self._clean_json_text(result_raw))
                if not isinstance(data, dict):
                    raise ValueError("LLM returned non-dict JSON")
                # === A1-R1: Canonicalize confidence IMMEDIATELY after JSON parse ===
                # Phase 2 LLMs may return band strings ('HIGH', 'MODERATE').
                # Canonicalize here so ALL downstream code — including formatted_text,
                # conf_adj, structured_result, traces — receives a float.
                data['confidence'] = parse_confidence(data.get('confidence', 0.5))

                # Apply Confidence Correction (use min to never inflate)
                if conf_adj is not None:
                    original_conf = data['confidence']  # Already a float from above
                    adjusted_conf = min(original_conf, conf_adj)
                    logger.info(
                        f"[{persona_key}] Confidence clamped: "
                        f"{original_conf:.2f} → {adjusted_conf:.2f} (adjustment: {conf_adj:.2f})"
                    )
                    data['confidence'] = adjusted_conf
            except json.JSONDecodeError:
                # Fallback to raw text if JSON fails
                data = {
                    'evidence': 'N/A',
                    'analysis': result_raw,
                    'confidence': 0.5,
                    'sentiment': 'NEUTRAL'
                }

            # Construct structured report object
            # We format 'data' as the readable text for compatibility
            formatted_text = (
                f"[EVIDENCE]: {data.get('evidence')}\n"
                f"[ANALYSIS]: {data.get('analysis')}\n"
                f"[CONFIDENCE]: {data.get('confidence')}\n"
                f"[SENTIMENT TAG]: [SENTIMENT: {data.get('sentiment', 'NEUTRAL')}]"
            )

            structured_result = {
                'data': formatted_text,
                'confidence': parse_confidence(data.get('confidence', 0.5)),
                'sentiment': data.get('sentiment', 'NEUTRAL'),
                'data_freshness_hours': grounded_data.data_freshness_hours
            }

            # Record Trace
            if self.observability:
                try:
                    trace = AgentTrace(
                        agent=persona_key,
                        timestamp=datetime.now(timezone.utc),
                        query=search_instruction,
                        retrieved_documents=relevant_context,
                        grounded_data=grounded_data.raw_findings, # Fix B3: Populate grounded data
                        output_text=formatted_text,
                        sentiment=data.get('sentiment', 'NEUTRAL'),
                        confidence=parse_confidence(data.get('confidence', 0.5)),
                        model_name=self.agent_model_name # Approximation, router might differ
                    )
                    self.observability.record_trace(trace)
                except Exception as e:
                    logger.error(f"Failed to record agent trace (non-fatal): {e}")

            # Store new insight in TMS
            if formatted_text and len(formatted_text) > 50:
                self.tms.encode(persona_key, formatted_text, {
                    "task": search_instruction,
                    "grounded": True,
                    "data_gathered_at": grounded_data.gathered_at.isoformat()
                })

            return structured_result
        except CriticalRPCError:
            raise  # Let executor shutdown bubble up
        except Exception as e:
            return {'data': f"Error conducting research: {str(e)}", 'confidence': 0.0, 'sentiment': 'NEUTRAL'}

    async def research_topic_with_reflexion(self, persona_key: str, search_instruction: str, regime_context: str = "") -> dict:
        """Research with self-correction loop."""

        # Step 1: Initial analysis
        initial_response_struct = await self.research_topic(persona_key, search_instruction, regime_context=regime_context)
        initial_text = initial_response_struct.get('data', '')

        # Step 2: Reflexion prompt
        reflexion_prompt = f"""You previously wrote this analysis:

{initial_text}

TASK: Critique your own analysis.
1. Are there logical fallacies? (Confirmation bias, recency bias, false causation?)
2. Did you cite specific sources, or did you make vague claims?
3. Is your sentiment tag justified by the evidence?

REVISED ANALYSIS: Provide an improved version that addresses any weaknesses.
OUTPUT FORMAT (JSON ONLY):
{{
  'evidence': '...',
  'analysis': '...',
  'confidence': 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME',
  'sentiment': 'BULLISH'|'BEARISH'|'NEUTRAL'
}}
"""
        role_map = {
            'agronomist': AgentRole.AGRONOMIST,
            'macro': AgentRole.MACRO_ANALYST,
            'geopolitical': AgentRole.GEOPOLITICAL_ANALYST,
            'sentiment': AgentRole.SENTIMENT_ANALYST,
            'technical': AgentRole.TECHNICAL_ANALYST,
            'volatility': AgentRole.VOLATILITY_ANALYST,
            'inventory': AgentRole.INVENTORY_ANALYST,
            'supply_chain': AgentRole.SUPPLY_CHAIN_ANALYST,
        }
        role = role_map.get(persona_key, AgentRole.AGRONOMIST)

        revised_raw = await self._route_call(role, reflexion_prompt, response_json=True)

        try:
            data = json.loads(self._clean_json_text(revised_raw))
            if not isinstance(data, dict):
                raise ValueError("LLM returned non-dict JSON")
            # === A1-R1: Canonicalize confidence immediately (reflexion path) ===
            data['confidence'] = parse_confidence(data.get('confidence', 0.5))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
             return initial_response_struct # Fallback to initial

        formatted_text = (
            f"[EVIDENCE]: {data.get('evidence')}\n"
            f"[ANALYSIS]: {data.get('analysis')}\n"
            f"[CONFIDENCE]: {data.get('confidence')}\n"
            f"[SENTIMENT TAG]: [SENTIMENT: {data.get('sentiment', 'NEUTRAL')}]"
        )

        return {
            'data': formatted_text,
            'confidence': parse_confidence(data.get('confidence', 0.5)),
            'sentiment': data.get('sentiment', 'NEUTRAL')
        }

    async def _get_red_team_analysis(self, persona_key: str, reports_text: str, market_data: dict, opponent_argument: str = None, market_context: str = "") -> str:
        """Gets critique/defense from Permabear or Permabull."""
        persona_prompt = self.personas.get(persona_key, "")

        context_prompt = ""
        if opponent_argument:
            context_prompt = f"\n\n--- OPPONENT ARGUMENT ---\n{opponent_argument}\n\nTASK: The Bear has argued X. You must explicitly refute this point with evidence."

        market_data_str = format_market_context_for_prompt(market_data)

        # Inject market context (includes sentinel briefing if emergency cycle)
        market_context_section = ""
        if market_context:
            market_context_section = f"\n--- MARKET CONTEXT ---\n{market_context}\n"

        prompt = (
            f"{persona_prompt}\n\n"
            f"--- DESK REPORTS ---\n{reports_text}\n\n"
            f"--- MARKET DATA ---\n{market_data_str}\n"
            f"{market_context_section}"
            f"{context_prompt}\n\n"
            f"Generate your response following the specified format:\n"
            f"{DEBATE_OUTPUT_SCHEMA}"
        )
        try:
            # Enforce JSON output for structured debate
            role = AgentRole.PERMABEAR if persona_key == 'permabear' else AgentRole.PERMABULL
            return await self._route_call(role, prompt, response_json=True)
        except Exception as e:
            return json.dumps({"error": str(e), "position": "NEUTRAL", "key_arguments": []})

    async def run_debate(self, reports_text: str, market_data: dict, market_context: str = "") -> tuple[str, str]:
        """
        Run sequential attack-defense debate.
        Step 1: PERMABEAR generates the Thesis (Attack).
        Step 2: PERMABULL must refute specific points (Defense).
        """

        # Step 1: Bear attacks the thesis (with market context including sentinel briefing)
        bear_json = await self._get_red_team_analysis("permabear", reports_text, market_data, market_context=market_context)

        # Step 2: Bull must RESPOND to the specific critique
        bull_json = await self._get_red_team_analysis("permabull", reports_text, market_data, opponent_argument=bear_json, market_context=market_context)

        return bear_json, bull_json

    async def run_devils_advocate(self, decision: dict, reports_text: str, market_context: str) -> dict:
        """
        Run Pre-Mortem analysis on a tentative decision.
        Returns: {'proceed': bool, 'risks': list, 'recommendation': str}

        CRITICAL: This function is FAIL-SAFE. If anything goes wrong,
        it returns proceed=False to block the trade.
        """
        if decision.get('direction') == 'NEUTRAL':
            return {'proceed': True, 'risks': [], 'recommendation': 'No trade proposed'}

        prompt = f"""You are the Devil's Advocate. Your job is to ATTACK this trade.

PROPOSED TRADE: {decision.get('direction')} {self.commodity_profile.name if self.commodity_profile else 'Futures'}
CONFIDENCE: {decision.get('confidence')}
REASONING: {decision.get('reasoning')}

MARKET CONTEXT:
{market_context}

AGENT REPORTS:
{reports_text}

TASK: Run a PRE-MORTEM analysis.
Assume this trade has FAILED CATASTROPHICALLY 5 days from now.
Write a report explaining:
1. What went wrong? (3 specific scenarios)
2. What bias did we succumb to? (Recency? Confirmation? Crowded trade?)
3. What data did we ignore or misinterpret?
4. VERDICT: Should we proceed? (YES/NO)

OUTPUT: JSON with 'proceed' (bool), 'risks' (list of strings), 'recommendation' (string)
"""

        # Retry Loop for JSON Parsing
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = await self._route_call(AgentRole.PERMABEAR, prompt, response_json=True)
                cleaned = self._clean_json_text(response)
                if not cleaned:
                    raise ValueError("Empty response")

                result = json.loads(cleaned)
                if not isinstance(result, dict):
                    raise ValueError("DA returned non-dict JSON")

                # Validate required fields
                if 'proceed' not in result:
                    logger.warning("DA response missing 'proceed' field - defaulting to BLOCK")
                    result['proceed'] = False
                    result['risks'] = result.get('risks', []) + ['Missing proceed field']
                    result['recommendation'] = result.get('recommendation', 'BLOCK: Malformed DA response')
                elif not isinstance(result['proceed'], bool):
                    # LLM returned "proceed": "no" or "proceed": "yes" as string.
                    # String "no" is truthy in Python → would bypass veto. Coerce strictly.
                    raw = result['proceed']
                    result['proceed'] = raw is True or (isinstance(raw, str) and raw.lower() in ('true', 'yes'))
                    logger.warning(f"DA 'proceed' was {type(raw).__name__} ({raw!r}), coerced to {result['proceed']}")

                return result

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"DA Parse Fail (Attempt {attempt+1}): {e}. Retrying...")
                    prompt += "\n\nOUTPUT VALID JSON ONLY. No markdown headers, no explanation, ONLY a JSON object."
                    continue

                # === FINAL ATTEMPT: Text extraction fallback ===
                logger.warning(f"DA JSON parse failed after {max_retries + 1} attempts. "
                              f"Attempting text-based extraction from raw response.")

                try:
                    raw_text = cleaned if 'cleaned' in locals() and cleaned else str(response)
                    raw_lower = raw_text.lower()

                    # Extract verdict from text
                    if any(kw in raw_lower for kw in ['verdict: no', 'should not proceed',
                                                       'do not proceed', 'veto', 'block']):
                        proceed = False
                    elif any(kw in raw_lower for kw in ['verdict: yes', 'should proceed',
                                                         'proceed with', 'approved']):
                        proceed = True
                    else:
                        # Ambiguous — default to BLOCK (fail-closed for parse errors)
                        # Rationale: The DA is an adversarial review gate. If we can't
                        # parse the verdict, the safe default is to block the trade.
                        proceed = False
                        logger.warning("DA text ambiguous — defaulting to BLOCK (fail-closed)")

                    # Extract risks via regex
                    import re
                    risks = re.findall(r'(?:risk|scenario|concern)\s*\d*\s*[:\-]\s*(.+?)(?:\n|$)',
                                      raw_text, re.IGNORECASE)
                    risks = risks[:5] if risks else ['DA response unparseable — text extraction used']

                    return {
                        'proceed': proceed,
                        'risks': risks,
                        'recommendation': f"{'PROCEED' if proceed else 'BLOCK'}: Text extraction fallback",
                        'da_extraction_method': 'text_fallback',
                        'da_bypassed': True  # R3: Signal to downstream components
                    }

                except Exception as fallback_err:
                    logger.error(f"DA text fallback also failed: {fallback_err}")
                    # Ultimate fallback: BLOCK (fail-closed)
                    return {
                        'proceed': False,
                        'risks': ['DA system completely failed — defaulting to block'],
                        'recommendation': 'BLOCK: DA system failure (fail-closed)',
                        'da_block_reason': 'TOTAL_FAILURE_FAILCLOSED',
                        'da_bypassed': True  # R3: Signal to downstream components
                    }

    async def decide(self, contract_name: str, market_data: dict, research_reports: dict, market_context: str, trigger_reason: str = None, cycle_id: str = "", trigger: SentinelTrigger = None) -> dict:
        """
        The Hegelian Loop: Thesis (Reports) -> Antithesis (Bear/Bull) -> Synthesis (Master).
        """
        # 1. Thesis: Format Research Reports
        # Handle dictionary reports with timestamps (from StateManager) vs strings (legacy/fresh)
        reports_text = ""
        for agent, report_obj in research_reports.items():
            report_content = "N/A"
            timestamp_str = ""

            if isinstance(report_obj, dict):
                report_content = report_obj.get('data', 'N/A')
                timestamp_str = report_obj.get('timestamp', '')
            else:
                report_content = str(report_obj)

            # Check for Staleness if timestamp exists
            stale_warning = ""
            # Note: StateManager.load_state() now adds "STALE (...)" prefix directly to data string if using the new method
            # But here we handle the case where we might be reading raw dicts or mixed
            # If report_content already has "STALE", we don't need to add it again.
            if "STALE" not in str(report_content) and timestamp_str:
                 try:
                    ts = datetime.fromisoformat(timestamp_str) if isinstance(timestamp_str, str) else datetime.fromtimestamp(timestamp_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - ts) > timedelta(hours=24):
                        stale_warning = "[WARNING: DATA IS STALE] "
                 except (ValueError, TypeError, OSError):
                    pass

            reports_text += f"\n--- {agent.upper()} REPORT ---\n{stale_warning}{report_content}\n"

        # Build structured sentinel briefing for debate & master injection
        sentinel_briefing = ""
        urgent_context = ""
        if trigger_reason:
            urgent_context = f"*** URGENT TRIGGER: {trigger_reason} ***\n\n"

            if trigger:
                sentinel_briefing = (
                    f"\n\n--- 🚨 SENTINEL EMERGENCY BRIEFING ---\n"
                    f"Source: {getattr(trigger, 'source', 'Unknown')}\n"
                    f"Alert: {trigger_reason}\n"
                    f"Severity: {getattr(trigger, 'severity', 'N/A')}/10\n"
                )
                if hasattr(trigger, 'payload') and trigger.payload:
                    try:
                        sentinel_briefing += f"Raw Payload: {json.dumps(trigger.payload, indent=2, default=str)}\n"
                    except Exception:
                        sentinel_briefing += f"Raw Payload: {str(trigger.payload)[:500]}\n"
                sentinel_briefing += (
                    f"INSTRUCTION: You MUST directly address this specific event in your analysis. "
                    f"Generic market commentary without referencing this trigger is UNACCEPTABLE.\n"
                    f"--- END SENTINEL BRIEFING ---\n"
                )

        # 2. Antithesis: The Dialectical Debate
        # Run sequentially (Bear Attack -> Bull Defense)
        # Inject sentinel briefing so debate agents know WHY this cycle exists
        debate_market_context = market_context + sentinel_briefing if sentinel_briefing else market_context
        bear_json, bull_json = await self.run_debate(reports_text, market_data, market_context=debate_market_context)

        # 3. Synthesis: Master Strategist
        master_persona = self.personas.get('master', "You are the Chief Strategist.")

        market_data_str = format_market_context_for_prompt(market_data)

        full_prompt = (
            f"{master_persona}\n\n"
            f"{urgent_context}"
            f"{sentinel_briefing}"
            f"You have received structured reports regarding {contract_name}.\n"
            f"{market_context}\n\n"
            f"MARKET DATA:\n{market_data_str}\n\n"
            f"--- DESK REPORTS (THESIS) ---\n{reports_text}\n\n"
            f"--- PERMABEAR CRITIQUE (ANTITHESIS) ---\n{bear_json}\n\n"
            f"--- PERMABULL DEFENSE (ANTITHESIS) ---\n{bull_json}\n\n"
            f"SYNTHESIS RULES:\n"
            f"1. If Bear and Bull BOTH acknowledge a risk -> Weight it 2x\n"
            f"2. If agent consensus confidence > 0.85 -> Strong signal, consider action\n"
            f"3. If agent conflict score is high -> Consider volatility play\n\n"
            f"TASK: Synthesize the evidence. Judge the debate. Render a verdict.\n"
            f"CRITICAL: DO NOT output price targets, stop-losses, or precise numerical confidence. Focus on reasoning quality.\n"
            f"OUTPUT FORMAT: Valid JSON object ONLY with these exact keys:\n"
            f"- 'direction': (string) 'BULLISH', 'BEARISH', or 'NEUTRAL'.\n"
            f"- 'thesis_strength': (string) 'SPECULATIVE', 'PLAUSIBLE', or 'PROVEN'.\n"
            f"- 'primary_catalyst': (string) The single most important driver.\n"
            f"- 'reasoning': (string) Full synthesis of evidence and debate.\n"
            f"- 'dissent_acknowledged': (string) Strongest counter-argument and why you override it.\n"
        )

        try:
            response_text = await self._route_call(AgentRole.MASTER_STRATEGIST, full_prompt, response_json=True)
            cleaned_text = self._clean_json_text(response_text)
            decision = json.loads(cleaned_text)
            if not isinstance(decision, dict):
                raise ValueError("Master Strategist returned non-dict JSON")

            # === Direction Validation ===
            # LLM must return a valid direction. If missing or invalid, default to NEUTRAL
            # (safe: prevents accidental trades on malformed output).
            VALID_DIRECTIONS = {'BULLISH', 'BEARISH', 'NEUTRAL'}
            raw_direction = str(decision.get('direction', '')).upper().strip()
            if raw_direction not in VALID_DIRECTIONS:
                logger.warning(
                    f"Master Strategist returned invalid direction '{decision.get('direction')}'. "
                    f"Defaulting to NEUTRAL (fail-safe)."
                )
                decision['direction'] = 'NEUTRAL'
            else:
                decision['direction'] = raw_direction

            # === v7.0: Thesis Strength Validation ===
            # Map thesis_strength to a numeric conviction for downstream compatibility.
            # These are coarse "buckets" — the system doesn't care about the difference
            # between 0.71 and 0.74, which was always noise anyway.
            #
            # NOTE: SPECULATIVE maps to 0.45, which is BELOW the default
            # signal_threshold of 0.5. This is BY DESIGN — SPECULATIVE theses
            # do not warrant risking capital. They are logged for learning
            # (Brier scoring) but not executed.
            #
            # Execution threshold:
            #   PROVEN (0.90)     → Executes ✓
            #   PLAUSIBLE (0.70)  → Executes ✓ (even with 0.5 conviction = 0.35... see note)
            #   SPECULATIVE (0.45) → Blocked ✗
            #
            # IMPORTANT: PLAUSIBLE + DIVERGENT consensus (0.70 * 0.5 = 0.35)
            # will ALSO be blocked. This is correct — low conviction + disagreement
            # should not trade.
            THESIS_TO_CONFIDENCE = {
                'PROVEN': 0.90,
                'PLAUSIBLE': 0.70,
                'SPECULATIVE': 0.45,
            }
            thesis = decision.get('thesis_strength', 'SPECULATIVE').upper()
            if thesis not in THESIS_TO_CONFIDENCE:
                logger.warning(f"Invalid thesis_strength '{thesis}'. Defaulting to SPECULATIVE.")
                thesis = 'SPECULATIVE'
            decision['thesis_strength'] = thesis
            decision['confidence'] = THESIS_TO_CONFIDENCE[thesis]

            # Ensure primary_catalyst exists
            if not decision.get('primary_catalyst'):
                decision['primary_catalyst'] = 'Not specified'

            # Ensure dissent_acknowledged exists
            if not decision.get('dissent_acknowledged'):
                decision['dissent_acknowledged'] = 'None stated'

            logger.info(
                f"Master Decision: {decision.get('direction')} | "
                f"Thesis: {thesis} (conf={decision['confidence']}) | "
                f"Catalyst: {decision.get('primary_catalyst', 'N/A')[:60]}"
            )
            return decision

        except Exception as e:
            logger.error(f"Master Strategist failed: {e}")
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "thesis_strength": "SPECULATIVE",
                "primary_catalyst": f"System Error: {str(e)}",
                "reasoning": f"Master Error: {str(e)}",
                "dissent_acknowledged": "N/A"
            }

    async def run_specialized_cycle(self, trigger: SentinelTrigger, contract_name: str, market_data: dict, market_context: str, ib=None, target_contract=None, cycle_id: str = "", regime_context: str = "") -> dict:
        """
        Runs a Triggered Cycle (Hybrid Decision Model).
        1. Wake up the RELEVANT agent based on Semantic Router.
        2. Load cached state for other agents.
        3. Run the Debate and Master.
        """
        logger.info(f"Running Specialized Cycle triggered by {trigger.source}...")
        
        # Load Cached State with Metadata (Fix #1: Graduated Staleness)
        cached_metadata = StateManager.load_state_with_metadata()
        cached_reports = {}

        for agent, meta in cached_metadata.items():
            # Construct report object with injected age metadata
            # weighted_voting.py will unwrap this to calculate staleness weight
            cached_reports[agent] = {
                'data': meta['data'],
                '_age_hours': meta['age_hours']
            }

        # Semantic Routing
        route = self.semantic_router.route(trigger)
        active_agent_key = route.primary_agent
        logger.info(f"Routing to {active_agent_key} (Reason: {route.reasoning})")

        # Construct specific search instruction based on trigger content
        # v7.1: Commodity-agnostic agent instructions
        from config.commodity_profiles import get_commodity_profile
        from trading_bot.utils import get_active_ticker
        _ticker = get_active_ticker(self.full_config)
        _profile = get_commodity_profile(_ticker)
        _commodity_name = _profile.name  # e.g., "Arabica Coffee", "Cocoa"

        search_instruction = (
            f"Investigate alert from {trigger.source}: {trigger.reason}. "
            f"Details: {trigger.payload}. "
            f"Analyze impact on {_commodity_name} prices."
        )

        # Run the Active Agent
        logger.info(f"Waking up {active_agent_key}...")

        # Use Reflexion for high-ambiguity roles
        reflexion_agents = self.full_config.get('strategy', {}).get('reflexion_agents', ['agronomist', 'macro'])
        if active_agent_key in reflexion_agents:
            fresh_report = await self.research_topic_with_reflexion(active_agent_key, search_instruction, regime_context=regime_context)
        else:
            fresh_report = await self.research_topic(active_agent_key, search_instruction, regime_context=regime_context)

        # Create combined reports for Decision context
        final_reports = cached_reports.copy()

        # Overwrite/Add the fresh report
        final_reports[active_agent_key] = fresh_report

        # --- Handle "Empty Brain" (Cold Start) ---
        expected_agents = [
            "agronomist", "macro", "geopolitical", "supply_chain",
            "inventory", "sentiment", "technical", "volatility"
        ]

        for agent in expected_agents:
            if agent not in final_reports:
                final_reports[agent] = "Data Unavailable (Cold Start)"

        # Save ONLY the fresh update to StateManager
        # StateManager will handle the merge and timestamping
        StateManager.save_state({active_agent_key: fresh_report})

        # === CROSS-CUE ACTIVATION (Fix #5) ===
        # Check if the fresh insight generates cross-cues for other agents
        if isinstance(fresh_report, dict):
            insight_text = fresh_report.get('data', '')
            cued_agents = self.tms.get_cross_cue_agents(active_agent_key, insight_text)

            if cued_agents:
                logger.info(f"Cross-Cue Triggered: {active_agent_key} -> {cued_agents}")

                # Filter cues: only wake up agents that haven't run recently?
                # For now, wake them all up to ensure responsiveness to the new insight.
                # But prevent self-cue (unlikely but safe)
                cues_to_run = [a for a in cued_agents if a != active_agent_key and a in self.personas]

                if cues_to_run:
                    # Construct context-aware prompt for cued agents
                    cue_instruction = (
                        f"{search_instruction}\n"
                        f"NEW INSIGHT from {active_agent_key.upper()}:\n{insight_text}\n"
                        f"TASK: Re-evaluate your domain in light of this new information."
                    )

                    # Run cued agents in parallel (with reflexion if configured)
                    logger.info(f"Waking up cued agents: {cues_to_run}")
                    cued_tasks = {}
                    for agent in cues_to_run:
                        if agent in reflexion_agents:
                            cued_tasks[agent] = self.research_topic_with_reflexion(agent, cue_instruction, regime_context=regime_context)
                        else:
                            cued_tasks[agent] = self.research_topic(agent, cue_instruction, regime_context=regime_context)

                    cued_results = await asyncio.gather(*cued_tasks.values(), return_exceptions=True)

                    # Update reports
                    for agent, res in zip(cued_tasks.keys(), cued_results):
                        if not isinstance(res, Exception):
                            final_reports[agent] = res
                            # Save to state
                            StateManager.save_state({agent: res})
                        else:
                            logger.error(f"Cued agent {agent} failed: {res}")

        # === NEW: Calculate Weighted Vote ===
        trigger_type = determine_trigger_type(trigger.source)
        min_quorum = self.full_config.get('strategy', {}).get('min_voter_quorum', 3)
        weighted_result = await calculate_weighted_decision(
            agent_reports=final_reports,
            trigger_type=trigger_type,
            market_data=market_data,
            ib=ib,
            contract=target_contract,
            min_quorum=min_quorum
        )

        if weighted_result.get('quorum_failure'):
            logger.warning(
                f"Quorum failure for {contract_name}: "
                f"Only {weighted_result.get('voters_present', [])} voted. "
                f"Skipping council for this contract."
            )
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reason": f"Quorum Failure: Insufficient agent participation (Need {min_quorum})",
                "prediction_type": "DIRECTIONAL",
                "quorum_failure": True
            }

        # Inject weighted context for Master
        weighted_context = (
            f"\n\n--- WEIGHTED VOTING RESULT ---\n"
            f"Consensus Direction: {weighted_result['direction']}\n"
            f"Consensus Confidence: {weighted_result['confidence']:.2f}\n"
            f"Weighted Score: {weighted_result['weighted_score']:.3f}\n"
            f"Dominant Agent: {weighted_result['dominant_agent']}\n"
        )

        # Add to market context
        enriched_context = market_context + weighted_context

        # Run Decision Loop with Context Injection
        decision = await self.decide(contract_name, market_data, final_reports, enriched_context, trigger_reason=trigger.reason, cycle_id=cycle_id, trigger=trigger)

        # === v7.0: Consensus Sensor (Emergency Path) ===
        # Master's DIRECTION is trusted. Vote only adjusts CONVICTION (sizing).
        master_dir = decision.get('direction', 'NEUTRAL')
        vote_dir = weighted_result['direction']

        if master_dir == vote_dir or vote_dir == 'NEUTRAL':
            conviction_multiplier = 1.0
        elif vote_dir != master_dir and master_dir != 'NEUTRAL':
            conviction_multiplier = 0.5
            decision['reasoning'] += f" [DIVERGENT CONSENSUS: Vote={vote_dir}, trading smaller]"
            logger.info(f"Emergency consensus divergent: Master={master_dir}, Vote={vote_dir} → half size")
        else:
            conviction_multiplier = 0.75

        decision['confidence'] = round(decision.get('confidence', 0.5) * conviction_multiplier, 2)
        decision['conviction_multiplier'] = conviction_multiplier

        # Inject vote breakdown into decision for dashboard visibility
        decision['vote_breakdown'] = weighted_result.get('vote_breakdown')
        decision['dominant_agent'] = weighted_result.get('dominant_agent')

        # === NEW: Devil's Advocate Check ===
        if decision.get('direction') != 'NEUTRAL' and decision.get('confidence', 0) > 0.5:
            da_review = await self.run_devils_advocate(decision, str(final_reports), enriched_context)

            if not da_review.get('proceed', True):
                logger.warning(f"Devil's Advocate VETOED emergency trade: {da_review.get('recommendation')}")
                decision['direction'] = 'NEUTRAL'
                decision['confidence'] = 0.0
                decision['reasoning'] += f" [DA VETO: {da_review.get('risks', ['Unknown'])[0]}]"
            else:
                logger.info(f"DEVIL'S ADVOCATE CHECK PASSED. Risks identified: {da_review.get('risks', [])}")

            # --- R3: Propagate Bypass Flag ---
            if da_review.get('da_bypassed'):
                decision['da_bypassed'] = True

        # === ENHANCED BRIER TRACKING ===
        # Record probabilistic prediction
        if hasattr(self, 'brier_tracker') and self.brier_tracker:
            try:
                # Extract probabilities from agent outputs (final_reports)
                sentiments = []
                for report in final_reports.values():
                    if isinstance(report, dict):
                        s = report.get('sentiment', 'NEUTRAL')
                        sentiments.append(s)
                    # If string, skip or assume neutral (omitted to avoid noise)

                if sentiments:
                    count = len(sentiments)
                    prob_bullish = sum(1 for s in sentiments if s == 'BULLISH') / count
                    prob_bearish = sum(1 for s in sentiments if s == 'BEARISH') / count
                    prob_neutral = 1.0 - prob_bullish - prob_bearish

                    # Ensure minimum probability to avoid 0.0 log issues if used later
                    prob_bullish = max(0.01, prob_bullish)
                    prob_bearish = max(0.01, prob_bearish)
                    prob_neutral = max(0.01, prob_neutral)

                    # Normalize
                    total = prob_bullish + prob_bearish + prob_neutral
                    prob_bullish /= total
                    prob_bearish /= total
                    prob_neutral /= total

                    self.brier_tracker.record_prediction(
                        agent="council",
                        prob_bullish=prob_bullish,
                        prob_neutral=prob_neutral,
                        prob_bearish=prob_bearish,
                        contract=contract_name,
                        cycle_id=cycle_id  # NEW — must be passed to decide()
                    )
            except Exception as e:
                logger.error(f"Failed to record Brier prediction (non-fatal): {e}")

        return decision

# Backward compatibility alias — safe to remove once all imports are updated
CoffeeCouncil = TradingCouncil
# Refined Diagnostic Fix 1
