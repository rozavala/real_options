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
from trading_bot.state_manager import StateManager
from trading_bot.sentinels import SentinelTrigger
from trading_bot.semantic_router import SemanticRouter
from trading_bot.market_data_provider import format_market_context_for_prompt
from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole, get_router
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
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, 'research_provenance.log'))
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
    data_freshness_hours: int = 48  # How recent the data claims to be

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
        Initializes the Coffee Council with configuration.

        Args:
            config: The full application configuration dictionary.
        """
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

        commodity_label = self.commodity_profile.name if self.commodity_profile else "General"
        logger.info(f"Trading Council initialized for {commodity_label}. Agents: {self.agent_model_name}, Master: {self.master_model_name}")

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
            # Force Gemini with tools - bypass router for data gathering
            response = await self._call_model(
                self.master_model_name,
                gathering_prompt,
                use_tools=True,
                response_json=True
            )

            # Parse response
            data = json.loads(self._clean_json_text(response))

            packet = GroundedDataPacket(
                search_query=search_instruction,
                raw_findings=data.get('raw_summary', ''),
                extracted_facts=data.get('dated_facts', []),
                source_urls=data.get('search_queries_used', []),
                data_freshness_hours=48 if 'today' in str(data.get('data_freshness', '')).lower() else 72
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

    async def research_topic(self, persona_key: str, search_instruction: str) -> str:
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
            max_age_days=14
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
                persona_prompt = get_agent_prompt(persona_key, self.commodity_profile)
                logger.debug(f"Using commodity-aware prompt for {persona_key}")
            except (ValueError, KeyError) as e:
                # Fall back to legacy persona prompt (no change in behavior)
                logger.debug(f"No commodity template for {persona_key}, using legacy: {e}")
                pass  # persona_prompt already set above

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

            # Parse JSON
            try:
                data = json.loads(self._clean_json_text(result_raw))
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
                'confidence': float(data.get('confidence', 0.5)),
                'sentiment': data.get('sentiment', 'NEUTRAL')
            }

            # Record Trace
            if self.observability:
                try:
                    trace = AgentTrace(
                        agent=persona_key,
                        timestamp=datetime.now(timezone.utc),
                        query=search_instruction,
                        retrieved_documents=relevant_context,
                        output_text=formatted_text,
                        sentiment=data.get('sentiment', 'NEUTRAL'),
                        confidence=float(data.get('confidence', 0.5)),
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
        except Exception as e:
            return {'data': f"Error conducting research: {str(e)}", 'confidence': 0.0, 'sentiment': 'NEUTRAL'}

    async def research_topic_with_reflexion(self, persona_key: str, search_instruction: str) -> dict:
        """Research with self-correction loop."""

        # Step 1: Initial analysis
        initial_response_struct = await self.research_topic(persona_key, search_instruction)
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
  'confidence': 0.0-1.0,
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
        except:
             return initial_response_struct # Fallback to initial

        formatted_text = (
            f"[EVIDENCE]: {data.get('evidence')}\n"
            f"[ANALYSIS]: {data.get('analysis')}\n"
            f"[CONFIDENCE]: {data.get('confidence')}\n"
            f"[SENTIMENT TAG]: [SENTIMENT: {data.get('sentiment', 'NEUTRAL')}]"
        )

        return {
            'data': formatted_text,
            'confidence': float(data.get('confidence', 0.5)),
            'sentiment': data.get('sentiment', 'NEUTRAL')
        }

    async def _get_red_team_analysis(self, persona_key: str, reports_text: str, ml_signal: dict, opponent_argument: str = None) -> str:
        """Gets critique/defense from Permabear or Permabull."""
        persona_prompt = self.personas.get(persona_key, "")

        context_prompt = ""
        if opponent_argument:
            context_prompt = f"\n\n--- OPPONENT ARGUMENT ---\n{opponent_argument}\n\nTASK: The Bear has argued X. You must explicitly refute this point with evidence."

        market_data_str = format_market_context_for_prompt(ml_signal)

        prompt = (
            f"{persona_prompt}\n\n"
            f"--- DESK REPORTS ---\n{reports_text}\n\n"
            f"--- MARKET DATA ---\n{market_data_str}\n"
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

    async def run_debate(self, reports_text: str, ml_signal: dict) -> tuple[str, str]:
        """
        Run sequential attack-defense debate.
        Step 1: PERMABEAR generates the Thesis (Attack).
        Step 2: PERMABULL must refute specific points (Defense).
        """

        # Step 1: Bear attacks the thesis
        bear_json = await self._get_red_team_analysis("permabear", reports_text, ml_signal)

        # Step 2: Bull must RESPOND to the specific critique
        # We assume bear_json is a JSON string, let's pass it as context
        bull_json = await self._get_red_team_analysis("permabull", reports_text, ml_signal, opponent_argument=bear_json)

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

        try:
            response = await self._route_call(AgentRole.PERMABEAR, prompt, response_json=True)

            # Guard: Empty response check
            if not response or not str(response).strip():
                logger.error("Devil's Advocate returned empty response - BLOCKING TRADE (fail-safe)")
                return {
                    'proceed': False,
                    'risks': ['DA returned empty response - system failure'],
                    'recommendation': 'BLOCK: Devil\'s Advocate system failure. Trade blocked for safety.'
                }

            # Parse response
            cleaned = self._clean_json_text(response)
            result = json.loads(cleaned)

            # Validate required fields
            if 'proceed' not in result:
                logger.warning("DA response missing 'proceed' field - defaulting to BLOCK")
                result['proceed'] = False
                result['risks'] = result.get('risks', []) + ['Missing proceed field']
                result['recommendation'] = result.get('recommendation', 'BLOCK: Malformed DA response')

            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"Devil's Advocate JSON parse failed - BLOCKING TRADE (fail-safe). "
                f"Error: {e}. Response preview: {str(response)[:300] if response else 'None'}"
            )
            return {
                'proceed': False,
                'risks': ['DA JSON parse failure - could not validate trade safety'],
                'recommendation': 'BLOCK: Devil\'s Advocate returned invalid JSON. Trade blocked for safety.'
            }
        except Exception as e:
            logger.error(f"Devil's Advocate system error - BLOCKING TRADE (fail-safe): {e}")
            return {
                'proceed': False,
                'risks': [f'DA system error: {str(e)[:100]}'],
                'recommendation': 'BLOCK: Devil\'s Advocate system error. Trade blocked for safety.'
            }

    async def decide(self, contract_name: str, ml_signal: dict, research_reports: dict, market_context: str, trigger_reason: str = None, cycle_id: str = "") -> dict:
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
                 except:
                    pass

            reports_text += f"\n--- {agent.upper()} REPORT ---\n{stale_warning}{report_content}\n"

        # 2. Antithesis: The Dialectical Debate
        # Run sequentially (Bear Attack -> Bull Defense)
        bear_json, bull_json = await self.run_debate(reports_text, ml_signal)

        # 3. Synthesis: Master Strategist
        master_persona = self.personas.get('master', "You are the Chief Strategist.")

        # Context Injection
        urgent_context = ""
        if trigger_reason:
            urgent_context = f"*** URGENT TRIGGER: {trigger_reason} ***\n\n"

        # Format market data context (replaces ML signal injection)
        market_data_str = format_market_context_for_prompt(ml_signal)

        full_prompt = (
            f"{master_persona}\n\n"
            f"{urgent_context}"
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
            f"NEGATIVE CONSTRAINT: DO NOT output specific numerical price projections unless explicitly provided in reports.\n"
            f"OUTPUT FORMAT: Valid JSON object ONLY.\n"
            f"- 'projected_price_5_day': (float) Target.\n"
            f"- 'direction': (string) 'BULLISH', 'BEARISH', or 'NEUTRAL'.\n"
            f"- 'confidence': (float) 0.0-1.0.\n"
            f"- 'reasoning': (string) Concise explanation.\n"
        )

        try:
            response_text = await self._route_call(AgentRole.MASTER_STRATEGIST, full_prompt, response_json=True)
            cleaned_text = self._clean_json_text(response_text)
            decision = json.loads(cleaned_text)

            # === Confidence Validation Layer ===
            raw_conf = decision.get('confidence', 0.5)
            if not isinstance(raw_conf, (int, float)):
                logger.warning(f"Non-numeric confidence received: {raw_conf}. Defaulting to 0.5")
                raw_conf = 0.5

            decision['confidence'] = max(0.0, min(1.0, float(raw_conf)))
            return decision
        except Exception as e:
            logger.error(f"Master Strategist failed: {e}")
            return {
                "projected_price_5_day": ml_signal.get('expected_price', 0.0),
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": f"Master Error: {str(e)}"
            }

    async def run_specialized_cycle(self, trigger: SentinelTrigger, contract_name: str, ml_signal: dict, market_context: str, ib=None, target_contract=None, cycle_id: str = "") -> dict:
        """
        Runs a Triggered Cycle (Hybrid Decision Model).
        1. Wake up the RELEVANT agent based on Semantic Router.
        2. Load cached state for other agents.
        3. Run the Debate and Master.
        """
        logger.info(f"Running Specialized Cycle triggered by {trigger.source}...")
        
        # Load Cached State (Sync call)
        # Note: load_state returns dict of key->data (str), potentially with "STALE" prefix
        cached_reports = StateManager.load_state()

        # Semantic Routing
        route = self.semantic_router.route(trigger)
        active_agent_key = route.primary_agent
        logger.info(f"Routing to {active_agent_key} (Reason: {route.reasoning})")

        # Construct specific search instruction based on trigger content
        search_instruction = f"Investigate alert from {trigger.source}: {trigger.reason}. Details: {trigger.payload}. Analyze impact on Coffee prices."

        # Run the Active Agent
        logger.info(f"Waking up {active_agent_key}...")

        # Use Reflexion for high-ambiguity roles
        if active_agent_key in ['agronomist', 'macro']:
            fresh_report = await self.research_topic_with_reflexion(active_agent_key, search_instruction)
        else:
            fresh_report = await self.research_topic(active_agent_key, search_instruction)

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

        # === NEW: Calculate Weighted Vote ===
        trigger_type = determine_trigger_type(trigger.source)
        weighted_result = await calculate_weighted_decision(
            agent_reports=final_reports,
            trigger_type=trigger_type,
            ml_signal=ml_signal,
            ib=ib,
            contract=target_contract
        )

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
        decision = await self.decide(contract_name, ml_signal, final_reports, enriched_context, trigger_reason=trigger.reason, cycle_id=cycle_id)

        # === NEW: Weighted Vote Override Logic ===
        master_dir = decision.get('direction', 'NEUTRAL')
        vote_dir = weighted_result['direction']

        if master_dir != vote_dir:
            if weighted_result['confidence'] > 0.85:
                decision['direction'] = vote_dir
                decision['reasoning'] += f" [OVERRIDE: Strong agent consensus ({vote_dir})]"
                logger.info(f"Master override by Weighted Vote: {vote_dir} (Conf: {weighted_result['confidence']})")
            elif master_dir != 'NEUTRAL':
                decision['direction'] = 'NEUTRAL'
                decision['reasoning'] += f" [VETO: Agents disagree ({vote_dir})]"
                logger.info(f"Master decision vetoed by conflicting Weighted Vote ({vote_dir})")

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
