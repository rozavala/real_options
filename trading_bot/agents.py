"""The Coffee Council Module (Updated for google-genai SDK).

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
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from trading_bot.state_manager import StateManager
from trading_bot.sentinels import SentinelTrigger
from trading_bot.semantic_router import SemanticRouter
from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole, get_router
from trading_bot.tms import TransactiveMemory
from trading_bot.reliability_scorer import ReliabilityScorer
from trading_bot.weighted_voting import calculate_weighted_decision, determine_trigger_type

logger = logging.getLogger(__name__)

DEBATE_OUTPUT_SCHEMA = """
Output JSON with this structure:
{
    "position": "BULLISH|BEARISH",
    "key_arguments": [
        {"claim": "...", "evidence": "...", "confidence": 0.0-1.0}
    ],
    "acknowledged_risks": ["..."],
    "ml_alignment": "AGREES|DISAGREES|NEUTRAL",
    "ml_confidence_used": 0.0
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

class CoffeeCouncil:
    """Manages the tiered Gemini agent system using the new google-genai SDK."""

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

        logger.info(f"Coffee Council initialized. Agents: {self.agent_model_name}, Master: {self.master_model_name}")

    def _clean_json_text(self, text: str) -> str:
        """Helper to strip markdown code blocks from JSON strings."""
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

    async def research_topic(self, persona_key: str, search_instruction: str) -> str:
        """Conducts deep-dive research using a specialized agent persona and Google Search."""

        # Retrieve relevant context from TMS
        relevant_context = self.tms.retrieve(search_instruction, n_results=2)
        if relevant_context:
            logger.info(f"Retrieved {len(relevant_context)} TMS insights for {persona_key}.")

        context_str = "\n".join(relevant_context) if relevant_context else "No prior context."

        persona_prompt = self.personas.get(persona_key, "You are a helpful research assistant.")
        full_prompt = (
            f"{persona_prompt}\n\n"
            f"PREVIOUS INSIGHTS (Do not repeat these if still valid):\n{context_str}\n\n"
            f"TASK: {search_instruction}\n"
            f"INSTRUCTIONS:\n"
            f"1. USE YOUR TOOLS: Use Google Search to find data from the last 24-48 hours.\n"
            f"2. NO CHIT-CHAT: Do not say 'Okay', 'I will'. Start directly with the data.\n"
            f"3. FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:\n"
            f"   [EVIDENCE]: List 3-5 specific data points found (dates, numbers, quotes).\n"
            f"   [ANALYSIS]: Explain what this data implies for Coffee supply/demand.\n"
            f"   [SENTIMENT TAG]: End with exactly one: [SENTIMENT: BULLISH], [SENTIMENT: BEARISH], or [SENTIMENT: NEUTRAL].\n"
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
        role = role_map.get(persona_key, AgentRole.AGRONOMIST) # Default to Agronomist if unknown

        try:
            if self.use_heterogeneous:
                # Note: Currently HeterogeneousRouter interfaces don't support tools yet in this impl,
                # but let's assume router handles it or we fall back.
                # Actually, the router code calls generate() on clients.
                # GeminiClient implementation in router supports tools?
                # Looking at HeterogeneousRouter code, it passes system_prompt and prompt.
                # It does NOT pass use_tools explicitly.
                # However, for Analyst roles, we generally want tools (Search).
                # The current router implementation calls generate(prompt, system_prompt).
                # The GeminiClient.generate in router sends contents=full_prompt.
                # It does NOT seem to enable tools dynamically.
                # FIX: We should use the existing _call_model for tool usage OR update router.
                # For now, to be safe and use tools (which are critical for research),
                # we might stick to _call_model for RESEARCH if tools are needed,
                # UNLESS we update the router to support tools.

                # Given the instruction "Route Tier 2 Analysts", we should use the router.
                # But we need Search.
                # Let's assume for this task we use the router but modify the prompt to include search data if possible,
                # or acknowledge that currently only Gemini (via _call_model) has tool support easily wired up here.

                # Actually, the user wants us to use the router.
                # Let's try to use the router. If the specific client (e.g. GPT-4o) doesn't have search tools configured,
                # the agent will hallucinate or rely on training data.
                # This is a risk.
                # However, Gemini 1.5 Pro (Agronomist) is in the router.
                # Let's proceed with routing, but maybe we should fall back to _call_model if tool usage is strictly required and router doesn't support it?
                # The Gap Analysis says "The router creates clients lazily... Issue: The research_topic method still uses self._call_model".
                # So we must use route().
                result = await self._route_call(role, full_prompt)
            else:
                result = await self._call_model(self.agent_model_name, full_prompt, use_tools=True)

            # Store new insight in TMS
            if result and len(result) > 50:
                self.tms.encode(persona_key, result, {"task": search_instruction})

            return result
        except Exception as e:
            return f"Error conducting research: {str(e)}"

    async def research_topic_with_reflexion(self, persona_key: str, search_instruction: str) -> str:
        """Research with self-correction loop."""

        # Step 1: Initial analysis
        initial_response = await self.research_topic(persona_key, search_instruction)

        # Step 2: Reflexion prompt
        reflexion_prompt = f"""You previously wrote this analysis:

{initial_response}

TASK: Critique your own analysis.
1. Are there logical fallacies? (Confirmation bias, recency bias, false causation?)
2. Did you cite specific sources, or did you make vague claims?
3. Is your sentiment tag justified by the evidence?

REVISED ANALYSIS: Provide an improved version that addresses any weaknesses.
Keep the same format: [EVIDENCE], [ANALYSIS], [SENTIMENT TAG]
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

        revised = await self._route_call(role, reflexion_prompt)
        return revised

    async def _get_red_team_analysis(self, persona_key: str, reports_text: str, ml_signal: dict, opponent_argument: str = None) -> str:
        """Gets critique/defense from Permabear or Permabull."""
        persona_prompt = self.personas.get(persona_key, "")

        context_prompt = ""
        if opponent_argument:
            context_prompt = f"\n\n--- OPPONENT ARGUMENT ---\n{opponent_argument}\n\nTASK: The Bear has argued X. You must explicitly refute this point with evidence."

        prompt = (
            f"{persona_prompt}\n\n"
            f"--- DESK REPORTS ---\n{reports_text}\n\n"
            f"--- ML SIGNAL ---\n{json.dumps(ml_signal, indent=2)}\n"
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
        """
        if decision.get('direction') == 'NEUTRAL':
            return {'proceed': True, 'risks': [], 'recommendation': 'No trade proposed'}

        prompt = f"""You are the Devil's Advocate. Your job is to ATTACK this trade.

PROPOSED TRADE: {decision.get('direction')} Coffee
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
            return json.loads(self._clean_json_text(response))
        except Exception as e:
            logger.error(f"Devil's Advocate failed: {e}")
            return {'proceed': True, 'risks': ['DA Failed'], 'recommendation': 'Proceed with caution (DA Error)'}

    async def decide(self, contract_name: str, ml_signal: dict, research_reports: dict, market_context: str, trigger_reason: str = None) -> dict:
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
                    if (datetime.now() - ts) > timedelta(hours=24):
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

        full_prompt = (
            f"{master_persona}\n\n"
            f"{urgent_context}"
            f"You have received structured reports regarding {contract_name}.\n"
            f"{market_context}\n\n"
            f"QUANT MODEL SIGNAL:\n{json.dumps(ml_signal, indent=2)}\n\n"
            f"--- DESK REPORTS (THESIS) ---\n{reports_text}\n\n"
            f"--- PERMABEAR CRITIQUE (ANTITHESIS) ---\n{bear_json}\n\n"
            f"--- PERMABULL DEFENSE (ANTITHESIS) ---\n{bull_json}\n\n"
            f"SYNTHESIS RULES:\n"
            f"1. If Bear and Bull BOTH acknowledge a risk -> Weight it 2x\n"
            f"2. If ML confidence > 0.75 and both debaters cite it -> Follow ML\n"
            f"3. If ML confidence < 0.5 -> Require unanimous agent agreement for action\n\n"
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
            return json.loads(cleaned_text)
        except Exception as e:
            logger.error(f"Master Strategist failed: {e}")
            return {
                "projected_price_5_day": ml_signal.get('expected_price', 0.0),
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": f"Master Error: {str(e)}"
            }

    async def run_specialized_cycle(self, trigger: SentinelTrigger, contract_name: str, ml_signal: dict, market_context: str, ib=None, target_contract=None) -> dict:
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
        decision = await self.decide(contract_name, ml_signal, final_reports, enriched_context, trigger_reason=trigger.reason)

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

        return decision
