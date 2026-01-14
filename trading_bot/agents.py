"""The Coffee Council Module (Updated for google-genai SDK).

This module implements the '5-headed beast' agent system using Google Gemini models.
It includes specialized research agents, a Dialectical Debate (Permabear/Permabull),
and a Master Strategist that synthesizes reports to make final trading decisions.
"""

import os
import json
import logging
import asyncio
from google import genai
from google.genai import types
from trading_bot.state_manager import StateManager
from trading_bot.sentinels import SentinelTrigger

logger = logging.getLogger(__name__)

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
        persona_prompt = self.personas.get(persona_key, "You are a helpful research assistant.")
        full_prompt = (
            f"{persona_prompt}\n\n"
            f"TASK: {search_instruction}\n"
            f"INSTRUCTIONS:\n"
            f"1. USE YOUR TOOLS: Use Google Search to find data from the last 24-48 hours.\n"
            f"2. NO CHIT-CHAT: Do not say 'Okay', 'I will'. Start directly with the data.\n"
            f"3. FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:\n"
            f"   [EVIDENCE]: List 3-5 specific data points found (dates, numbers, quotes).\n"
            f"   [ANALYSIS]: Explain what this data implies for Coffee supply/demand.\n"
            f"   [SENTIMENT TAG]: End with exactly one: [SENTIMENT: BULLISH], [SENTIMENT: BEARISH], or [SENTIMENT: NEUTRAL].\n"
        )

        try:
            return await self._call_model(self.agent_model_name, full_prompt, use_tools=True)
        except Exception as e:
            return f"Error conducting research: {str(e)}"

    async def _get_red_team_analysis(self, persona_key: str, reports_text: str, ml_signal: dict) -> str:
        """Gets critique/defense from Permabear or Permabull."""
        persona_prompt = self.personas.get(persona_key, "")
        prompt = (
            f"{persona_prompt}\n\n"
            f"--- DESK REPORTS ---\n{reports_text}\n\n"
            f"--- ML SIGNAL ---\n{json.dumps(ml_signal, indent=2)}\n\n"
            f"Generate your response following the specified format."
        )
        try:
            return await self._call_model(self.agent_model_name, prompt)
        except Exception as e:
            return f"Error: {str(e)}"

    async def decide(self, contract_name: str, ml_signal: dict, research_reports: dict, market_context: str) -> dict:
        """
        The Hegelian Loop: Thesis (Reports) -> Antithesis (Bear/Bull) -> Synthesis (Master).
        """
        # 1. Thesis: Format Research Reports
        reports_text = ""
        for agent, report in research_reports.items():
            reports_text += f"\n--- {agent.upper()} REPORT ---\n{report}\n"

        # 2. Antithesis: The Dialectical Debate
        # Run in parallel
        bear_task = self._get_red_team_analysis("permabear", reports_text, ml_signal)
        bull_task = self._get_red_team_analysis("permabull", reports_text, ml_signal)

        critique, defense = await asyncio.gather(bear_task, bull_task)

        # 3. Synthesis: Master Strategist
        master_persona = self.personas.get('master', "You are the Chief Strategist.")
        full_prompt = (
            f"{master_persona}\n\n"
            f"You have received structured reports regarding {contract_name}.\n"
            f"{market_context}\n\n"
            f"QUANT MODEL SIGNAL:\n{json.dumps(ml_signal, indent=2)}\n\n"
            f"--- DESK REPORTS (THESIS) ---\n{reports_text}\n\n"
            f"--- PERMABEAR CRITIQUE (ANTITHESIS) ---\n{critique}\n\n"
            f"--- PERMABULL DEFENSE (ANTITHESIS) ---\n{defense}\n\n"
            f"TASK: Synthesize the evidence. Judge the debate. Render a verdict.\n"
            f"NEGATIVE CONSTRAINT: DO NOT output specific numerical price projections unless explicitly provided in reports.\n"
            f"OUTPUT FORMAT: Valid JSON object ONLY.\n"
            f"- 'projected_price_5_day': (float) Target.\n"
            f"- 'direction': (string) 'BULLISH', 'BEARISH', or 'NEUTRAL'.\n"
            f"- 'confidence': (float) 0.0-1.0.\n"
            f"- 'reasoning': (string) Concise explanation.\n"
        )

        try:
            response_text = await self._call_model(self.master_model_name, full_prompt, response_json=True)
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

    async def run_specialized_cycle(self, trigger: SentinelTrigger, contract_name: str, ml_signal: dict, market_context: str) -> dict:
        """
        Runs a Triggered Cycle (Hybrid Decision Model).
        1. Wake up the RELEVANT agent based on trigger source.
        2. Load cached state for other agents.
        3. Run the Debate and Master.
        """
        logger.info(f"Running Specialized Cycle triggered by {trigger.source}...")
        
        # Load Cached State
        cached_reports = StateManager.load_state()

        # Determine relevant agent
        active_agent_key = None
        search_instruction = ""

        if trigger.source == "WeatherSentinel":
            active_agent_key = "agronomist"
            search_instruction = f"Investigate recent weather alert: {trigger.reason}. Search for 'coffee crop impact {trigger.payload.get('location')}'. Analyze if this is a yield-destroying event."
        elif trigger.source == "LogisticsSentinel":
            active_agent_key = "supply_chain"
            search_instruction = f"Investigate supply chain alert: {trigger.reason}. Search for specific details on '{trigger.payload.get('headlines', [])}'. Estimate impact on export flow."
        elif trigger.source == "PriceSentinel":
            active_agent_key = "technical"
            search_instruction = f"Investigate price shock: {trigger.reason}. Search for 'coffee futures news last hour'. Identify the catalyst (Algo liquidation? News leak?)."
        else:
            # Fallback for NewsSentinel or others -> maybe wake up everyone?
            # For now, let's just default to Master reviewing existing info + maybe Macro?
            active_agent_key = "macro"
            search_instruction = f"Investigate market sentiment shift: {trigger.reason}."

        # Run the Active Agent
        logger.info(f"Waking up {active_agent_key}...")
        fresh_report = await self.research_topic(active_agent_key, search_instruction)

        # Update Reports (Fresh + Cached)
        final_reports = cached_reports.copy()
        final_reports[active_agent_key] = fresh_report # Overwrite with fresh data

        # --- Handle "Empty Brain" (Cold Start) ---
        expected_agents = [
            "agronomist", "macro", "geopolitical", "supply_chain",
            "inventory", "sentiment", "technical", "volatility"
        ]

        for agent in expected_agents:
            if agent not in final_reports:
                final_reports[agent] = "Data Unavailable (Cold Start)"

        # Note: We should probably update the state file with this fresh report too,
        # so subsequent runs have the latest info.
        StateManager.save_state(final_reports)

        # Run Decision Loop
        return await self.decide(contract_name, ml_signal, final_reports, market_context)
