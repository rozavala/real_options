"""The Coffee Council Module (Updated for google-genai SDK).

This module implements the '5-headed beast' agent system using Google Gemini models.
It includes specialized research agents (Meteorologist, Macro, Geopolitical, Sentiment)
that use Google Search tools to gather real-time intelligence, and a Master Strategist
that synthesizes these reports with quantitative ML signals to make final trading decisions.
"""

import os
import json
import logging
import asyncio
from google import genai
from google.genai import types

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
        self.agent_model_name = self.config.get('agent_model', 'gemini-3.0-flash')
        self.master_model_name = self.config.get('master_model', 'gemini-2.5-pro')

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

    async def research_topic(self, persona_key: str, search_instruction: str) -> str:
        """
        Conducts deep-dive research using a specialized agent persona and Google Search.

        Args:
            persona_key: The key for the persona in config (e.g., 'meteorologist').
            search_instruction: The specific research task (e.g., "Search for rainfall...").

        Returns:
            A string summary of the research findings.
        """
        persona_prompt = self.personas.get(persona_key, "You are a helpful research assistant.")

        # CHAIN-OF-THOUGHT PROMPT
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
            # NEW SDK SYNTAX for Tools
            # TEMPERATURE 0.0 = DETERMINISTIC (Removes "Saturday Randomness")
            response = await self.client.aio.models.generate_content(
                model=self.agent_model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.0,
                    safety_settings=self.safety_settings
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Agent '{persona_key}' failed: {e}")
            return f"Error conducting research: {str(e)}"

    async def decide(self, contract_name: str, ml_signal: dict, research_reports: dict, market_context: str) -> dict:
        """
        Synthesizes research reports and ML signals to make a final trading decision.

        Args:
            contract_name: Name of the contract (e.g. "KC H25")
            ml_signal: The raw signal dictionary from the ML model.
            research_reports: A dictionary of report strings from the agents.
            market_context: A string containing recent price history context.

        Returns:
            A dictionary containing the final decision (JSON parsed).
        """
        master_persona = self.personas.get('master', "You are the Chief Strategist.")

        reports_text = ""
        for agent, report in research_reports.items():
            reports_text += f"\n--- {agent.upper()} REPORT ---\n{report}\n"

        full_prompt = (
            f"{master_persona}\n\n"
            f"You have received structured reports regarding {contract_name}.\n"
            f"{market_context}\n\n"
            f"QUANT MODEL SIGNAL:\n{json.dumps(ml_signal, indent=2)}\n\n"
            f"--- DESK REPORTS ---\n{reports_text}\n\n"
            f"TASK: Synthesize the evidence. Apply 'Second-Level Thinking':\n"
            f"1. If news is Bullish but '24h Change' is already UP, is it priced in?\n"
            f"2. If news is Bullish but '24h Change' is DOWN, is the market ignoring it (Divergence)?\n"
            f"OUTPUT FORMAT: You must output a valid JSON object ONLY, with no markdown formatting. "
            f"The JSON must have these keys:\n"
            f"- 'projected_price_5_day': (float) Your price target.\n"
            f"- 'direction': (string) 'BULLISH', 'BEARISH', or 'NEUTRAL'.\n"
            f"- 'confidence': (float) A value between 0.0 and 1.0.\n"
            f"- 'reasoning': (string) A concise explanation of your decision (max 2 sentences).\n"
        )

        try:
            # NEW SDK SYNTAX (No tools for Master)
            response = await self.client.aio.models.generate_content(
                model=self.master_model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", # Native JSON Enforcement
                    temperature=0.1,
                    safety_settings=self.safety_settings
                )
            )

            # The new SDK handles JSON parsing cleaner if mime_type is set
            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Master Strategist failed: {e}")
            # Fallback
            return {
                "projected_price_5_day": ml_signal.get('expected_price', 0.0),
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": f"Master Error: {str(e)}"
            }

    async def audit_decision(self, contract_name: str, research_reports: dict, decision: dict) -> dict:
        """
        Uses a separate 'Compliance' agent to verify the Master's decision is supported by the evidence.
        """
        compliance_persona = self.personas.get('compliance', "You are a Compliance Officer.")

        # Format the evidence
        evidence_text = ""
        for agent, report in research_reports.items():
            evidence_text += f"[{agent.upper()} REPORT]: {report}\n"

        full_prompt = (
            f"{compliance_persona}\n"
            f"--- EVIDENCE ---\n{evidence_text}\n"
            f"--- DECISION TO AUDIT ---\n{json.dumps(decision)}\n\n"
            f"TASK: Verify that the 'reasoning' in the decision is strictly supported by the EVIDENCE. "
            f"Check for hallucinations (claims not in evidence). "
            f"OUTPUT: JSON {{'approved': true/false, 'flagged_reason': '...'}}"
        )

        try:
            # Use the Agent Model (Flash) for speed/cost - it's good at verification
            response = await self.client.aio.models.generate_content(
                model=self.agent_model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Compliance Audit failed: {e}")
            # FAIL CLOSED: Block the trade if the auditor fails
            return {
                "approved": False,
                "flagged_reason": f"AUDIT SYSTEM FAILURE: {str(e)}"
            }
