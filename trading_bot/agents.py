"""The Coffee Council Module.

This module implements the '5-headed beast' agent system using Google Gemini models.
It includes specialized research agents (Meteorologist, Macro, Geopolitical, Sentiment)
that use Google Search tools to gather real-time intelligence, and a Master Strategist
that synthesizes these reports with quantitative ML signals to make final trading decisions.
"""

import os
import json
import logging
import asyncio
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class CoffeeCouncil:
    """Manages the tiered Gemini agent system."""

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

        genai.configure(api_key=api_key)

        # 2. Initialize Models
        agent_model_name = self.config.get('agent_model', 'gemini-3.0-flash')
        master_model_name = self.config.get('master_model', 'gemini-2.5-pro')

        # Safety settings (block few)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        # Agent Model (Flash with Search Tools)
        self.agent_model = genai.GenerativeModel(
            agent_model_name,
            tools=[{'google_search': {}}],
            safety_settings=safety_settings
        )

        # Master Model (Pro, no tools)
        self.master_model = genai.GenerativeModel(
            master_model_name,
            safety_settings=safety_settings
        )

        logger.info(f"Coffee Council initialized. Agents: {agent_model_name}, Master: {master_model_name}")

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

        full_prompt = (
            f"{persona_prompt}\n\n"
            f"TASK: {search_instruction}\n"
            f"IMPORTANT: Use your Google Search tool to find the absolute latest "
            f"news and data from the last 24-48 hours. Do not rely on internal knowledge. "
            f"Summarize your findings with a focus on market impact."
        )

        try:
            # We use generate_content_async.
            # Note: In real-world async usage with 'tools', ensure the library supports it properly.
            # The synchronous call is robust; async support is in newer versions.
            response = await self.agent_model.generate_content_async(full_prompt)

            # Extract text. If search was used, the response handles the grounding.
            return response.text
        except Exception as e:
            logger.error(f"Agent '{persona_key}' failed: {e}")
            return f"Error conducting research: {str(e)}"

    async def decide(self, contract_name: str, ml_signal: dict, research_reports: dict) -> dict:
        """
        Synthesizes research reports and ML signals to make a final trading decision.

        Args:
            contract_name: Name of the contract (e.g. "KC H25")
            ml_signal: The raw signal dictionary from the ML model.
            research_reports: A dictionary of report strings from the agents.

        Returns:
            A dictionary containing the final decision (JSON parsed).
        """
        master_persona = self.personas.get('master', "You are the Chief Strategist.")

        # Format the reports for the prompt
        reports_text = ""
        for agent, report in research_reports.items():
            reports_text += f"\n--- {agent.upper()} REPORT ---\n{report}\n"

        full_prompt = (
            f"{master_persona}\n\n"
            f"You have received reports from your 4 specialized desks and a Quantitative ML model "
            f"regarding contract {contract_name}.\n\n"
            f"QUANT MODEL SIGNAL:\n{json.dumps(ml_signal, indent=2)}\n"
            f"{reports_text}\n\n"
            f"TASK: Synthesize these inputs. Weigh them based on current market narratives "
            f"(e.g., weather trumps macro during flowering season).\n"
            f"OUTPUT FORMAT: You must output a valid JSON object ONLY, with no markdown formatting. "
            f"The JSON must have these keys:\n"
            f"- 'projected_price_5_day': (float) Your price target.\n"
            f"- 'direction': (string) 'BULLISH', 'BEARISH', or 'NEUTRAL'.\n"
            f"- 'confidence': (float) A value between 0.0 and 1.0.\n"
            f"- 'reasoning': (string) A concise explanation of your decision (max 2 sentences).\n"
        )

        try:
            # Enforce JSON if supported by the model version, otherwise prompt engineering handles it.
            # generation_config = genai.GenerationConfig(response_mime_type="application/json")
            # For compatibility, we'll rely on the prompt but stripping code blocks is wise.

            response = await self.master_model.generate_content_async(full_prompt)
            text_response = response.text.strip()

            # Clean up markdown code blocks if present
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]

            return json.loads(text_response.strip())

        except json.JSONDecodeError:
            logger.error(f"Master Strategist failed to output valid JSON. Raw: {response.text}")
            # Fallback to ML signal if Master fails to format correctly
            return {
                "projected_price_5_day": ml_signal.get('expected_price', 0.0),
                "direction": "NEUTRAL", # Safe default on parse error
                "confidence": 0.0,
                "reasoning": f"Master Strategist JSON Error. Fallback to ML."
            }
        except Exception as e:
            logger.error(f"Master Strategist failed: {e}")
            raise e
