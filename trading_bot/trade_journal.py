"""
Automated Trade Journal — Post-mortem narratives for every closed trade.

Triggered by reconciliation when a position is closed.
Generates a structured narrative stored in TMS for retrieval by
DSPy, TextGrad, and Reflexion loops.
"""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TradeJournal:
    """Generate and store post-mortem narratives for closed trades."""

    def __init__(self, config: dict, tms=None, router=None):
        self.config = config
        self.tms = tms
        self.router = router
        data_dir = config.get('data_dir', 'data')
        self.journal_file = os.path.join(data_dir, "trade_journal.json")
        self._entries = self._load_entries()

    def _load_entries(self) -> list:
        if os.path.exists(self.journal_file):
            try:
                with open(self.journal_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_entries(self):
        os.makedirs(os.path.dirname(self.journal_file) or '.', exist_ok=True)
        with open(self.journal_file, 'w') as f:
            json.dump(self._entries, f, indent=2, default=str)

    async def generate_post_mortem(
        self,
        position_id: str,
        entry_decision: dict,
        exit_data: dict,
        pnl: float,
        contract: str,
    ) -> Optional[dict]:
        """Generate a structured post-mortem for a closed trade."""
        try:
            entry = {
                "position_id": position_id,
                "contract": contract,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": pnl,
                "outcome": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT",
                "entry_thesis": entry_decision.get('reasoning', 'Unknown'),
                "entry_direction": entry_decision.get('direction', 'Unknown'),
                "entry_confidence": entry_decision.get('confidence', 0.0),
                "entry_strategy": entry_decision.get('strategy_type', 'Unknown'),
                "trigger_type": entry_decision.get('trigger_type', 'Unknown'),
            }

            # Generate narrative (LLM if available, template if not)
            if self.router:
                try:
                    narrative = await self._generate_llm_narrative(entry, exit_data)
                    entry["narrative"] = narrative
                    entry["key_lesson"] = narrative.get("lesson", "No lesson extracted")
                except Exception as e:
                    logger.warning(f"LLM narrative generation failed: {e}")
                    entry["narrative"] = self._generate_template_narrative(entry)
                    entry["key_lesson"] = (
                        f"Trade {'succeeded' if pnl > 0 else 'failed'} — review thesis manually"
                    )
            else:
                entry["narrative"] = self._generate_template_narrative(entry)
                entry["key_lesson"] = (
                    f"Trade {'succeeded' if pnl > 0 else 'failed'} — review thesis manually"
                )

            # Persist to journal file
            self._entries.append(entry)
            self._save_entries()

            # Store in TMS for retrieval by Reflexion loops
            if self.tms:
                try:
                    doc_text = (
                        json.dumps(entry["narrative"])
                        if isinstance(entry["narrative"], dict)
                        else str(entry["narrative"])
                    )
                    self.tms.encode("trade_journal", doc_text, {
                        "position_id": position_id,
                        "contract": contract,
                        "outcome": entry["outcome"],
                        "pnl": str(pnl),
                        "direction": entry["entry_direction"],
                        "strategy": entry["entry_strategy"],
                    })
                except Exception as e:
                    logger.warning(f"Failed to store journal entry in TMS: {e}")

            logger.info(
                f"Trade journal entry created: {contract} {entry['outcome']} "
                f"(${pnl:.2f})"
            )
            return entry

        except Exception as e:
            logger.error(f"Failed to generate post-mortem for {position_id}: {e}")
            return None

    async def _generate_llm_narrative(self, entry: dict, exit_data: dict) -> dict:
        """Use LLM to generate structured narrative."""
        from trading_bot.heterogeneous_router import AgentRole

        prompt = f"""Analyze this closed trade and generate a structured post-mortem.

TRADE DETAILS:
- Contract: {entry['contract']}
- Direction: {entry['entry_direction']}
- Strategy: {entry['entry_strategy']}
- Entry Confidence: {entry['entry_confidence']}
- Trigger: {entry['trigger_type']}
- P&L: ${entry['pnl']:.2f} ({'WIN' if entry['pnl'] > 0 else 'LOSS'})
- Original Thesis: {entry['entry_thesis'][:500]}

Generate a JSON object with:
1. "summary": One-sentence summary of what happened
2. "thesis_validated": true/false
3. "what_went_right": List of things the system got right
4. "what_went_wrong": List of things the system got wrong
5. "lesson": One specific, actionable lesson for future trades
6. "rule_suggestion": A concrete rule (e.g., "Avoid straddles when IV rank > 80th pctl")
"""

        response = await self.router.route(
            AgentRole.PERMABEAR, prompt, response_json=True
        )

        # Parse JSON response (using robust regex extraction)
        import re
        cleaned = response.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        return json.loads(cleaned)

    def _generate_template_narrative(self, entry: dict) -> str:
        """Template narrative when LLM unavailable."""
        outcome = "succeeded" if entry['pnl'] > 0 else "failed"
        return (
            f"Trade on {entry['contract']}: Entered {entry['entry_direction']} "
            f"({entry['entry_strategy']}) with confidence {entry['entry_confidence']:.2f}. "
            f"Triggered by {entry['trigger_type']}. "
            f"Trade {outcome} with P&L ${entry['pnl']:.2f}. "
            f"Original thesis: {entry['entry_thesis'][:200]}"
        )

    def get_recent_entries(self, n: int = 10) -> list:
        return self._entries[-n:]

    def get_entries_by_outcome(self, outcome: str) -> list:
        return [e for e in self._entries if e.get('outcome') == outcome]
