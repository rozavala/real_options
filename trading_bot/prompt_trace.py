"""Prompt Trace Logger — Records prompt stack and model routing per LLM call.

WHY THIS EXISTS:
The trading council assembles prompts from 5+ layers across 8 agents per cycle,
routed to 4 LLM providers with fallback logic. No record exists of which prompt
stack was used, which model actually handled requests after fallbacks, or whether
DSPy-optimized prompts were active. This blocks DSPy A/B testing, anomaly
debugging, and model performance attribution.

SCHEMA (20 columns):
    timestamp              — UTC ISO8601
    cycle_id               — FK to council_history
    commodity              — Ticker (KC, CC)
    contract               — Contract month (KCN6)
    phase                  — research/debate/decision/compliance/devils_advocate
    agent                  — Canonical agent name
    prompt_source          — legacy/commodity_profile/dspy_optimized
    model_provider         — Actual provider used (after fallbacks)
    model_name             — Actual model ID used
    assigned_provider      — Intended provider from assignments
    assigned_model         — Intended model from assignments
    persona_hash           — SHA256[:12] of persona text
    dspy_version           — ISO8601 mtime of DSPy file, empty if N/A
    demo_count             — Few-shot examples injected
    tms_context_count      — TMS documents retrieved
    grounded_freshness_hours — Hours since grounded data gathered
    reflexion_applied      — Whether reflexion block was injected
    prompt_tokens          — Input tokens (0 if direct Gemini fallback)
    completion_tokens      — Output tokens (0 if direct Gemini fallback)
    latency_ms             — Wall-clock time of the LLM call

USAGE:
    from trading_bot.prompt_trace import PromptTraceCollector, PromptTraceRecord

    collector = PromptTraceCollector(cycle_id="KC-a1b2c3d4", commodity="KC", contract="KCN6")
    collector.record(PromptTraceRecord(phase="research", agent="agronomist", ...))
    collector.flush()  # Writes to prompt_trace.csv

    # For dashboard / reports:
    from trading_bot.prompt_trace import get_prompt_trace_df
    df = get_prompt_trace_df(commodity="KC")
"""

import os
import csv
import hashlib
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Optional, List

import pandas as pd
from trading_bot.timestamps import format_ts, parse_ts_column

logger = logging.getLogger(__name__)

# File lives alongside other CSV data files
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_TRACE_PATH = os.path.join(_BASE_DIR, 'prompt_trace.csv')


def set_data_dir(data_dir: str):
    """Configure prompt trace path for a commodity-specific data directory."""
    global PROMPT_TRACE_PATH
    PROMPT_TRACE_PATH = os.path.join(data_dir, 'prompt_trace.csv')
    logger.info(f"PromptTrace data_dir set to: {data_dir}")


# Canonical schema — order matters for CSV columns
SCHEMA_COLUMNS = [
    'timestamp',
    'cycle_id',
    'commodity',
    'contract',
    'phase',
    'agent',
    'prompt_source',
    'model_provider',
    'model_name',
    'assigned_provider',
    'assigned_model',
    'persona_hash',
    'dspy_version',
    'demo_count',
    'tms_context_count',
    'grounded_freshness_hours',
    'reflexion_applied',
    'prompt_tokens',
    'completion_tokens',
    'latency_ms',
]

# String columns get empty string default, numeric get 0
_STRING_COLUMNS = {
    'timestamp', 'cycle_id', 'commodity', 'contract', 'phase', 'agent',
    'prompt_source', 'model_provider', 'model_name', 'assigned_provider',
    'assigned_model', 'persona_hash', 'dspy_version',
}


def hash_persona(text: str) -> str:
    """SHA256[:12] of persona text for drift detection."""
    if not text:
        return hashlib.sha256(b'').hexdigest()[:12]
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


@dataclass
class PromptTraceRecord:
    """One row in prompt_trace.csv."""
    timestamp: str = ""
    cycle_id: str = ""
    commodity: str = ""
    contract: str = ""
    phase: str = ""
    agent: str = ""
    prompt_source: str = "legacy"
    model_provider: str = ""
    model_name: str = ""
    assigned_provider: str = ""
    assigned_model: str = ""
    persona_hash: str = ""
    dspy_version: str = ""
    demo_count: int = 0
    tms_context_count: int = 0
    grounded_freshness_hours: float = 0.0
    reflexion_applied: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class PromptTraceCollector:
    """Buffers trace records for a single cycle, then flushes to CSV."""

    def __init__(self, cycle_id: str = "", commodity: str = "", contract: str = ""):
        self._records: List[PromptTraceRecord] = []
        self._lock = threading.Lock()
        self.cycle_id = cycle_id
        self.commodity = commodity
        self.contract = contract

    def record(self, rec: PromptTraceRecord):
        """Add a trace record, injecting cycle-level fields if not set."""
        if not rec.timestamp:
            rec.timestamp = format_ts()
        if not rec.cycle_id:
            rec.cycle_id = self.cycle_id
        if not rec.commodity:
            rec.commodity = self.commodity
        if not rec.contract:
            rec.contract = self.contract
        with self._lock:
            self._records.append(rec)

    def flush(self) -> int:
        """Write buffered records to CSV. Returns count written. Never raises."""
        with self._lock:
            records = list(self._records)
            self._records.clear()

        if not records:
            return 0

        try:
            file_exists = os.path.exists(PROMPT_TRACE_PATH)
            os.makedirs(os.path.dirname(PROMPT_TRACE_PATH), exist_ok=True)

            with open(PROMPT_TRACE_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=SCHEMA_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                for rec in records:
                    row = asdict(rec)
                    writer.writerow({k: row.get(k, '') for k in SCHEMA_COLUMNS})

            logger.info(f"Flushed {len(records)} prompt traces to {PROMPT_TRACE_PATH}")
            return len(records)

        except Exception as e:
            logger.error(f"Failed to flush prompt traces: {e}", exc_info=True)
            return 0


def get_prompt_trace_df(commodity: Optional[str] = None) -> pd.DataFrame:
    """
    Read prompt_trace.csv into a DataFrame with parsed timestamps.
    Returns empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(PROMPT_TRACE_PATH):
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    try:
        df = pd.read_csv(PROMPT_TRACE_PATH)
        # Forward-compat: add missing columns with appropriate defaults
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = '' if col in _STRING_COLUMNS else 0
        if 'timestamp' in df.columns:
            df['timestamp'] = parse_ts_column(df['timestamp'])
        if commodity:
            df = df[df['commodity'] == commodity]
        return df

    except Exception as e:
        logger.error(f"Failed to read prompt_trace.csv: {e}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
