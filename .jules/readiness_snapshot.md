# System Readiness Snapshot - 2026-02-06

This snapshot represents the system state as of February 6, 2026, run in the development sandbox.

## Summary
- **Total Checks:** 27
- **Passed:** 17
- **Failed:** 6 (mostly due to missing API keys)
- **Warnings:** 3
- **Skipped:** 1 (IBKR)

## Key Findings
- **API Keys:** Missing `XAI_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
- **Data Directories:** `./data/tms` and `./logs` were reported as missing during the check.
- **Market Status:** Market was CLOSED at the time of the check (17:26 NY time).
- **YFinance Fallback:** Successfully fetched data for `KC=F`.

## Detailed Report
```
  ❌ Environment Variables: 4 required keys MISSING
  ✅ Configuration: Valid config (34 sections)
  ✅ State Manager: Read/Write OK | 0 keys
  ❌ Data Directories: Missing: ['./data/tms', './logs']
  ⚠️ Trade Ledger: File missing (will be created)
  ⚠️ Council History: File missing (will be created)
  ✅ Chronometer: UTC: 22:26:05 | NY: 17:26:05 | Market: CLOSED
  ✅ Holiday Calendar: Calendar loaded
  ⏭️ IBKR: Skipped by user
  ✅ YFinance Fallback: Data OK (KC=F) | Rows: 1
  ❌ X Sentinel: XAI_API_KEY not set
  ✅ Price Sentinel: Initialized (Lazy) | Threshold: 1.5%
  ✅ Weather Sentinel: 8 regions (profile)
  ❌ News Sentinel: Missing key inputs argument! (API Key missing)
  ❌ Logistics Sentinel: Missing key inputs argument! (API Key missing)
  ✅ Microstructure Sentinel: Config loaded
  ✅ Prediction Market Sentinel: v2.0 | 4 topics
  ✅ Heterogeneous Router: 0 providers (due to missing keys)
  ❌ Trading Council: Gemini API Key not found.
  ✅ TMS (ChromaDB): Collection OK
  ✅ Semantic Router: 20 routes
  ✅ Rate Limiter: Providers: ['gemini', 'openai', 'anthropic', 'xai']
  ✅ Strategy Definitions: Import OK
  ✅ Position Sizer: Initialized
  ✅ Notifications: Enabled
```
