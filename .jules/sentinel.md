## 2025-02-18 - Hardcoded Secrets in Config
**Vulnerability:** Found hardcoded Pushover, FRED, and Nasdaq API keys in `config.json`.
**Learning:** `config.json` was being loaded directly without checking for environment variables for these specific keys, despite other keys having "LOADED_FROM_ENV" support.
**Prevention:** Ensure `config_loader.py` checks for environment variable overrides for ALL sensitive fields, not just some. Use `LOADED_FROM_ENV` placeholder in `config.json` to signal that a value is expected from the environment.
