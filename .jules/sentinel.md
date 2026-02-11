## 2025-02-18 - [Security] Sentinel DoS & XML Hardening
**Vulnerability:**
1.  **Denial of Service (DoS):** `_fetch_rss_safe` in `trading_bot/sentinels.py` used `await response.text()`, which loads the entire response body into memory. A malicious or misconfigured RSS feed (controlled via config overrides) could serve an infinitely large file, causing an Out-Of-Memory (OOM) crash.
2.  **Insufficient XML Sanitization:** The `_escape_xml` method only escaped basic XML entities (`&`, `<`, `>`) but failed to remove invalid XML control characters (e.g., null bytes, backspace). This could potentially disrupt LLM prompt construction or be exploited if the LLM's XML parser is strict.

**Learning:**
- **Trust No External Input Size:** When fetching data from external URLs (even if trusted domains like Google News), always enforce a strict size limit. `aiohttp.ClientResponse.text()` is dangerous for untrusted content.
- **Bytes vs Strings:** `feedparser` handles encoding detection robustly when provided with *bytes*. Manually decoding with `errors='ignore'` is a regression that breaks non-UTF-8 feeds. Always pass raw bytes to parsers that support it.
- **XML is Strict:** XML 1.0 forbids most control characters (0x00-0x1F) except tab, newline, and carriage return. Regex sanitization is required before embedding untrusted text into XML-based LLM prompts.

**Prevention:**
- Use `response.content.iter_chunked(n)` with a running total check to enforce size limits on all external HTTP fetches.
- Use `re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)` to sanitize text before inclusion in XML payloads.
- Verify encoding handling by preferring byte-stream processing over manual decoding where possible.
