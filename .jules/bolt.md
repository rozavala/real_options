## 2024-05-22 - Efficient Log Reading
**Learning:** Naively using `readlines()` on potentially large log files to get the last N lines is extremely inefficient (O(file_size)). Seeking to the end and reading backwards (O(N)) provides massive speedups (150x+ for 10MB files).
**Action:** Use the new `tail_file` utility in `dashboard_utils.py` for any log tailing requirements.
## 2025-01-31 - Efficient Timestamp Parsing
**Learning:** Parsing timestamps on a large, concatenated dataframe repeatedly is O(N) where N grows with history. Parsing on the components *before* concatenation allows caching the parsed result for static historical data, reducing the runtime cost to O(New Data).
**Action:** Implemented in `dashboard_utils.py` for council history loading.
