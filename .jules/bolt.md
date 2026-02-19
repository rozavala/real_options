## 2024-05-22 - Efficient Log Reading
**Learning:** Naively using `readlines()` on potentially large log files to get the last N lines is extremely inefficient (O(file_size)). Seeking to the end and reading backwards (O(N)) provides massive speedups (150x+ for 10MB files).
**Action:** Use the new `tail_file` utility in `dashboard_utils.py` for any log tailing requirements.
## 2025-01-31 - Efficient Timestamp Parsing
**Learning:** Parsing timestamps on a large, concatenated dataframe repeatedly is O(N) where N grows with history. Parsing on the components *before* concatenation allows caching the parsed result for static historical data, reducing the runtime cost to O(New Data).
**Action:** Implemented in `dashboard_utils.py` for council history loading.
## 2025-02-04 - Parallel Sentinel Fetching
**Learning:** Sequential execution of I/O bound tasks (like RSS fetching in sentinels) inside an `async` loop needlessly increases cycle duration. `asyncio.gather` reduces total latency from `sum(tasks)` to `max(tasks)`, which is critical for maintaining responsiveness in the single-threaded orchestrator loop.
**Action:** Parallelize independent I/O tasks in `LogisticsSentinel` and `NewsSentinel`.
## 2025-02-17 - Vectorized Decision Grading
**Learning:** Iterating over DataFrame rows with `iterrows()` for conditional logic is extremely slow (O(N) Python loop overhead). Boolean masking and vectorized pandas operations provided a ~30x speedup for decision grading logic, which is critical for dashboard responsiveness as history grows.
**Action:** Always prefer vectorized operations (`loc` with boolean masks) over row-wise iteration for data processing.
## 2025-02-13 - Efficient DataFrame Copying
**Learning:** Calling `df.copy()` on a large DataFrame copies all columns, including large text fields not needed for downstream processing. This introduces significant memory and CPU overhead. Subsetting to only necessary columns *before* copying reduced execution time by ~11% on large datasets (10k rows) and significantly lowered peak memory usage.
**Action:** Always filter DataFrames to the minimal required columns before creating a copy for isolated processing.
## 2025-02-24 - Async Mocking of Context Managers
**Learning:** When unit testing `aiohttp` client interactions, `session.get()` returns an async context manager, not a direct coroutine. Mocking it as a simple `AsyncMock` fails because `async with` expects an object with `__aenter__` and `__aexit__`. Correct approach is to use `MagicMock` that returns an object with `AsyncMock` for `__aenter__`.
**Action:** Use proper context manager mocking patterns for `aiohttp` tests to avoid `RuntimeWarning: coroutine was never awaited`.
