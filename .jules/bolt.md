## 2024-05-22 - Efficient Log Reading
**Learning:** Naively using `readlines()` on potentially large log files to get the last N lines is extremely inefficient (O(file_size)). Seeking to the end and reading backwards (O(N)) provides massive speedups (150x+ for 10MB files).
**Action:** Use the new `tail_file` utility in `dashboard_utils.py` for any log tailing requirements.
