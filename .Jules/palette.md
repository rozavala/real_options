## 2026-02-22 - [Timezone Naivety in Heartbeats]
**Learning:** `datetime.fromtimestamp` returns local naive time. When passed to a relative time helper that defaults to `utcnow()` for naive inputs, it causes massive offsets (e.g. "5h ago" instead of "just now").
**Action:** Always convert naive system timestamps to UTC-aware (`.astimezone(timezone.utc)`) before passing to relative time logic.
