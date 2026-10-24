# Ethical OSINT & Rate Limiting Principles

## 1. Respectful Data Retrieval
When using public endpoints or DuckDuckGo searches:
- Implement exponential backoff.
- Enforce strict sleep delays (minimum 2.0s between requests).
- Restrict discovery to prioritized top candidates rather than broad automated sweeping.

## 2. PII Isolation
Contact details and individual commercial information must remain in `data/private/` and must never be committed to source control.
Aggregate statistics (e.g., municipal digital maturity indexes, SCIAN activity distributions) are the only data products safe for public release.
