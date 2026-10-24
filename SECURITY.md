# Security Policy & Clean-Room Directive

## 1. Clean-Room Architecture

`mexico-commercial-osint` is an open-source public toolkit strictly isolated from proprietary customer systems, tenant records, operational POS data, or private commercial instances.

### Strict Data Separation
- **Public & Tracked in Git:**
  - Code, open schema contracts, reproducible workflows, documentation, and synthetic unit test fixtures.
- **Private & Gitignored (`data/private/`, `data/raw/`, `.env`):**
  - Real runtime exports containing individual establishment contact details, specific targeted lead tables, operational credentials, or API keys.

## 2. Forbidden Patterns & Guardrails

The repository enforces an automated pre-commit and CI guardrail via `scripts/check_blacklist.sh`. Commits containing any of the following patterns will be rejected:
- Tenant / local operation identifiers
- Private branch coordinates or branch IDs
- Master sales datasets or private client vault references
- Internal API tokens or credentials

## 3. Reporting a Vulnerability

If you discover a security vulnerability or credential leak within this public repository, please open a private report or contact the maintainer directly instead of opening a public issue.
