#!/usr/bin/env bash
set -euo pipefail

PATTERNS=(
  "paseoms"
  "paseo_ms"
  "TC-001"
  "TC-002"
  "Branch_ID"
  "Client_Vault"
  "Active_Branches"
  "Master_Sales_Data"
  "Drop_Zone"
  "OPENCLAW_TOKEN"
  "GEMINI_API_KEY"
  "ODOO_PASSWORD"
  "Silao"
  "silao"
  "guanajuato_paseo"
  "emmanuel_ortiz_private"
  "franchicia_maestra"
)

REGEX=$(IFS=\|; echo "${PATTERNS[*]}")

echo "[Blacklist Guardrail] Auditing tracked and working tree files..."

# Check git-tracked files, staged files, and untracked files (excluding .git)
FAILED=0
FILES=$(git ls-files --cached --others --exclude-standard)

for FILE in $FILES; do
  if [[ -f "$FILE" ]]; then
    # Ignore check_blacklist.sh and .gitignore from matching rule definitions
    if [[ "$FILE" == "scripts/check_blacklist.sh" || "$FILE" == ".gitignore" ]]; then
      continue
    fi
    if grep -Eiq "$REGEX" "$FILE"; then
      echo "ERROR: Blacklisted pattern found in: $FILE"
      grep -Ein "$REGEX" "$FILE" || true
      FAILED=1
    fi
  fi
done

if [[ $FAILED -ne 0 ]]; then
  echo "Blacklist check FAILED. Commit aborted."
  exit 1
fi

echo "Blacklist check PASSED. Zero clean-room violations found."
exit 0
