#!/usr/bin/env bash
# Copyright (c) 2026 Kishore and Farhan
# Tatung University — I4210 AI實務專題
# deploy/healthcheck.sh - verify the inference container is healthy.
# Polls /healthz; requires 3 consecutive 200 responses inside 60 s total.
set -euo pipefail

URL="${HEALTHZ_URL:-http://localhost:8000/healthz}"
DEADLINE=$((SECONDS + 60))
STREAK=0
NEEDED=3

while [ "$SECONDS" -lt "$DEADLINE" ]; do
    if body=$(curl -fSS --max-time 2 "$URL" 2>/dev/null) && \
       echo "$body" | jq -e '.status == "healthy"' >/dev/null 2>&1; then
        STREAK=$((STREAK + 1))
        echo "[healthcheck] OK ($STREAK/$NEEDED): $body"
        [ "$STREAK" -ge "$NEEDED" ] && exit 0
    else
        [ "$STREAK" -gt 0 ] && echo "[healthcheck] streak broken at $STREAK"
        STREAK=0
    fi
    sleep 2
done

echo "[healthcheck] FAILED - no $NEEDED consecutive successes in 60 s" >&2
exit 1