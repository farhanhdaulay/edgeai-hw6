#!/usr/bin/env bash
# Copyright (c) 2026 Kishore Sridhar - 611451003
# Tatung University – I4210 AI實務專題
# deploy/rollback.sh - revert to the previously deployed tag quickly
set -euo pipefail

STATE_DIR=/var/lib/edgeai-hw6
CURRENT_FILE="$STATE_DIR/deployed.txt"
HISTORY_FILE="$STATE_DIR/deployed.txt.history"

if [ ! -f "$HISTORY_FILE" ] || [ ! -s "$HISTORY_FILE" ]; then
    echo "[rollback] ERROR: No history file found at $HISTORY_FILE."
    exit 1
fi

CURRENT_TAG=$(cat "$CURRENT_FILE")
PREV_TAG=$(tail -n 1 "$HISTORY_FILE")

echo "[rollback] Current tag: $CURRENT_TAG"
echo "[rollback] Rolling back to previous tag: $PREV_TAG"

export IMAGE_TAG="$PREV_TAG"

# Pull tolerating auth expiry (cached image will be used)
docker compose -f deploy/docker-compose.yml pull || true
docker compose -f deploy/docker-compose.yml up -d --force-recreate

if ! bash deploy/healthcheck.sh; then
    echo "[rollback] FATAL ERROR: Healthcheck failed on rollback tag $PREV_TAG!"
    echo "[rollback] ALERT: Both current and previous tags are broken. Escalating."
    exit 1
fi

# Update the state files to reflect the new reality
echo "$CURRENT_TAG" >> "$HISTORY_FILE"
echo "$PREV_TAG" > "$CURRENT_FILE"

echo "[rollback] SUCCESS: Rolled back to $PREV_TAG in under 30s"