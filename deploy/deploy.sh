#!/usr/bin/env bash
# Copyright (c) 2026 Kishore Sridhar - 611451003
# Tatung University — I4210 AI實務專題
# deploy/deploy.sh — pull tag, set nvpmodel, restart, healthcheck
# → rollback on fail
set -euo pipefail

TAG="${1:?Usage: deploy.sh <vX.Y.Z>}"
ENV="${DEPLOY_ENV:-production}"
STATE_DIR=/var/lib/edgeai-hw6

# STATE_DIR must be writable by the deploy user — done once during D5 setup:
#   sudo mkdir -p /var/lib/edgeai-hw6
#   sudo chown $USER:$USER /var/lib/edgeai-hw6
# This script avoids sudo on state-file ops so deploy.yml's non-interactive
# SSH session doesn't get blocked waiting for a password it can't supply.
# The only sudo'd commands below are nvpmodel and jetson_clocks, which
# Step 0.0 already configured for NOPASSWD.
mkdir -p "$STATE_DIR"

# 1. Resolve the configured power-mode NAME → numeric ID for THIS Jetson SKU.
MODE_NAME=$(jq -r ".$ENV" deploy/power_profile.json)
PAT="<\s*POWER_MODEL\s+ID=[0-9]+\s+NAME=$MODE_NAME\s*>"
MODE_ID=$(grep -oE "$PAT" /etc/nvpmodel.conf | grep -oE "ID=[0-9]+" | cut -d= -f2 | head -1)

if [ -z "$MODE_ID" ]; then
    echo "[deploy] ERROR: power mode '$MODE_NAME' not found in /etc/nvpmodel.conf"
    exit 1
fi
echo "[deploy] Setting nvpmodel to $MODE_NAME (ID=$MODE_ID) for env=$ENV"
sudo nvpmodel -m "$MODE_ID"
sudo jetson_clocks
sleep 2

# 2. Save the currently-deployed tag for Part E's rollback.sh
if [ -f "$STATE_DIR/deployed.txt" ]; then
    PREV=$(cat "$STATE_DIR/deployed.txt")
    echo "$PREV" >> "$STATE_DIR/deployed.txt.history"
    echo "[deploy] Previous tag was $PREV (saved for rollback)"
fi

# 3. Pull the requested tag, recreate the inference container.
export IMAGE_TAG="$TAG"
docker compose -f deploy/docker-compose.yml pull || \
    echo "[deploy] WARNING: pull failed; using local cache"
docker compose -f deploy/docker-compose.yml up -d --force-recreate

# 4. Wait for health (D3); roll back on fail (Part E hooks here).
if ! bash deploy/healthcheck.sh; then
    echo "[deploy] Healthcheck failed - rolling back"
    if [ -x deploy/rollback.sh ]; then
        bash deploy/rollback.sh
    else
        # Part E delivers rollback.sh; until then, just fail loud.
        echo "[deploy] WARNING: deploy/rollback.sh not yet implemented (Part E)"
    fi
    exit 1
fi

# 5. Mark this tag as the new current.
echo "$TAG" > "$STATE_DIR/deployed.txt"
echo "[deploy] Deployed $TAG at power mode $MODE_NAME"