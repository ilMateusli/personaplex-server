#!/usr/bin/env bash
set -euo pipefail

# ─── PersonaPlex + Qwen3-TTS Start Script ────────────────────────────────────
# Detects the runtime environment and starts the appropriate entrypoint:
#   - RunPod Serverless: handler.py (starts supervisord + proxies jobs)
#   - Standalone/Koyeb:  supervisord (Nginx + PersonaPlex + Qwen3-TTS)

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[start]${NC} $1"; }
ok()  { echo -e "${GREEN}[  OK ]${NC} $1"; }

if [ -n "${RUNPOD_POD_ID:-}" ] || [ -n "${RUNPOD_ENDPOINT_ID:-}" ]; then
  log "RunPod environment detected — starting handler..."
  exec python3 /app/handler.py
else
  log "Starting services via supervisord..."
  log "  Nginx          → :8998 (reverse proxy)"
  log "  PersonaPlex    → :8999 (WebSocket voice)"
  log "  Qwen3-TTS      → :8880 (voice cloning API)"
  echo ""
  exec supervisord -c /app/supervisord.conf
fi
