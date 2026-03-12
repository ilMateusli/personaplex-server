#!/usr/bin/env bash
set -euo pipefail

# ─── PersonaPlex + Qwen3-TTS Start Script ────────────────────────────────────
# Koyeb/standalone bootstrap:
#   - Nginx         → :8998 (reverse proxy)
#   - PersonaPlex   → :8999 (WebSocket voice)
#   - Qwen3-TTS     → :8880 (voice cloning API)

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[start]${NC} $1"; }
ok()  { echo -e "${GREEN}[  OK ]${NC} $1"; }

log "Starting services via supervisord..."
log "  Nginx          → :8998 (reverse proxy)"
log "  PersonaPlex    → :8999 (WebSocket voice)"
log "  Qwen3-TTS      → :8880 (voice cloning API)"
echo ""
exec supervisord -c /app/supervisord.conf
