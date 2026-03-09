"""
RunPod Serverless Handler — PersonaPlex + Qwen3-TTS
====================================================

Starts both services (PersonaPlex on :8999, Qwen3-TTS on :8880) via Supervisor,
then proxies RunPod job requests to the appropriate backend.

Supported endpoints (via input.endpoint):
  - "voice-clone": POST /v1/audio/voice-clone on Qwen3-TTS
  - "speech":      POST /v1/audio/speech on Qwen3-TTS
  - "voices":      GET /v1/audio/voices on Qwen3-TTS
  - "health":      Combined health check (both services)
  - "personaplex-health": PersonaPlex health only
  - "tts-health":  Qwen3-TTS health only

For PersonaPlex WebSocket sessions, use RunPod's proxy port 8998 directly
(Nginx routes /api/chat to PersonaPlex WebSocket).

Example voice-clone input:
{
    "input": {
        "endpoint": "voice-clone",
        "text": "Hello world",
        "language": "English",
        "ref_audio": "<base64 or URL>",
        "ref_text": "Reference transcript"
    }
}
"""

import base64
import json
import logging
import os
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request

import runpod

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("runpod-handler")

# ─── Service URLs ─────────────────────────────────────────────────────────────

TTS_URL = "http://127.0.0.1:8880"
PERSONAPLEX_URL = "http://127.0.0.1:8999"
STARTUP_TIMEOUT = int(os.getenv("RUNPOD_STARTUP_TIMEOUT", "600"))
STARTUP_POLL_INTERVAL = float(os.getenv("RUNPOD_STARTUP_POLL_INTERVAL", "5"))

# ─── Startup ──────────────────────────────────────────────────────────────────

supervisor_proc = None
services_started = False


def start_services():
    """Start Nginx + PersonaPlex + Qwen3-TTS via Supervisor."""
    global supervisor_proc

    if supervisor_proc and supervisor_proc.poll() is None:
        logger.info("Services already running under supervisord")
        return

    logger.info("Starting services via supervisord...")
    supervisor_proc = subprocess.Popen(
        ["supervisord", "-c", "/app/supervisord.conf"],
    )
    logger.info(f"Supervisord started (pid={supervisor_proc.pid})")


def wait_for_services(timeout=600):
    """Wait until at least Qwen3-TTS is ready."""
    logger.info("Waiting for services to become ready...")
    start = time.time()

    tts_ready = False
    personaplex_ready = False

    while time.time() - start < timeout:
        if supervisor_proc and supervisor_proc.poll() is not None:
            raise RuntimeError(
                f"supervisord exited before startup completed (code={supervisor_proc.returncode})"
            )

        if not tts_ready:
            try:
                res = urlopen(f"{TTS_URL}/health", timeout=5)
                data = json.loads(res.read())
                if data.get("status") == "ok":
                    tts_ready = True
                    elapsed = time.time() - start
                    logger.info(f"Qwen3-TTS ready after {elapsed:.0f}s")
            except Exception:
                pass

        if not personaplex_ready:
            try:
                res = urlopen(f"{PERSONAPLEX_URL}/", timeout=5)
                if res.status == 200:
                    personaplex_ready = True
                    elapsed = time.time() - start
                    logger.info(f"PersonaPlex ready after {elapsed:.0f}s")
            except Exception:
                pass

        if tts_ready and personaplex_ready:
            logger.info("All services ready")
            return True

        if tts_ready:
            # TTS is the critical one; PersonaPlex can take longer
            logger.info("TTS ready, PersonaPlex still loading (continuing anyway)")
            return True

        time.sleep(STARTUP_POLL_INTERVAL)

    logger.warning(f"Services not fully ready after {timeout}s (tts={tts_ready}, personaplex={personaplex_ready})")
    return tts_ready  # At minimum TTS must be up


def ensure_services_started():
    """Boot the local services once before the RunPod worker starts accepting jobs."""
    global services_started

    if services_started:
        return

    start_services()

    if not wait_for_services(timeout=STARTUP_TIMEOUT):
        raise RuntimeError(
            "Timed out waiting for Qwen3-TTS to become ready. "
            "Check supervisord logs for the failing service."
        )

    services_started = True


def _fetch_json(url, data=None, timeout=120):
    """Helper to POST/GET JSON to a local service."""
    if data is not None:
        body = json.dumps(data).encode()
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
    else:
        req = Request(url)

    res = urlopen(req, timeout=timeout)
    content_type = res.headers.get("Content-Type", "")

    if "json" in content_type:
        return json.loads(res.read())
    else:
        # Binary response (audio)
        return res.read()


# ─── Handler ──────────────────────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler — routes to the right backend."""
    job_input = job.get("input", {})
    endpoint = job_input.get("endpoint", "health")

    try:
        if endpoint == "health":
            return _handle_health()
        elif endpoint == "tts-health":
            return _handle_tts_health()
        elif endpoint == "personaplex-health":
            return _handle_personaplex_health()
        elif endpoint == "speech":
            return _handle_speech(job_input)
        elif endpoint == "voice-clone":
            return _handle_voice_clone(job_input)
        elif endpoint == "voices":
            return _handle_voices()
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}
    except Exception as e:
        logger.error(f"[{endpoint}] Error: {e}")
        return {"error": str(e)}


def _handle_health():
    """Combined health check for both services."""
    tts_ok = False
    personaplex_ok = False
    tts_data = {}

    try:
        data = _fetch_json(f"{TTS_URL}/health", timeout=10)
        tts_ok = data.get("status") == "ok"
        tts_data = data
    except Exception as e:
        tts_data = {"error": str(e)}

    try:
        urlopen(f"{PERSONAPLEX_URL}/", timeout=5)
        personaplex_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if (tts_ok and personaplex_ok) else "degraded" if tts_ok else "loading",
        "tts": {"ready": tts_ok, **tts_data},
        "personaplex": {"ready": personaplex_ok},
    }


def _handle_tts_health():
    data = _fetch_json(f"{TTS_URL}/health", timeout=10)
    return data


def _handle_personaplex_health():
    try:
        res = urlopen(f"{PERSONAPLEX_URL}/", timeout=5)
        return {"status": "ok", "http_status": res.status}
    except (HTTPError, URLError) as e:
        return {"status": "error", "error": str(e)}


def _handle_voices():
    data = _fetch_json(f"{TTS_URL}/v1/audio/voices", timeout=10)
    return data


def _handle_speech(job_input):
    """Proxy to Qwen3-TTS /v1/audio/speech."""
    text = job_input.get("text", "")
    if not text:
        return {"error": "Missing 'text' field"}

    payload = {
        "input": text,
        "voice": job_input.get("language", "Auto"),
        "response_format": job_input.get("response_format", "wav"),
    }

    audio_bytes = _fetch_json(f"{TTS_URL}/v1/audio/speech", data=payload, timeout=120)

    if isinstance(audio_bytes, dict):
        return audio_bytes  # Error response

    return {
        "audio": base64.b64encode(audio_bytes).decode(),
        "format": payload["response_format"],
    }


def _handle_voice_clone(job_input):
    """Proxy to Qwen3-TTS /v1/audio/voice-clone."""
    text = job_input.get("text", "")
    if not text:
        return {"error": "Missing 'text' field"}

    ref_audio = job_input.get("ref_audio", "")
    if not ref_audio:
        return {"error": "Missing 'ref_audio' field"}

    payload = {
        "text": text,
        "language": job_input.get("language", "Auto"),
        "ref_audio": ref_audio,
        "ref_text": job_input.get("ref_text", None),
        "response_format": job_input.get("response_format", "wav"),
        "stream": False,  # RunPod serverless is request/response, no streaming
    }

    audio_bytes = _fetch_json(f"{TTS_URL}/v1/audio/voice-clone", data=payload, timeout=120)

    if isinstance(audio_bytes, dict):
        return audio_bytes  # Error response

    return {
        "audio": base64.b64encode(audio_bytes).decode(),
        "format": payload["response_format"],
    }


def main():
    ensure_services_started()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
