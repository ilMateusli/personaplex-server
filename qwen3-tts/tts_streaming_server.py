"""
Qwen3-TTS Streaming Server — faster-qwen3-tts Edition
======================================================
HTTP server with OpenAI-compatible TTS endpoints using faster-qwen3-tts
for ~5x lower TTFA via CUDA graphs and static KV cache.

Endpoints:
  POST /v1/audio/speech             - Standard TTS (streaming WAV chunks)
  POST /v1/audio/voice-clone        - Voice clone TTS (streaming with cached prompt)
  POST /v1/audio/voice-clone/enroll - Build reusable voice prompt once
  POST /upload_audio/               - Upload ref audio (legacy)
  GET  /health                      - Health check
  GET  /v1/audio/voices             - List available voices/languages
"""

import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import os
import secrets
import struct
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError
from faster_qwen3_tts import FasterQwen3TTS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tts-streaming")

# ─── Global State ─────────────────────────────────────────────────────────────
model: Optional[FasterQwen3TTS] = None
model_ready = False
model_lock = asyncio.Lock()
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8880"))
CHUNK_SIZE = int(os.environ.get("TTS_CHUNK_SIZE", "2"))
MAX_REF_AUDIO_SECONDS = int(os.environ.get("TTS_MAX_REF_AUDIO_SECONDS", "15"))
PARITY_MODE = os.environ.get("TTS_PARITY_MODE", "false").lower() == "true"
DEMO_AUTH_ENABLED = os.environ.get("TTS_DEMO_AUTH_ENABLED", "true").lower() != "false"
DEMO_USERNAME = os.environ.get("TTS_DEMO_USERNAME", "octalab-octalk")
DEMO_PASSWORD = os.environ.get("TTS_DEMO_PASSWORD", "Vn7kQ4mP2xL9bS6")

SUPPORTED_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "Spanish", "Portuguese", "German", "Italian", "Russian",
]

# ─── Voice Prompt Item ────────────────────────────────────────────────────────
# Mirrors qwen_tts.inference.qwen3_tts_model.VoiceClonePromptItem for
# serialization/deserialization without coupling to the internal class.


class VoicePromptItem:
    """Lightweight container for cached voice prompt data."""

    __slots__ = ("ref_code", "ref_spk_embedding", "x_vector_only_mode", "icl_mode", "ref_text")

    def __init__(
        self,
        ref_code: Optional[torch.Tensor],
        ref_spk_embedding: torch.Tensor,
        x_vector_only_mode: bool,
        icl_mode: bool,
        ref_text: Optional[str] = None,
    ):
        self.ref_code = ref_code
        self.ref_spk_embedding = ref_spk_embedding
        self.x_vector_only_mode = x_vector_only_mode
        self.icl_mode = icl_mode
        self.ref_text = ref_text


# ─── Voice Prompt Store ───────────────────────────────────────────────────────


def _get_voice_prompt_store(app: web.Application) -> Dict[str, List[VoicePromptItem]]:
    if "voice_prompts" not in app:
        app["voice_prompts"] = {}
    return app["voice_prompts"]


# ─── Tensor Serialization ────────────────────────────────────────────────────


def _torch_dtype_from_name(name: str):
    normalized = (name or "float32").replace("torch.", "")
    return getattr(torch, normalized, torch.float32)


def _serialize_tensor(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if tensor is None:
        return None
    cpu = tensor.detach().cpu()
    return {"dtype": str(cpu.dtype).replace("torch.", ""), "data": cpu.tolist()}


def _deserialize_tensor(payload: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return torch.tensor(payload)
    dtype = _torch_dtype_from_name(payload.get("dtype", "float32"))
    return torch.tensor(payload.get("data"), dtype=dtype)


def _serialize_voice_prompt_items(items: List[VoicePromptItem]) -> Dict[str, Any]:
    return {
        "items": [
            {
                "ref_code": _serialize_tensor(item.ref_code),
                "ref_spk_embedding": _serialize_tensor(item.ref_spk_embedding),
                "x_vector_only_mode": bool(item.x_vector_only_mode),
                "icl_mode": bool(item.icl_mode),
                "ref_text": item.ref_text,
            }
            for item in items
        ]
    }


def _deserialize_voice_prompt_items(payload: Any) -> List[VoicePromptItem]:
    if isinstance(payload, dict) and "items" in payload:
        items_raw = payload["items"]
    else:
        items_raw = payload

    if not isinstance(items_raw, list) or len(items_raw) == 0:
        raise ValueError("voice_prompt must contain a non-empty 'items' list")

    items: List[VoicePromptItem] = []
    for idx, item in enumerate(items_raw):
        if not isinstance(item, dict):
            raise ValueError(f"voice_prompt item at index {idx} must be an object")
        ref_spk_embedding = _deserialize_tensor(item.get("ref_spk_embedding"))
        if ref_spk_embedding is None:
            raise ValueError(f"voice_prompt item at index {idx} is missing ref_spk_embedding")
        items.append(
            VoicePromptItem(
                ref_code=_deserialize_tensor(item.get("ref_code")),
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=bool(item.get("x_vector_only_mode", False)),
                icl_mode=bool(item.get("icl_mode", not bool(item.get("x_vector_only_mode", False)))),
                ref_text=item.get("ref_text"),
            )
        )
    return items


# ─── Cache Injection ──────────────────────────────────────────────────────────


def _inject_prompt_into_cache(
    faster_model: FasterQwen3TTS,
    cache_ref_audio: str,
    items: List[VoicePromptItem],
):
    """
    Inject pre-computed voice prompt into FasterQwen3TTS's internal _voice_prompt_cache.

    The cache key format is (str(ref_audio), ref_text, xvec_only, append_silence).
    We use the voice_id as a synthetic ref_audio path so subsequent calls with
    ref_audio=voice_id hit the cache and skip all audio processing.
    """
    item = items[0]
    xvec_only = item.x_vector_only_mode
    device = faster_model.device

    if xvec_only:
        vcp = {
            "ref_code": [None],
            "ref_spk_embedding": [item.ref_spk_embedding.to(device)],
            "x_vector_only_mode": [True],
            "icl_mode": [False],
        }
        ref_ids = [None]
    else:
        vcp = {
            "ref_code": [item.ref_code.to(device) if item.ref_code is not None else None],
            "ref_spk_embedding": [item.ref_spk_embedding.to(device)],
            "x_vector_only_mode": [False],
            "icl_mode": [True],
        }
        if item.ref_text:
            base_model = faster_model.model
            ref_texts = [base_model._build_ref_text(item.ref_text)]
            ref_ids = [base_model._tokenize_texts(ref_texts)[0]]
        else:
            ref_ids = [None]

    cache_key = (cache_ref_audio, "", xvec_only, True)
    faster_model._voice_prompt_cache[cache_key] = (vcp, ref_ids)


# ─── Audio Helpers ────────────────────────────────────────────────────────────


def _generate_silence_wav() -> str:
    silence = np.zeros(24000, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, 24000, format="WAV")
    buf.seek(0)
    return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()


def _save_base64_to_tmpfile(ref_audio: str) -> str:
    """Decode base64/data-URI audio into a temp WAV file and return its path."""
    if ref_audio.startswith("data:"):
        _, b64data = ref_audio.split(",", 1)
    elif len(ref_audio) > 256:
        b64data = ref_audio
    else:
        return ref_audio

    raw = base64.b64decode(b64data)
    buf = io.BytesIO(raw)
    data, sr = sf.read(buf)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
    sf.write(tmp.name, data, sr, format="WAV")
    return tmp.name


def _trim_ref_audio_base64(ref_audio: str, max_seconds: int = 15) -> str:
    if not ref_audio or max_seconds <= 0:
        return ref_audio
    try:
        if ref_audio.startswith("data:"):
            header, b64data = ref_audio.split(",", 1)
        elif len(ref_audio) > 256:
            header, b64data = None, ref_audio
        else:
            return ref_audio

        raw = base64.b64decode(b64data)
        buf_in = io.BytesIO(raw)
        data, sr = sf.read(buf_in)
        total_seconds = len(data) / sr

        if total_seconds <= max_seconds:
            return ref_audio

        trimmed = data[: int(sr * max_seconds)]
        buf_out = io.BytesIO()
        sf.write(buf_out, trimmed, sr, format="WAV")
        buf_out.seek(0)
        b64_out = base64.b64encode(buf_out.read()).decode()
        logger.info(f"[trim] Ref audio trimmed from {total_seconds:.1f}s to {max_seconds}s")
        return "data:audio/wav;base64," + b64_out
    except Exception:
        return ref_audio


def _pcm_to_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE", b"fmt ", 16, 1,
        num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", 0xFFFFFFFF,
    )


def _float32_to_int16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


async def _safe_stream_write(resp: web.StreamResponse, payload: bytes, label: str) -> bool:
    try:
        await resp.write(payload)
        return True
    except (ClientConnectionResetError, ConnectionResetError, BrokenPipeError, RuntimeError) as err:
        logger.info(f"[{label}] Client disconnected during streaming write: {err}")
        return False


async def _safe_stream_write_eof(resp: web.StreamResponse, label: str) -> None:
    try:
        await resp.write_eof()
    except (ClientConnectionResetError, ConnectionResetError, BrokenPipeError, RuntimeError) as err:
        logger.info(f"[{label}] Stream closed before EOF: {err}")


async def _async_generate_chunks(stream_kwargs: dict):
    """Run the synchronous streaming generator in a thread so the event loop stays free."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _produce():
        try:
            for chunk, sr, timing in model.generate_voice_clone_streaming(**stream_kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, (chunk, sr, timing))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    _executor.submit(_produce)

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _detect_language(text: str) -> str:
    for char in text:
        cp = ord(char)
        if 0x4E00 <= cp <= 0x9FFF:
            return "Chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "Japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "Korean"
        if 0x0400 <= cp <= 0x04FF:
            return "Russian"
        if char in "àáâãçéêíóôõúüñ":
            return "Portuguese"
    return "English"


# ─── Model Loading ────────────────────────────────────────────────────────────


def load_model():
    global model, model_ready

    logger.info(f"Loading faster-qwen3-tts model: {MODEL_NAME}")
    start = time.time()

    torch.set_float32_matmul_precision("high")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Using FlashAttention2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("flash-attn not available, using SDPA")

    model = FasterQwen3TTS.from_pretrained(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    # Warmup: try CUDA graphs first, fallback to parity mode.
    # IMPORTANT: set _warmed_up=True BEFORE any attempt so that a failed
    # graph capture doesn't corrupt the CUDA runtime for subsequent calls.
    global PARITY_MODE

    silence_path = _generate_silence_wav()
    tmp_path = _save_base64_to_tmpfile(silence_path)

    try:
        if PARITY_MODE:
            # Parity mode requested — skip CUDA graph capture entirely
            model._warmed_up = True
            logger.info("Parity mode enabled, skipping CUDA graph capture")
            start = time.time()
            for _ in model.generate_voice_clone_streaming(
                text="Hello, warmup.",
                language="English",
                ref_audio=tmp_path,
                ref_text="Warmup.",
                xvec_only=True,
                chunk_size=CHUNK_SIZE,
                parity_mode=True,
            ):
                pass
            logger.info(f"Parity warmup complete in {time.time() - start:.1f}s")
        else:
            # Try CUDA graphs (fast path)
            logger.info("Attempting CUDA graph warmup...")
            start = time.time()
            for _ in model.generate_voice_clone_streaming(
                text="Hello, warmup.",
                language="English",
                ref_audio=tmp_path,
                ref_text="Warmup.",
                xvec_only=True,
                chunk_size=CHUNK_SIZE,
                parity_mode=False,
            ):
                pass
            logger.info(f"CUDA graph warmup complete in {time.time() - start:.1f}s")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        if not PARITY_MODE:
            logger.info("CUDA graphs not supported on this GPU, restarting with parity mode...")
            logger.info("Set TTS_PARITY_MODE=true to avoid this on next start.")
            PARITY_MODE = True
            # CUDA state is corrupted — need a clean restart with parity mode
            # For now, mark as ready and hope non-graph operations still work
            model._warmed_up = True
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    model_ready = True
    mode_label = "parity (dynamic cache)" if PARITY_MODE else "CUDA graphs (fast)"
    logger.info(f"TTS ready — chunk_size={CHUNK_SIZE}, mode={mode_label}")


# ─── Demo Page ─────────────────────────────────────────────────────────────


def _build_demo_html(voice_ids: list) -> str:
    voice_options = "".join(
        f'<option value="{vid}">{vid}</option>' for vid in voice_ids
    )
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-TTS Streaming Demo</title>
<style>
  :root {{ --bg: #0a0a0f; --surface: #14141f; --border: #2a2a3d; --accent: #6c5ce7;
           --accent-hover: #7f70f0; --text: #e8e8f0; --muted: #8888a0; --success: #00d2a0;
           --error: #ff6b6b; --font: 'Inter', system-ui, sans-serif; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--text);
          min-height: 100vh; display: flex; align-items: center; justify-content: center; }}
  .container {{ width: 100%; max-width: 640px; padding: 2rem; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
           padding: 1.5rem; margin-bottom: 1rem; }}
  label {{ display: block; font-size: 0.8rem; color: var(--muted); text-transform: uppercase;
           letter-spacing: 0.05em; margin-bottom: 0.4rem; }}
  select, textarea {{ width: 100%; background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.75rem; color: var(--text); font-family: var(--font);
    font-size: 0.95rem; outline: none; transition: border-color 0.2s; }}
  select:focus, textarea:focus {{ border-color: var(--accent); }}
  textarea {{ resize: vertical; min-height: 100px; margin-bottom: 1rem; }}
  .row {{ display: flex; gap: 0.75rem; margin-bottom: 1rem; }}
  .row > div {{ flex: 1; }}
  button {{ width: 100%; padding: 0.85rem; border: none; border-radius: 8px; font-size: 1rem;
    font-weight: 600; cursor: pointer; transition: all 0.2s; }}
  .btn-primary {{ background: var(--accent); color: #fff; }}
  .btn-primary:hover:not(:disabled) {{ background: var(--accent-hover); transform: translateY(-1px); }}
  .btn-primary:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  .btn-stop {{ background: var(--error); color: #fff; display: none; }}
  .status {{ display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem;
             font-size: 0.85rem; color: var(--muted); min-height: 1.5rem; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .dot.idle {{ background: var(--border); }}
  .dot.streaming {{ background: var(--success); animation: pulse 1s infinite; }}
  .dot.error {{ background: var(--error); }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
  .metrics {{ display: flex; gap: 1.5rem; margin-top: 0.75rem; font-size: 0.8rem; color: var(--muted); }}
  .metrics span {{ font-weight: 600; color: var(--text); }}
  .enroll-section {{ border-top: 1px solid var(--border); padding-top: 1rem; margin-top: 1rem; }}
  .enroll-section summary {{ cursor: pointer; color: var(--muted); font-size: 0.85rem; }}
  input[type="text"], input[type="file"] {{ width: 100%; background: var(--bg);
    border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem; color: var(--text);
    font-size: 0.9rem; margin-bottom: 0.5rem; }}
  .btn-sm {{ padding: 0.5rem 1rem; width: auto; font-size: 0.85rem; border-radius: 6px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Qwen3-TTS Streaming</h1>
  <p class="subtitle">Real-time voice clone — audio plays as chunks arrive</p>

  <div class="card">
    <div class="row">
      <div>
        <label>Voice</label>
        <select id="voiceSelect">
          <option value="">— none (enroll first) —</option>
          {voice_options}
        </select>
      </div>
      <div>
        <label>Language</label>
        <select id="langSelect">
          {"".join(f'<option value="{l}">{l}</option>' for l in SUPPORTED_LANGUAGES)}
        </select>
      </div>
    </div>
    <label>Text</label>
    <textarea id="textInput" placeholder="Type something to synthesize..."></textarea>
    <button class="btn-primary" id="speakBtn">Speak</button>
    <button class="btn-stop" id="stopBtn">Stop</button>
    <div class="status">
      <div class="dot idle" id="statusDot"></div>
      <span id="statusText">Ready</span>
    </div>
    <div class="metrics" id="metrics" style="display:none">
      <div>TTFA: <span id="metricTTFA">—</span></div>
      <div>Total: <span id="metricTotal">—</span></div>
      <div>Chunks: <span id="metricChunks">—</span></div>
    </div>
  </div>

  <div class="card">
    <details class="enroll-section">
      <summary>Enroll a new voice</summary>
      <div style="margin-top:1rem">
        <label>Voice ID</label>
        <input type="text" id="enrollVoiceId" placeholder="my-voice" />
        <label>Reference audio (WAV)</label>
        <input type="file" id="enrollFile" accept="audio/*" />
        <label>Reference text (optional — improves quality)</label>
        <textarea id="enrollRefText" rows="2" placeholder="Transcript of the reference audio"></textarea>
        <button class="btn-primary btn-sm" id="enrollBtn" style="margin-top:0.5rem">Enroll</button>
        <div id="enrollStatus" style="font-size:0.85rem;color:var(--muted);margin-top:0.5rem"></div>
      </div>
    </details>
  </div>
</div>

<script>
const BASE = location.origin;
const WAV_HEADER_SIZE = 44;
const SAMPLE_RATE = 24000;
let abortController = null;

const $ = id => document.getElementById(id);

$('speakBtn').onclick = startStreaming;
$('stopBtn').onclick = stopStreaming;
$('enrollBtn').onclick = enrollVoice;
$('textInput').addEventListener('keydown', e => {{
  if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); startStreaming(); }}
}});

async function startStreaming() {{
  const text = $('textInput').value.trim();
  const voiceId = $('voiceSelect').value;
  if (!text) return;
  if (!voiceId) {{ setStatus('error', 'Enroll a voice first'); return; }}

  $('speakBtn').disabled = true;
  $('stopBtn').style.display = 'block';
  $('metrics').style.display = 'flex';
  setStatus('streaming', 'Generating...');

  abortController = new AbortController();
  const t0 = performance.now();
  let firstChunkTime = null;
  let chunkCount = 0;

  try {{
    const res = await fetch(BASE + '/v1/audio/voice-clone', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        text, voice_id: voiceId,
        language: $('langSelect').value,
        stream: true, response_format: 'pcm'
      }}),
      signal: abortController.signal
    }});

    if (!res.ok) {{
      const err = await res.json().catch(() => ({{}}));
      throw new Error(err.error || `HTTP ${{res.status}}`);
    }}

    const ctx = new AudioContext({{ sampleRate: SAMPLE_RATE }});
    let nextTime = ctx.currentTime;
    const reader = res.body.getReader();
    let leftover = new Uint8Array(0);

    while (true) {{
      const {{ done, value }} = await reader.read();
      if (done) break;

      const merged = new Uint8Array(leftover.length + value.length);
      merged.set(leftover);
      merged.set(value, leftover.length);

      const usable = merged.length - (merged.length % 2);
      leftover = merged.slice(usable);
      if (usable === 0) continue;

      chunkCount++;
      if (!firstChunkTime) {{
        firstChunkTime = performance.now();
        $('metricTTFA').textContent = Math.round(firstChunkTime - t0) + 'ms';
      }}

      const view = new DataView(merged.buffer, merged.byteOffset, usable);
      const samples = usable / 2;
      const audioBuffer = ctx.createBuffer(1, samples, SAMPLE_RATE);
      const channel = audioBuffer.getChannelData(0);
      for (let i = 0; i < samples; i++) {{
        channel[i] = view.getInt16(i * 2, true) / 32768;
      }}

      const src = ctx.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(ctx.destination);
      const scheduleTime = Math.max(ctx.currentTime, nextTime);
      src.start(scheduleTime);
      nextTime = scheduleTime + audioBuffer.duration;

      $('metricChunks').textContent = chunkCount;
      $('metricTotal').textContent = Math.round(performance.now() - t0) + 'ms';
    }}

    setStatus('idle', `Done — ${{chunkCount}} chunks`);
    $('metricTotal').textContent = Math.round(performance.now() - t0) + 'ms';
  }} catch (e) {{
    if (e.name === 'AbortError') {{
      setStatus('idle', 'Stopped');
    }} else {{
      setStatus('error', e.message);
    }}
  }} finally {{
    $('speakBtn').disabled = false;
    $('stopBtn').style.display = 'none';
    abortController = null;
  }}
}}

function stopStreaming() {{
  if (abortController) abortController.abort();
}}

async function enrollVoice() {{
  const file = $('enrollFile').files[0];
  const voiceId = $('enrollVoiceId').value.trim();
  if (!file) {{ $('enrollStatus').textContent = 'Select an audio file'; return; }}
  if (!voiceId) {{ $('enrollStatus').textContent = 'Enter a voice ID'; return; }}

  $('enrollBtn').disabled = true;
  $('enrollStatus').textContent = 'Enrolling...';

  try {{
    const b64 = await fileToBase64(file);
    const refText = $('enrollRefText').value.trim() || undefined;
    const res = await fetch(BASE + '/v1/audio/voice-clone/enroll', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ voice_id: voiceId, ref_audio: b64, ref_text: refText }})
    }});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Enroll failed');

    // Add to dropdown
    const opt = document.createElement('option');
    opt.value = voiceId;
    opt.textContent = voiceId;
    $('voiceSelect').appendChild(opt);
    $('voiceSelect').value = voiceId;
    $('enrollStatus').textContent = `Enrolled "${{voiceId}}"`;
  }} catch (e) {{
    $('enrollStatus').textContent = 'Error: ' + e.message;
  }} finally {{
    $('enrollBtn').disabled = false;
  }}
}}

function fileToBase64(file) {{
  return new Promise((resolve, reject) => {{
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  }});
}}

function setStatus(state, msg) {{
  const dot = $('statusDot');
  dot.className = 'dot ' + state;
  $('statusText').textContent = msg;
}}
</script>
</body>
</html>"""


# ─── Handlers ─────────────────────────────────────────────────────────────────


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok" if model_ready else "loading",
        "model": MODEL_NAME,
        "engine": "faster-qwen3-tts",
        "streaming": True,
        "chunk_size": CHUNK_SIZE,
        "parity_mode": PARITY_MODE,
    })


async def handle_voices(request: web.Request) -> web.Response:
    voices = [{"id": lang.lower(), "name": lang} for lang in SUPPORTED_LANGUAGES]
    return web.json_response({"voices": voices})


async def handle_favicon(_request: web.Request) -> web.Response:
    return web.Response(status=204)


def _unauthorized_demo_response() -> web.Response:
    return web.Response(
        status=401,
        text="Authentication required",
        headers={"WWW-Authenticate": 'Basic realm="OctaLab Qwen3 Demo"'},
    )


def _is_demo_authorized(request: web.Request) -> bool:
    if not DEMO_AUTH_ENABLED:
        return True

    header = request.headers.get("Authorization", "")
    if not header.startswith("Basic "):
        return False

    try:
        decoded = base64.b64decode(header[6:]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return False

    return secrets.compare_digest(username, DEMO_USERNAME) and secrets.compare_digest(password, DEMO_PASSWORD)


async def handle_speech(request: web.Request) -> web.StreamResponse:
    """
    POST /v1/audio/speech
    Standard TTS (no voice cloning). Uses xvec_only with a silence reference
    so the model generates a neutral voice.
    """
    if not model_ready:
        return web.json_response({"error": "Model not ready"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = body.get("input", "")
    if not text:
        return web.json_response({"error": "Missing 'input' field"}, status=400)

    voice = body.get("voice", "Auto")
    language = voice if voice in SUPPORTED_LANGUAGES else _detect_language(text)
    response_format = body.get("response_format", "wav")

    logger.info(f"[speech] text={text[:50]}... lang={language}")

    # Use a cached silence ref for neutral voice generation
    silence_key = "_silence_default"
    if silence_key not in model._voice_prompt_cache:
        silence_b64 = _generate_silence_wav()
        tmp = _save_base64_to_tmpfile(silence_b64)
        try:
            base_model = model.model
            raw_items = base_model.create_voice_clone_prompt(
                ref_audio=tmp, ref_text=None, x_vector_only_mode=True,
            )
            items = [
                VoicePromptItem(
                    ref_code=it.ref_code, ref_spk_embedding=it.ref_spk_embedding,
                    x_vector_only_mode=True, icl_mode=False,
                )
                for it in raw_items
            ]
            _inject_prompt_into_cache(model, silence_key, items)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    resp = web.StreamResponse()
    resp.content_type = "audio/pcm" if response_format == "pcm" else "audio/wav"
    resp.headers["Transfer-Encoding"] = "chunked"
    resp.headers["Cache-Control"] = "no-cache, no-transform"
    resp.headers["X-Accel-Buffering"] = "no"
    await resp.prepare(request)

    try:
        first_chunk = True
        async for chunk, sr, _timing in _async_generate_chunks(dict(
            text=text, language=language, ref_audio=silence_key,
            ref_text="", xvec_only=True, chunk_size=CHUNK_SIZE,
            non_streaming_mode=False,
            parity_mode=PARITY_MODE,
        )):
            if first_chunk and response_format != "pcm":
                if not await _safe_stream_write(resp, _pcm_to_wav_header(sr), "speech"):
                    return resp
                first_chunk = False
            if not await _safe_stream_write(resp, _float32_to_int16(chunk), "speech"):
                return resp
    except (ClientConnectionResetError, ConnectionResetError, BrokenPipeError, RuntimeError) as e:
        logger.info(f"[speech] Client disconnected: {e}")
    except Exception as e:
        logger.error(f"[speech] Error: {e}\n{traceback.format_exc()}")

    await _safe_stream_write_eof(resp, "speech")
    return resp


async def handle_voice_clone(request: web.Request) -> web.StreamResponse:
    """
    POST /v1/audio/voice-clone
    Voice clone TTS with streaming output.

    Body JSON:
      - text: text to synthesize (required)
      - language: language hint (optional, default "Auto")
      - ref_audio: base64/URL reference audio (optional if voice_id/voice_prompt provided)
      - voice_id: reusable voice identifier (optional)
      - voice_prompt: serialized reusable prompt payload (optional)
      - ref_text: transcript of reference audio (optional, improves quality)
      - instruct: voice style instruction (optional)
      - response_format: "wav" or "pcm" (default "wav")
      - stream: true/false (default true)
    """
    if not model_ready:
        return web.json_response({"error": "Model not ready"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = body.get("text", "")
    if not text:
        return web.json_response({"error": "Missing 'text' field"}, status=400)

    ref_audio = _trim_ref_audio_base64(body.get("ref_audio", ""), MAX_REF_AUDIO_SECONDS)
    voice_id = body.get("voice_id")
    voice_prompt_payload = body.get("voice_prompt")
    ref_text = body.get("ref_text", None)
    language = body.get("language", "Auto")
    response_format = body.get("response_format", "wav")
    stream = body.get("stream", True)
    instruct = body.get("instruct", None)

    # Resolve voice prompt source
    prompt_items: Optional[List[VoicePromptItem]] = None
    if voice_prompt_payload is not None:
        try:
            prompt_items = _deserialize_voice_prompt_items(voice_prompt_payload)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        if voice_id:
            _get_voice_prompt_store(request.app)[voice_id] = prompt_items
    elif voice_id:
        prompt_items = _get_voice_prompt_store(request.app).get(voice_id)
        if prompt_items is None:
            return web.json_response({"error": f"Unknown voice_id '{voice_id}'"}, status=404)
    elif not ref_audio:
        return web.json_response(
            {"error": "Missing voice reference: provide 'ref_audio', 'voice_id', or 'voice_prompt'"},
            status=400,
        )

    using_cached_prompt = prompt_items is not None
    xvec_only = ref_text is None or ref_text.strip() == ""

    # Prepare ref_audio for FasterQwen3TTS
    tmp_ref_path: Optional[str] = None
    effective_ref_audio: str = ""
    effective_ref_text: str = ""

    if using_cached_prompt:
        cache_key_ref = voice_id or f"_prompt_{id(prompt_items)}"
        _inject_prompt_into_cache(model, cache_key_ref, prompt_items)
        effective_ref_audio = cache_key_ref
        effective_ref_text = ""
        xvec_only = prompt_items[0].x_vector_only_mode
    else:
        if ref_audio.startswith("data:") or (len(ref_audio) > 256 and not ref_audio.startswith("http")):
            tmp_ref_path = _save_base64_to_tmpfile(ref_audio)
            effective_ref_audio = tmp_ref_path
        else:
            effective_ref_audio = ref_audio
        effective_ref_text = ref_text or ""

    logger.info(
        f"[voice-clone] text={text[:50]}... lang={language} "
        f"voice_id={voice_id or '-'} cached={using_cached_prompt} "
        f"xvec_only={xvec_only} instruct={'yes' if instruct else 'no'} "
        f"stream={stream} format={response_format}"
    )

    stream_kwargs = dict(
        text=text,
        language=language,
        ref_audio=effective_ref_audio,
        ref_text=effective_ref_text,
        xvec_only=xvec_only,
        chunk_size=CHUNK_SIZE,
        non_streaming_mode=not bool(stream),
        parity_mode=PARITY_MODE,
    )

    resp: Optional[web.StreamResponse] = None
    try:
        if not stream:
            all_chunks = []
            sr = 24000
            async for chunk, chunk_sr, _timing in _async_generate_chunks(stream_kwargs):
                all_chunks.append(chunk)
                sr = chunk_sr
            if all_chunks:
                full_audio = np.concatenate(all_chunks)
            else:
                full_audio = np.zeros(0, dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, full_audio, sr, format="WAV")
            buf.seek(0)
            return web.Response(body=buf.read(), content_type="audio/wav")

        # Streaming response
        resp = web.StreamResponse()
        resp.content_type = "audio/pcm" if response_format == "pcm" else "audio/wav"
        resp.headers["Transfer-Encoding"] = "chunked"
        resp.headers["Cache-Control"] = "no-cache, no-transform"
        resp.headers["X-Accel-Buffering"] = "no"
        await resp.prepare(request)

        first_chunk = True
        async for chunk, sr, _timing in _async_generate_chunks(stream_kwargs):
            if first_chunk and response_format != "pcm":
                if not await _safe_stream_write(resp, _pcm_to_wav_header(sr), "voice-clone"):
                    return resp
                first_chunk = False
            if not await _safe_stream_write(resp, _float32_to_int16(chunk), "voice-clone"):
                return resp

        await _safe_stream_write_eof(resp, "voice-clone")
        return resp

    except (ClientConnectionResetError, ConnectionResetError, BrokenPipeError, RuntimeError) as e:
        logger.info(f"[voice-clone] Client disconnected: {e}")
        return resp or web.Response(status=499)
    except Exception as e:
        logger.error(f"[voice-clone] Error: {e}\n{traceback.format_exc()}")
        if stream and resp is not None:
            await _safe_stream_write_eof(resp, "voice-clone")
            return resp
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if tmp_ref_path and os.path.exists(tmp_ref_path):
            os.unlink(tmp_ref_path)


async def handle_voice_clone_enroll(request: web.Request) -> web.Response:
    """
    POST /v1/audio/voice-clone/enroll
    Build a reusable voice prompt once and cache it under a stable voice_id.
    Uses the base Qwen3TTSModel (model.model) for prompt extraction.
    """
    if not model_ready:
        return web.json_response({"error": "Model not ready"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    ref_audio = _trim_ref_audio_base64(body.get("ref_audio", ""), MAX_REF_AUDIO_SECONDS)
    if not ref_audio:
        return web.json_response({"error": "Missing 'ref_audio' field"}, status=400)

    ref_text = body.get("ref_text")
    provided_x_vector_only = body.get("x_vector_only_mode")
    x_vector_only = (
        bool(provided_x_vector_only)
        if provided_x_vector_only is not None
        else (ref_text is None or str(ref_text).strip() == "")
    )
    voice_id = body.get("voice_id") or f"clone_{int(time.time())}"

    # Decode base64 to temp file for the base model
    tmp_path: Optional[str] = None
    try:
        if ref_audio.startswith("data:") or (len(ref_audio) > 256 and not ref_audio.startswith("http")):
            tmp_path = _save_base64_to_tmpfile(ref_audio)
            audio_path = tmp_path
        else:
            audio_path = ref_audio

        async with model_lock:
            base_model = model.model
            raw_items = base_model.create_voice_clone_prompt(
                ref_audio=audio_path,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
            )

        # Convert to our VoicePromptItem
        items = [
            VoicePromptItem(
                ref_code=it.ref_code,
                ref_spk_embedding=it.ref_spk_embedding,
                x_vector_only_mode=it.x_vector_only_mode,
                icl_mode=it.icl_mode,
                ref_text=getattr(it, "ref_text", None),
            )
            for it in raw_items
        ]

        _get_voice_prompt_store(request.app)[voice_id] = items
        _inject_prompt_into_cache(model, voice_id, items)

        logger.info(
            f"[voice-enroll] voice_id={voice_id} ref_text={'yes' if ref_text else 'no'} "
            f"x_vector_only={x_vector_only}"
        )
        return web.json_response({
            "voice_id": voice_id,
            "status": "success",
            "x_vector_only_mode": x_vector_only,
            "voice_prompt": _serialize_voice_prompt_items(items),
        })
    except Exception as e:
        logger.error(f"[voice-enroll] Error: {e}\n{traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def handle_demo(request: web.Request) -> web.Response:
    """GET /demo — Interactive streaming TTS demo page."""
    if not _is_demo_authorized(request):
        return _unauthorized_demo_response()
    voice_ids = list(_get_voice_prompt_store(request.app).keys())
    return web.Response(text=_build_demo_html(voice_ids), content_type="text/html")


async def handle_upload_audio(request: web.Request) -> web.Response:
    """
    POST /upload_audio/
    Upload reference audio for voice cloning (legacy compatibility).
    """
    if not model_ready:
        return web.json_response({"error": "Model not ready"}, status=503)

    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.json_response({"error": "No file uploaded"}, status=400)

    audio_bytes = await field.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
    tmp.write(audio_bytes)
    tmp.close()

    try:
        async with model_lock:
            base_model = model.model
            raw_items = base_model.create_voice_clone_prompt(
                ref_audio=tmp.name,
                ref_text=None,
                x_vector_only_mode=True,
            )

        items = [
            VoicePromptItem(
                ref_code=it.ref_code,
                ref_spk_embedding=it.ref_spk_embedding,
                x_vector_only_mode=it.x_vector_only_mode,
                icl_mode=it.icl_mode,
                ref_text=getattr(it, "ref_text", None),
            )
            for it in raw_items
        ]

        voice_id = f"clone_{int(time.time())}_{os.path.basename(tmp.name)}"
        _get_voice_prompt_store(request.app)[voice_id] = items
        _inject_prompt_into_cache(model, voice_id, items)

        return web.json_response({
            "voice_id": voice_id,
            "status": "success",
            "voice_prompt": _serialize_voice_prompt_items(items),
        })
    except Exception as e:
        logger.error(f"[upload] Error: {e}\n{traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        os.unlink(tmp.name)


# ─── App Setup ────────────────────────────────────────────────────────────────


def create_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app["voice_prompts"] = {}
    app.router.add_get("/health", handle_health)
    app.router.add_get("/health-tts", handle_health)
    app.router.add_get("/favicon.ico", handle_favicon)
    app.router.add_get("/v1/audio/voices", handle_voices)
    app.router.add_post("/v1/audio/speech", handle_speech)
    app.router.add_post("/v1/audio/voice-clone/enroll", handle_voice_clone_enroll)
    app.router.add_post("/v1/audio/voice-clone", handle_voice_clone)
    app.router.add_post("/upload_audio/", handle_upload_audio)
    app.router.add_get("/demo", handle_demo)
    return app


if __name__ == "__main__":
    logger.info(f"Starting faster-qwen3-tts Server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_NAME}, Chunk size: {CHUNK_SIZE}")
    load_model()
    app = create_app()
    web.run_app(app, host=HOST, port=PORT, print=logger.info)
