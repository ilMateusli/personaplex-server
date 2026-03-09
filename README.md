[![Runpod](https://api.runpod.io/badge/ilMateusli/personaplex-server)](https://console.runpod.io/hub/ilMateusli/personaplex-server)

# PersonaPlex + Qwen3-TTS Server

Serverless worker for Runpod Hub that boots two local services behind one container:

- PersonaPlex on `:8999` for real-time voice sessions over WebSocket
- Qwen3-TTS on `:8880` for speech synthesis and voice cloning
- Nginx on `:8998` as the shared reverse proxy

## Runpod handler

The serverless entrypoint lives in [handler.py](./handler.py) and starts the Runpod worker with:

```python
runpod.serverless.start({"handler": handler})
```

Supported `input.endpoint` values:

- `health`
- `personaplex-health`
- `tts-health`
- `voices`
- `speech`
- `voice-clone`

## Example payloads

Text-to-speech:

```json
{
  "input": {
    "endpoint": "speech",
    "text": "Testing standard text to speech synthesis.",
    "language": "English"
  }
}
```

Voice clone:

```json
{
  "input": {
    "endpoint": "voice-clone",
    "text": "Hello world",
    "language": "English",
    "ref_audio": "data:audio/wav;base64,...",
    "ref_text": "Reference transcript"
  }
}
```

## Hub files

- Hub metadata: [`.runpod/hub.json`](./.runpod/hub.json)
- Validation tests: [`.runpod/tests.json`](./.runpod/tests.json)
- Container image: [`Dockerfile`](./Dockerfile)
