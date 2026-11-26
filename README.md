# AI Chemistry Video Request Service

Backend-first FastAPI prototype that turns one of three supported chemistry questions into a validated, narrated, animated MP4 explainer. Jobs are queued in-process, persisted locally, and exposed through create, poll, list, filter, and download endpoints.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the lifecycle, component boundaries, validation safeguards, media contract, and design tradeoffs.

## Prerequisites and setup

- Python 3.11 or newer
- FFmpeg and ffprobe on `PATH`
- Network access for the preferred Edge TTS voice; optional LLM planning also requires network access and `GEMINI_API_KEY` or `OPENAI_API_KEY`

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

Start the API from the repository root:

```powershell
uvicorn app.main:app --reload
```

Check readiness at `http://127.0.0.1:8000/health`. Runtime job records and artifacts are written under `data/runtime_jobs/`.

## Swagger UI

Open `http://127.0.0.1:8000/docs`, expand `POST /api/v1/videos`, select **Try it out**, enter a supported question, and execute it. Use the returned `status_url` with `GET /api/v1/videos/{job_id}` until `status` is `completed`; then open `artifact_url` or use the download endpoint.

## API examples

The following PowerShell commands use the real curl executable (not the `curl` alias):

```powershell
# Create a job (HTTP 202) and retain its response.
$create = curl.exe -sS -X POST "http://127.0.0.1:8000/api/v1/videos" `
  -H "Content-Type: application/json" `
  -d '{"question":"How does the pH scale work?","subject":"chemistry"}' |
  ConvertFrom-Json
$create

# Poll this command until status is completed or failed.
curl.exe -sS $create.status_url

# List all jobs, newest first.
curl.exe -sS "http://127.0.0.1:8000/api/v1/videos"

# Filter by any lifecycle status, for example completed.
curl.exe -sS "http://127.0.0.1:8000/api/v1/videos?status=completed"

# Download the verified MP4 after completion.
curl.exe -fL "http://127.0.0.1:8000/api/v1/videos/$($create.job_id)/video" `
  -o explanation.mp4
```

Valid filter values are `queued`, `planning`, `validating_plan`, `synthesizing_audio`, `rendering_visuals`, `verifying_artifact`, `completed`, and `failed`. Downloading before completion returns HTTP 409.

## Required samples

Generate or refresh all three sample targets with:

```powershell
python scripts/generate_all.py
```

The committed sample set is:

| Required learner query | Generated sample |
| --- | --- |
| `How does the pH scale work?` | [samples/01_ph_scale.mp4](samples/01_ph_scale.mp4) |
| `Why do atoms form covalent bonds?` | [samples/02_covalent_bonds.mp4](samples/02_covalent_bonds.mp4) |
| `What is the difference between ionic and covalent bonding?` | [samples/03_ionic_vs_covalent.mp4](samples/03_ionic_vs_covalent.mp4) |

[samples/manifest.json](samples/manifest.json) records each exact query, output path, duration, byte size, and validation result. After every video passes ffprobe verification, the generator publishes the batch with per-file atomic replacement, rollback on failure, and the manifest written last as its commit marker.

## Speech synthesis behavior

For each scene, the service first calls Edge TTS (`en-US-AriaNeural`), which requires outbound network access and has a 12-second timeout. If it fails, Windows runs the built-in `System.Speech` SAPI engine through PowerShell and writes an offline WAV. If SAPI is unavailable or fails, the API pipeline can use a portable deterministic WAV generator for CI and degraded operation; that last fallback is not intelligible natural speech. The committed-sample generator deliberately disables that final fallback, so it either produces natural Edge/SAPI narration or fails without publishing a partial batch. Audio duration is measured with ffprobe before rendering.

## Tests

```powershell
python -m pytest -q
```

## Prototype limitations

- Only the three questions above are supported by the curated topic registry and chemistry rules.
- Jobs run as background tasks inside one API process, with at most two active pipelines; there is no durable queue, crash recovery, or horizontal worker scaling.
- Persistence and artifacts are local files, not a database or object store.
- There is no frontend, authentication, authorization, rate limiting, or cloud deployment configuration.
- Curated validators are topic-specific guardrails, not a general chemistry fact checker.
- Optional model planning may use `GEMINI_API_KEY` or `OPENAI_API_KEY`, but neither key is required because curated planning and deterministic fallbacks are included. Never commit credentials.
