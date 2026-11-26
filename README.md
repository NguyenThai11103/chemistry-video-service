# AI Chemistry Video Request Service

Backend-first prototype for turning a learner's chemistry question into a validated, narrated, animated MP4 explainer.

> Status: draft / work in progress. The domain, planning, validation, persistence, speech, and rendering layers are implemented. FastAPI routes and the end-to-end background job worker are not wired yet.

## Supported prototype topics

- “How does the pH scale work?”
- “Why do atoms form covalent bonds?”
- “What is the difference between ionic and covalent bonding?”

The topic registry is subject-scoped so physics, biology, and other STEM modules can be added without changing the core job pipeline.

## What is implemented

- Pydantic job and storyboard domain models.
- Extensible chemistry topic registry and deterministic intent matching.
- Curated storyboards plus optional Gemini/OpenAI JSON planning.
- Validation retries, misconception checks, and a validated deterministic fallback.
- Atomic JSON job persistence and safe per-job artifact workspaces.
- Edge TTS with Windows SAPI and deterministic pure-Python WAV fallbacks.
- Pillow animations for pH, covalent bonding, ionic-versus-covalent bonding, and summary cards.
- Direct RGB24 frame streaming to FFmpeg at 1280×720 and 20 fps.
- H.264/AAC output verification through ffprobe.

See [ARCHITECTURE.md](ARCHITECTURE.md) for lifecycle, boundaries, safeguards, and prototype tradeoffs.

## Local setup

Requirements:

- Python 3.11+
- FFmpeg and ffprobe available on `PATH`
- Network access only if using Edge TTS or an optional LLM planner

PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Optional planner credentials are read from `GEMINI_API_KEY` or `OPENAI_API_KEY`. Set them in the shell environment before starting the service; never commit credentials. The repository does not require either key because curated planning and offline audio fallbacks are included.

## Current verification

- 15 automated tests.
- Curated plans validated for all three required learner queries.
- Offline WAV, SAPI, and Edge TTS paths probed successfully.
- Real four-template render verified as H.264 1280×720 with AAC audio.

## Project layout

```text
app/
  models.py       Domain and API schemas
  topics.py       Subject/topic registry
  planner.py      Curated and optional LLM planning
  validator.py    Storyboard and media quality gates
  store.py        JSON and in-memory job repositories
  artifacts.py    Local artifact storage
  tts.py          Resilient speech synthesis
  renderer.py     Pillow/FFmpeg animation pipeline
tests/
  test_reliability.py
  test_media_pipeline.py
```

Runtime jobs are written under `data/runtime_jobs/` and ignored by Git. Reviewed demonstration videos may later be committed under `samples/`.
