# AI Chemistry Video Request Service - Architecture Note

## Scope

This backend-only FastAPI prototype accepts one of three required chemistry questions, runs an asynchronous generation job, and exposes a verified narrated MP4 for download. The supported questions are:

1. `How does the pH scale work?`
2. `Why do atoms form covalent bonds?`
3. `What is the difference between ionic and covalent bonding?`

The scope is deliberately narrow so the science and visuals can be reviewed and generated consistently. New STEM topics can be added through the topic registry without changing the HTTP, persistence, or artifact layers. A frontend, user accounts, distributed workers, cloud storage, and production horizontal scaling are out of scope.

## API and job lifecycle

The HTTP layer is intentionally thin:

- `POST /api/v1/videos` persists a `queued` job, schedules an in-process task, and returns `202 Accepted` with the job ID and polling URL.
- `GET /api/v1/videos` lists jobs newest-first and can filter by status.
- `GET /api/v1/videos/{job_id}` returns status, progress, timestamps, planner attempts, fallback use, safe errors, storyboard, and artifact metadata.
- `GET /api/v1/videos/{job_id}/video` downloads a completed MP4. It returns `404` for an unknown/missing artifact and `409` while a known job is not complete.
- `GET /health` reports whether `ffmpeg` and `ffprobe` are discoverable on `PATH`, returning `503` when either is missing. Media stages also detect invocation failures.

The persisted lifecycle is:

```text
queued -> planning -> validating_plan -> synthesizing_audio
       -> rendering_visuals -> verifying_artifact -> completed

any active state -> failed
```

The pipeline owns transitions and writes each state before beginning the next stage. `completed` is written only after ffprobe verification and promotion of the staged MP4. A stable error code and safe message are stored on failure. Planner retry details are stored as structured attempt records.

FastAPI retains strong references to background tasks, and the pipeline uses an `asyncio.Semaphore(2)` to bound complete job executions. On orderly shutdown, active tasks are cancelled; the pipeline catches cancellation and attempts to persist `WORKER_INTERRUPTED`. Hard crashes can leave a non-terminal record: startup reconciliation, resumability, a durable queue, and multi-process coordination are explicit prototype limitations. A client retries by creating a new job rather than mutating a terminal one.

## Boundaries and persistence

The orchestration layer depends on four small `Protocol` boundaries:

- `JobRepository` creates, reads, lists, compare-and-set updates, and transitions job records. The local implementation stores all records in `data/runtime_jobs/jobs.json`, guarded by an in-process lock and written through a temporary file plus atomic replace.
- `ArtifactStore` owns job-scoped paths, storyboard bytes, staged/final videos, metadata, and cleanup. Generated keys are service-controlled, and the local implementation is rooted at `data/runtime_jobs/artifacts/{job_id}/`.
- `StoryboardPlanner` receives the learner query, matched `TopicDefinition`, and optional validation issue strings from the previous attempt. Implementations include curated plans and optional Gemini/OpenAI JSON planners.
- `SpeechSynthesizer` receives narration text and a requested output path and returns the path it actually produced. Edge TTS, Windows SAPI, and deterministic WAV adapters sit behind this interface.

Topic matching, validation, Pillow rendering, FFmpeg encoding, and ffprobe verification are deterministic application services. The renderer and verifier are injectable for tests without adding protocols that the prototype does not need.

Runtime state is disposable and ignored by Git. Final demonstration artifacts are separate:

```text
samples/
  01_ph_scale.mp4
  02_covalent_bonds.mp4
  03_ionic_vs_covalent.mp4
  manifest.json
```

The manifest maps each exact learner query to its MP4, duration, byte size, and validation result. `scripts/generate_all.py` generates all three through the production pipeline before publishing. Publication uses per-file atomic replacement, rollback on failure, stable-path re-verification, and publishes the manifest last as the batch commit marker.

## Reliable planning and chemistry safeguards

Planner output is untrusted. A candidate cannot reach speech synthesis or rendering until it passes the strict Pydantic storyboard schema and topic-specific validation. Storyboards contain three to five ordered scenes with bounded text, contiguous indexes, allowlisted visual templates, and validated JSON parameters. Unknown fields, unsafe paths/URLs, raw markup, invalid coordinates, and unsupported template data are rejected.

The topic rules also check required concepts and known misconceptions. Examples include the logarithmic pH relationship and tenfold change per unit; electron-density sharing and lower-energy bonding for covalent bonds; and the distinction between electrostatic ionic attraction and shared electron density, while acknowledging a bonding continuum. These rules are reviewed guardrails for the three supported topics, not a general chemistry proof system.

Planning is bounded to two primary attempts. After a failed candidate, the exact validation issue strings are passed to the next attempt. Each attempt has a timeout and records provider, duration, issues, and pass/fail result. If both attempts fail, a reviewed curated plan is tried as a third, separately validated fallback. An invalid fallback fails the job with `PLANNING_FAILED`; validation is never bypassed. With no model API key, the curated planner is the primary as well, so the required scope remains deterministic and does not require network access for planning.

## Audio, rendering, and completion gate

Each validated scene narration is synthesized independently. The preferred adapter uses Edge TTS with a 12-second timeout, then falls back to Windows SAPI and, for the API/CI path, a deterministic WAV generator. Audio duration is measured before rendering. The deterministic fallback preserves pipeline continuity but is not natural intelligible speech; sample generation disables it so committed showcases contain Edge or SAPI narration.

Pillow creates 1280x720 RGB frames from four allowlisted chemistry visual templates and streams them to FFmpeg. FFmpeg produces a 20 fps H.264/yuv420p MP4 with AAC audio and `+faststart`. Commands use argument arrays with `shell=False`; planner text never becomes executable code or a filesystem path.

FFmpeg success alone does not complete a job. The verifier runs ffprobe and requires a readable MP4, H.264 video at 1280x720, AAC audio, at least 12 seconds of runtime, an acceptable duration tolerance, and a file size between 30 KiB and 6 MiB. Only a passing staged video is promoted. The job stores the verified duration and final byte size; SHA-256 is available from artifact-store metadata but is not part of the persisted job response. Verification failures store `MEDIA_VERIFICATION_FAILED` and never expose a partial video as completed.

## Tradeoffs and verification evidence

Local JSON/filesystem storage and in-process tasks keep setup and the learner flow clear within the challenge timebox, while the repository, artifact, planner, and speech boundaries show where a database, object store, durable worker, or different provider would fit. Curated content and deterministic rendering favor repeatability and educational clarity over broad topic support or photorealistic media. The hard ffprobe gate adds a local dependency and latency but prevents corrupt, silent, wrong-resolution, or oversized results from being published.

The checked-in tests cover lifecycle progress, two-job concurrency, unsupported topics, verification-before-publish, API create/list/filter/poll/conflict/download behavior, planner retry feedback and timeout handling, validated fallback behavior, misconceptions for all three topics, offline audio fallback, every visual template, and invalid visual parameters. The black-box HTTP lifecycle test exercises a genuine H.264/AAC pH artifact through real ffprobe; the other required topics are covered by curated-plan validation and committed pipeline-generated samples. All three sample MP4s are 1280x720, are recorded in `samples/manifest.json`, and are re-verified at their stable paths before the manifest is published.
