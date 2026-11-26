# AI Chemistry Video Request Service — Prototype Architecture

## 1. Scope and success criteria

This is a backend-only FastAPI prototype that accepts a learner's question, creates a short narrated chemistry explainer, and exposes job status plus the completed MP4 artifact.

The prototype must complete these questions end to end:

1. `How does the pH scale work?`
2. `Why do atoms form covalent bonds?`
3. `What is the difference between ionic and covalent bonding?`

The initial topic catalog is deliberately narrow. A topic registry will own question matching, chemistry validation rules, and deterministic storyboard templates so that another chemistry or STEM topic can be added without changing the job orchestration or storage layers.

Out of scope for this slice: a frontend, user accounts, distributed workers, cloud object storage, arbitrary user-authored animation code, a general-purpose scientific fact checker, and production-grade horizontal scaling.

## 2. HTTP and processing flow

The minimal API surface is:

- `POST /jobs` with `{ "question": "..." }`: persist a job in `queued` and return `202 Accepted` with its ID and status URL.
- `GET /jobs/{job_id}`: return the current state, timestamps, retry/fallback metadata, and a safe error object when applicable.
- `GET /jobs/{job_id}/artifact`: stream or download the MP4 only after the job is `completed`; otherwise return a conflict/not-ready response.
- `GET /health`: report process health and availability of required local tools such as FFmpeg/ffprobe.

For the prototype, a single in-process worker consumes persisted job IDs from an asynchronous queue. The repository is the source of truth; the queue is only a wake-up mechanism. On startup, a job left in a non-terminal state by an earlier process is transitioned to `failed` with `worker_interrupted`; the client may create a new job. This deliberately avoids duplicate TTS/render side effects and does not claim resumability in the short prototype slice. One process and one worker avoid multi-writer races. A production version would replace the in-process queue with a durable broker/worker system while retaining the same domain interfaces.

High-level flow:

```text
learner question
    -> persist queued job
    -> select topic rules
    -> plan storyboard
    -> schema + chemistry validation
         -> retry planner with structured errors
         -> deterministic topic template after retry exhaustion
    -> synthesize narration
    -> render visual frames and mux audio/video
    -> verify final MP4 with ffprobe
    -> publish artifact and mark completed
```

## 3. Job lifecycle state machine

Canonical states:

```text
queued
  -> planning
  -> validating_plan
       -> planning                 (corrective retry with validation feedback)
       -> synthesizing_audio       (validated candidate or validated fallback)
  -> rendering_visuals
  -> verifying_artifact
  -> completed

Any non-terminal state -> failed
```

| State | Responsibility | Successful transition |
| --- | --- | --- |
| `queued` | The request and job ID have been durably recorded. | `planning` |
| `planning` | Identify the supported topic and ask `StoryboardPlanner` for a schema-shaped candidate. On corrective attempts, include the prior structured validation errors. | `validating_plan` |
| `validating_plan` | Run strict storyboard schema checks and the selected topic's chemistry checks. Retry by returning to `planning`; after retry exhaustion, load and validate the deterministic fallback. | `planning` or `synthesizing_audio` |
| `synthesizing_audio` | Generate narration through `SpeechSynthesizer`, inspect the produced audio, and derive final scene timing from measured audio duration. | `rendering_visuals` |
| `rendering_visuals` | Render deterministic Pillow frames at 1280x720 and use FFmpeg to encode/mux H.264 video with AAC audio. | `verifying_artifact` |
| `verifying_artifact` | Run ffprobe and enforce the media contract before publishing the artifact. | `completed` |
| `completed` | Terminal success. The verified artifact is available. | none |
| `failed` | Terminal failure with the failing stage, stable error code, safe message, and diagnostic metadata recorded. | none |

State updates are explicit and persisted before work for the next state begins. Each update records `updated_at` and appends a small event containing the state, attempt number, and outcome. `completed` is written only after the verified MP4 has been atomically promoted to its final artifact key. A retry requested by a client creates a new job rather than mutating a terminal job.

The core job record uses `job_id`, `query`, `subject`, `topic_id`, `status`, `progress_pct`, planner `attempts`, `used_fallback`, safe error fields, an optional validated `storyboard`, artifact metadata, and UTC lifecycle timestamps. External responses expose only safe error details; command output and provider diagnostics remain local.

## 4. Domain boundaries

The domain/orchestration layer depends on Python `Protocol` interfaces, not concrete JSON, filesystem, planner-provider, or edge-tts implementations. Protocols exchange domain records and artifact keys rather than FastAPI request objects or provider-specific response objects.

### `JobRepository`

Owns job metadata and lifecycle persistence, but knows nothing about media bytes.

Required operations:

- create a new queued job and reject duplicate IDs;
- retrieve a job by ID;
- list non-terminal jobs at startup so interrupted work can be reconciled to `failed`;
- atomically transition an expected current state to the next state while updating fields/events;
- record terminal failure details.

The local implementation stores all job records in `data/runtime_jobs/jobs.json` and writes the complete document through a temporary file followed by an atomic replace. State transitions use an in-process lock plus expected-state comparison. A database implementation can later provide the same compare-and-set semantics.

### `ArtifactStore`

Owns intermediate and final blobs and their metadata, but does not decide job state.

Required operations:

- write/read a named artifact within a job namespace;
- provide a safe local staging path for tools that require filesystem input;
- report existence, size, media type, and checksum;
- atomically promote a verified staged video to its final key;
- remove disposable intermediates for a job without touching committed samples.

Keys are generated by the service, never accepted as raw user paths. The local implementation is rooted under `data/runtime_jobs/artifacts/{job_id}/`, separate from repository records. A future object-store implementation can download to a temporary staging area while preserving the protocol.

### `StoryboardPlanner`

Turns a supported learner question into an untrusted storyboard candidate. It does not validate its own output, synthesize speech, render frames, or mutate job state.

Its planning input contains the normalized question, selected topic ID, schema version, attempt number, and zero or more structured validation errors from the previous attempt. Its result is a candidate document plus planner metadata. This boundary supports a remote model over `httpx`, a test fake, or a deterministic planner without changing the pipeline.

### `SpeechSynthesizer`

Turns validated narration segments into an audio artifact plus measured metadata. It does not decide storyboard content, scene layout, or final video encoding.

Its input includes narration text/segments, language, voice, and speaking-rate settings. Its result identifies the generated audio and includes duration, format, and any timing data available. The initial adapter uses `edge-tts`; a later offline or paid provider can implement the same protocol.

### Deliberately concrete services in this slice

Storyboard validation, topic-rule lookup, Pillow frame rendering, FFmpeg invocation, and ffprobe verification are deterministic application services in the prototype. They remain separate modules, but additional protocols are deferred until a second implementation is justified. The orchestrator alone coordinates them and owns state transitions.

## 5. Storyboard contract and pre-render safeguards

Planner output is untrusted and cannot reach Pillow, the filesystem, or FFmpeg until it passes both validation layers.

### Strict schema validation

The current Pydantic storyboard schema includes:

- `topic_id`, `subject`, `title`, and the original `learner_query`;
- an ordered list of three to five scenes with contiguous indexes;
- for each scene: heading, narration, caption, an allowlisted `VisualTemplate`, and JSON-only template parameters;
- bounded text lengths and forbidden unknown model fields.

The topic definition supplies the allowed template set, required concept synonym groups, and forbidden misconceptions used by semantic validation. Template-specific validation must reject unapproved URLs, file paths, HTML, executable expressions, invalid colors, and unsafe coordinates before rendering. Schema versioning, structured claim objects, and explicit timing fields can be added when a second planner contract requires them.

### Chemistry concept validation

The selected topic module validates structured claims, narration, labels, comparisons, numerical values, and important visual relationships. Each error has a stable code, field/scene location, and actionable message. The initial rule packs enforce at least these ideas:

- **pH scale:** pH is logarithmic and commonly expressed as `-log10[H+]`; one pH unit represents a tenfold hydrogen-ion concentration change; at about 25 °C neutral water is pH 7, with lower values acidic and higher values basic. The validator rejects the common overclaim that pH is always restricted to 0–14.
- **covalent bonding:** atoms share electron density because the bonded arrangement can be lower in energy; attraction between nuclei and shared electrons matters. The octet rule is presented as a useful heuristic with exceptions, not as intent or a universal law.
- **ionic versus covalent:** ionic models emphasize ions and electrostatic attraction (often in extended lattices), while covalent models emphasize shared electron density. The validator requires acknowledgement that bonding lies on a continuum and rejects universal property claims such as “all ionic compounds dissolve” or “all covalent substances have low melting points.”

This curated validator is a guardrail for the supported topics, not a general scientific proof system. Extending the service means registering a topic matcher, claim/rule pack, and reviewed fallback storyboard together.

### Error-feedback retry and deterministic fallback

The primary planner receives one initial attempt and one corrective attempt. After the first candidate, schema errors and chemistry errors are collected together and returned unchanged as structured feedback on the second `planning` pass. Invalid candidate text is retained only for local diagnostics and is never rendered.

If both primary attempts fail, the implementation selects a reviewed, version-controlled fallback storyboard for the matched topic. The fallback is recorded as the third observable attempt and is passed through the exact same schema and chemistry validators. If it fails, the job becomes `failed` with an internal configuration error; validation is never bypassed. The job record exposes its attempts and `used_fallback`, making the behavior testable and observable.

The implementation must ship one reviewed fallback for each of the three required questions, so an unavailable or malformed planner cannot prevent the core demonstration from completing. Future topics are not declared supported until they supply the same matcher/rules/fallback bundle.

## 6. Audio, visuals, and encoding

The validated scene narrations are synthesized independently. The primary adapter uses edge-tts with `en-US-AriaNeural` and a 12-second per-scene timeout. If online synthesis fails, the service uses Windows SAPI when available and finally a deterministic pure-Python WAV generator, so offline and CI runs still produce timed media. Every produced file is inspected with ffprobe; its measured duration plus 0.4 seconds determines the corresponding scene length rather than trusting a planner estimate.

Pillow renders crisp 1280x720 frames from the allowlisted visual primitives and streams RGB24 bytes directly to FFmpeg stdin; individual frame files are never written to runtime storage. Motion uses smooth markers, particles, orbital/energy diagrams, arrows, progressive cards, and diagram transitions while keeping text legible. Fonts and other renderer assets are service-owned and resolved from a controlled fallback list.

FFmpeg receives an argument array rather than an interpolated shell command and produces:

- MP4 container with `+faststart`;
- 1280x720 H.264 video, `yuv420p`, 20 fps;
- quality-based encoding with `libx264 -preset fast -crf 24`;
- AAC audio at 128 kbps;
- a target artifact size of roughly 1–5 MiB for these short, mostly graphical explainers, with a 6 MiB hard ceiling for the prototype.

These choices favor universal playback, crisp diagrams, and small artifacts over photorealism, complex animation, or exact byte-for-byte reproducibility across FFmpeg versions.

## 7. Post-render verification with ffprobe

FFmpeg exiting successfully is not sufficient. In `verifying_artifact`, ffprobe runs with JSON output against the staged MP4. Verification fails unless all of the following hold:

- ffprobe exits successfully and reports a readable, non-empty container;
- at least one usable video stream is H.264 at 1280x720 with a browser-compatible pixel format;
- at least one usable audio stream is AAC with a positive sample rate and channel count;
- the container duration is at least 12 seconds and within 3.5 seconds of the expected rendered duration;
- file size is between 30 KiB and 6 MiB inclusive;
- the final file can be opened from the artifact staging path and has an MP4-compatible format name.

The parsed ffprobe summary, file size, and SHA-256 checksum are recorded with the job. Only a passing artifact is atomically promoted and exposed by the download endpoint. A failed verification records the check and moves the job to `failed`; a corrupt or partial file is never published as completed.

## 8. Local persistence layout

```text
data/runtime_jobs/jobs.json

data/runtime_jobs/artifacts/{job_id}/
  storyboard.json
  audio/
    scene-01.mp3 (or .wav after offline fallback)
  explanation.staged.mp4
  ffprobe.json
  explanation.mp4

samples/
  ...three reviewed demonstration videos, one per required query...
```

`data/runtime_jobs/` is disposable runtime state and is ignored by Git. `samples/` is intentionally not ignored so the three best generated videos can be reviewed, committed, and included in a zip. Runtime cleanup is confined to the artifact-store root and never removes `samples/`.

## 9. Failure handling and observability

Failures use stable codes such as `unsupported_topic`, `worker_interrupted`, `plan_schema_invalid`, `plan_concept_invalid`, `tts_unavailable`, `render_failed`, and `artifact_verification_failed`. The job record includes the failing state and retry information. Logs include the job ID, state, elapsed time, planner attempt, fallback usage, FFmpeg exit status, and verification results, but never secrets or unsanitized command strings.

Transient planner or TTS transport failures may receive a small bounded retry with backoff inside their current stage. Content-correction retries remain the explicit `planning`/`validating_plan` loop. No retry is unbounded, and every failure path ends in a persisted terminal state.

## 10. Tradeoffs for the 90–120 minute slice

- **Backend only:** FastAPI endpoints and generated artifacts demonstrate the entire learner flow; no frontend is built.
- **Local persistence behind interfaces:** one JSON record and a job-scoped file directory minimize setup time, while `JobRepository` and `ArtifactStore` preserve migration paths to a database and object storage.
- **Single local worker:** enough for a reliable demo and straightforward lifecycle behavior; multi-process coordination, brokers, cancellation, quotas, and scheduling are deferred.
- **Narrow reviewed science scope:** the three required questions receive topic-specific validation and deterministic fallbacks. Broad STEM support is an extension mechanism, not an unsupported claim.
- **Pillow + resilient speech + FFmpeg:** Pillow makes diagrams/text predictable, edge-tts provides natural narration when reachable, SAPI/pure-Python WAV provides offline continuity, and FFmpeg provides efficient 1280x720 H.264/AAC output. The fallback waveform favors pipeline reliability over natural speech quality, and the animations remain intentionally focused rather than photorealistic.
- **Small, crisp video over elaborate visuals:** constrained primitives, limited motion, CRF encoding, and short scripts keep output legible and generally within a few megabytes.
- **ffprobe as a hard completion gate:** adds one local binary dependency and a little latency, but prevents missing-audio, corrupt, wrong-resolution, or oversized artifacts from being reported as successful.
- **Unpinned initial dependency list:** `requirements.txt` names the requested prototype dependencies for quick setup. After the first known-good environment is exercised, a lock or constraints file should capture exact versions for reproducible delivery.

System prerequisites such as FFmpeg/ffprobe and a usable font are checked at startup; they are not Python packages and therefore do not appear in `requirements.txt`.

## 11. Test strategy for the implementation phase

The later implementation should include:

- unit tests for legal/illegal state transitions and atomic repository updates;
- schema and chemistry-validator tests, including known misconceptions for all three topics;
- retry tests proving structured feedback is passed unchanged and the deterministic fallback is used after two failed primary candidates;
- protocol fakes for planner, TTS, repository, and artifact storage;
- renderer/ffprobe contract tests for correct streams, resolution, codecs, duration tolerance, and size ceiling;
- end-to-end tests that submit each required question, poll to a terminal state, and verify a downloadable MP4;
- failure tests proving invalid plans and corrupt media are never exposed as `completed`.
