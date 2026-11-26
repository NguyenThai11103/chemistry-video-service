"""Generate the three reviewed showcase videos and their manifest.

This script intentionally uses the production pipeline with curated planning,
natural speech synthesis, the fixed 1280x720 renderer, and the real ffprobe
media gate.  All expensive work is staged below ``data/runtime_jobs``; the
public sample set is replaced only after the complete batch has passed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


# Make repository imports independent of the caller's current directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.artifacts import LocalArtifactStore  # noqa: E402
from app.models import JobStatus, VideoJob  # noqa: E402
from app.pipeline import VideoGenerationPipeline  # noqa: E402
from app.planner import (  # noqa: E402
    CuratedChemistryPlanner,
    ResilientPlannerOrchestrator,
)
from app.renderer import Renderer  # noqa: E402
from app.store import JsonFileJobRepository  # noqa: E402
from app.tts import (  # noqa: E402
    EdgeTTSSynthesizer,
    ResilientSpeechSynthesizer,
    WindowsSAPISynthesizer,
)
from app.validator import verify_rendered_video  # noqa: E402


RUNTIME_ROOT = REPOSITORY_ROOT / "data" / "runtime_jobs"
SAMPLES_ROOT = REPOSITORY_ROOT / "samples"
MANIFEST_PATH = SAMPLES_ROOT / "manifest.json"


@dataclass(frozen=True, slots=True)
class SampleSpec:
    learner_query: str
    output_file: str
    job_id: str


SAMPLE_SPECS = (
    SampleSpec(
        learner_query="How does the pH scale work?",
        output_file="samples/01_ph_scale.mp4",
        job_id="sample-01-ph-scale",
    ),
    SampleSpec(
        learner_query="Why do atoms form covalent bonds?",
        output_file="samples/02_covalent_bonds.mp4",
        job_id="sample-02-covalent-bonds",
    ),
    SampleSpec(
        learner_query="What is the difference between ionic and covalent bonding?",
        output_file="samples/03_ionic_vs_covalent.mp4",
        job_id="sample-03-ionic-vs-covalent",
    ),
)


class SampleGenerationError(RuntimeError):
    """Raised when a sample cannot be safely generated or published."""


def _require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise SampleGenerationError(
            f"Required media tool(s) not found on PATH: {joined}"
        )


def _write_manifest_source(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    path.write_text(document, encoding="utf-8", newline="\n")


def _prepare_atomic_copy(source: Path, destination: Path) -> Path:
    """Copy to a temporary peer that can be atomically replaced into place."""

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with (
            source.open("rb") as input_handle,
            os.fdopen(descriptor, "wb") as output_handle,
        ):
            descriptor = -1
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        return temporary
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _restore_target(backup: Path | None, destination: Path) -> None:
    if backup is None:
        destination.unlink(missing_ok=True)
        return
    replacement = _prepare_atomic_copy(backup, destination)
    try:
        os.replace(replacement, destination)
    finally:
        replacement.unlink(missing_ok=True)


def _publish_batch(
    generated: list[tuple[SampleSpec, Path, float]],
    manifest_source: Path,
    backup_root: Path,
) -> None:
    """Publish the complete batch, restoring the prior set on any failure."""

    destinations = [REPOSITORY_ROOT / spec.output_file for spec, _, _ in generated]
    destinations.append(MANIFEST_PATH)
    sources = [source for _, source, _ in generated]
    sources.append(manifest_source)

    SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=True)

    backups: dict[Path, Path | None] = {}
    prepared: dict[Path, Path] = {}
    commit_started = False
    try:
        # Complete every copy and backup before changing any public target.
        for index, (source, destination) in enumerate(
            zip(sources, destinations, strict=True), start=1
        ):
            if destination.exists() and not destination.is_file():
                raise SampleGenerationError(
                    f"Sample target is not a regular file: {destination}"
                )
            backup = None
            if destination.is_file():
                backup = backup_root / f"{index:02d}-{destination.name}"
                shutil.copy2(destination, backup)
            backups[destination] = backup
            prepared[destination] = _prepare_atomic_copy(source, destination)

        # Publish videos first, then verify the bytes at their stable paths.
        commit_started = True
        for spec, _, _ in generated:
            destination = REPOSITORY_ROOT / spec.output_file
            os.replace(prepared.pop(destination), destination)

        for spec, _, expected_duration in generated:
            destination = REPOSITORY_ROOT / spec.output_file
            verification = verify_rendered_video(destination, expected_duration)
            if not verification.passed:
                detail = "; ".join(verification.issues)
                raise SampleGenerationError(
                    f"Published verification failed for {spec.output_file}: {detail}"
                )

        # The manifest is the final commit marker for the complete sample set.
        os.replace(prepared.pop(MANIFEST_PATH), MANIFEST_PATH)
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        if commit_started:
            for destination in reversed(destinations):
                try:
                    _restore_target(backups[destination], destination)
                except Exception as rollback_error:  # pragma: no cover - OS failure
                    rollback_errors.append(f"{destination}: {rollback_error}")
        if rollback_errors and hasattr(publish_error, "add_note"):
            publish_error.add_note(
                "Rollback also failed for: " + "; ".join(rollback_errors)
            )
        raise
    finally:
        for temporary in prepared.values():
            temporary.unlink(missing_ok=True)


async def _generate(run_root: Path) -> None:
    repository = JsonFileJobRepository(run_root / "jobs.json")
    artifacts = LocalArtifactStore(run_root / "artifacts")

    curated = CuratedChemistryPlanner()
    planner = ResilientPlannerOrchestrator(
        primary=curated,
        fallback=curated,
        max_attempts=1,
    )
    # Deliberately exclude the portable synthetic fallback: committed showcase
    # samples must contain real Edge speech or Windows' intelligible SAPI voice.
    speech = ResilientSpeechSynthesizer(
        primary=EdgeTTSSynthesizer(),
        offline=WindowsSAPISynthesizer(),
    )
    pipeline = VideoGenerationPipeline(
        repository=repository,
        artifact_store=artifacts,
        planner=planner,
        speech_synthesizer=speech,
        renderer=Renderer(width=1280, height=720),
        max_concurrent_jobs=1,
        media_verifier=verify_rendered_video,
    )

    generated: list[tuple[SampleSpec, Path, float]] = []
    records: list[dict[str, Any]] = []
    total = len(SAMPLE_SPECS)

    for position, spec in enumerate(SAMPLE_SPECS, start=1):
        print(f"[{position}/{total}] Generating {spec.output_file}", flush=True)
        print(f"          Query: {spec.learner_query}", flush=True)
        await repository.create(
            VideoJob(
                job_id=spec.job_id,
                query=spec.learner_query,
                subject="chemistry",
            )
        )
        completed = await pipeline.execute(spec.job_id)
        if completed.status is not JobStatus.COMPLETED:
            code = completed.error_code or "PIPELINE_FAILED"
            message = completed.error_message or "No failure detail was recorded."
            raise SampleGenerationError(
                f"{spec.output_file} failed ({code}): {message}"
            )
        if completed.duration_seconds is None:
            raise SampleGenerationError(
                f"{spec.output_file} completed without duration metadata"
            )

        source = await artifacts.get_video_path(spec.job_id)
        if source is None:
            raise SampleGenerationError(
                f"{spec.output_file} completed without a final video artifact"
            )
        verification = await asyncio.to_thread(
            verify_rendered_video,
            source,
            completed.duration_seconds,
        )
        if not verification.passed:
            detail = "; ".join(verification.issues)
            raise SampleGenerationError(
                f"Final verification failed for {spec.output_file}: {detail}"
            )
        if (
            verification.duration_seconds is None
            or verification.file_size_bytes is None
        ):
            raise SampleGenerationError(
                f"Verification returned incomplete metadata for {spec.output_file}"
            )

        duration = float(verification.duration_seconds)
        generated.append((spec, source, duration))
        records.append(
            {
                "learner_query": spec.learner_query,
                "output_file": spec.output_file,
                "duration_seconds": duration,
                "file_size_bytes": verification.file_size_bytes,
                "validation_passed": True,
            }
        )
        print(
            f"[{position}/{total}] Verified {duration:.3f}s, "
            f"{verification.file_size_bytes:,} bytes",
            flush=True,
        )

    manifest_source = run_root / "publish" / "manifest.json"
    _write_manifest_source(manifest_source, records)
    print("[publish] All three videos passed; publishing sample set...", flush=True)
    await asyncio.to_thread(
        _publish_batch,
        generated,
        manifest_source,
        run_root / "publish" / "backups",
    )
    print(f"[done] Wrote {MANIFEST_PATH}", flush=True)


def main() -> int:
    run_root: Path | None = None
    try:
        _require_media_tools()
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        run_root = Path(tempfile.mkdtemp(prefix="sample-generation-", dir=RUNTIME_ROOT))
        print(f"Runtime workspace: {run_root}", flush=True)
        asyncio.run(_generate(run_root))
        print("Sample generation completed successfully.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("ERROR: Sample generation was interrupted.", file=sys.stderr, flush=True)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if run_root is not None:
            try:
                shutil.rmtree(run_root)
            except OSError as cleanup_error:
                print(
                    f"WARNING: Could not remove runtime workspace: {cleanup_error}",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    raise SystemExit(main())
