"""End-to-end orchestration for persisted chemistry video jobs.

The pipeline owns lifecycle transitions, but delegates storage, planning,
speech, rendering, and topic selection to injected collaborators.  A job is
published only after its staged video passes the media quality gate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
import math
from pathlib import Path
from typing import Any

from .artifacts import (
    FINAL_VIDEO_FILENAME,
    ArtifactStore,
    LocalArtifactStore,
)
from .models import JobStatus, VideoJob, utc_now
from .planner import (
    ResilientPlannerOrchestrator,
    StoryboardPlanningError,
)
from .renderer import Renderer
from .store import (
    JobNotFoundError,
    JobRepository,
    JobStateConflictError,
)
from .topics import TopicDefinition, TopicRegistry, match_topic
from .tts import ResilientSpeechSynthesizer, SpeechSynthesizer
from .validator import MediaVerificationResult, verify_rendered_video


ArtifactUrlBuilder = Callable[[str], str]
MediaVerifier = Callable[[str | Path, float], MediaVerificationResult]


class VideoGenerationPipeline:
    """Run one persisted job through the validated local media pipeline.

    ``repository`` is deliberately required so callers cannot accidentally run
    work without a durable source of truth.  Other defaults are production
    local adapters; every collaborator remains injectable for tests.
    """

    def __init__(
        self,
        repository: JobRepository,
        artifact_store: ArtifactStore | None = None,
        planner: Any | None = None,
        speech_synthesizer: SpeechSynthesizer | None = None,
        renderer: Renderer | None = None,
        topic_matcher: TopicRegistry
        | Callable[..., TopicDefinition | None]
        | None = None,
        *,
        max_concurrent_jobs: int = 2,
        artifact_url_builder: ArtifactUrlBuilder | None = None,
        media_verifier: MediaVerifier | None = None,
    ) -> None:
        if (
            not isinstance(max_concurrent_jobs, int)
            or isinstance(max_concurrent_jobs, bool)
            or max_concurrent_jobs < 1
        ):
            raise ValueError("max_concurrent_jobs must be a positive integer")
        if media_verifier is not None and not callable(media_verifier):
            raise TypeError("media_verifier must be callable")

        self.repository = repository
        self.artifact_store = (
            artifact_store if artifact_store is not None else LocalArtifactStore()
        )
        self.planner = (
            planner if planner is not None else ResilientPlannerOrchestrator()
        )
        self.speech_synthesizer = (
            speech_synthesizer
            if speech_synthesizer is not None
            else ResilientSpeechSynthesizer()
        )
        self.renderer = renderer if renderer is not None else Renderer()
        self.topic_matcher = topic_matcher if topic_matcher is not None else match_topic
        self.artifact_url_builder = (
            artifact_url_builder
            if artifact_url_builder is not None
            else lambda job_id: f"/api/v1/videos/{job_id}/video"
        )
        self.media_verifier = (
            media_verifier if media_verifier is not None else verify_rendered_video
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)

    async def execute(self, job_id: str) -> VideoJob:
        """Execute a queued job and return its final persisted record.

        Terminal jobs are returned unchanged, making duplicate queue wake-ups
        harmless.  A non-terminal job that is no longer queued is rejected so
        a second worker cannot repeat TTS or rendering side effects.
        """

        async with self._semaphore:
            job = await self.repository.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return job
            if job.status != JobStatus.QUEUED:
                raise JobStateConflictError(
                    f"Job {job_id!r} is already in state {job.status.value!r}"
                )

            current_status = JobStatus.QUEUED
            try:
                job = await self.repository.transition(
                    job_id,
                    JobStatus.QUEUED,
                    JobStatus.PLANNING,
                    progress_pct=15,
                    started_at=utc_now(),
                    error_code=None,
                    error_message=None,
                    artifact_url=None,
                    duration_seconds=None,
                    file_size_bytes=None,
                )
                current_status = JobStatus.PLANNING

                topic = self._match_topic(job.query, job.subject)
                if topic is None:
                    return await self._mark_failed(
                        job_id,
                        current_status,
                        "UNSUPPORTED_TOPIC",
                        "This service currently supports registered chemistry/STEM "
                        "topics such as pH, covalent bonds, and ionic versus "
                        "covalent bonding.",
                    )

                job = await self.repository.update(
                    job_id,
                    {"topic_id": topic.topic_id},
                    expected_status=JobStatus.PLANNING,
                )

                try:
                    planning_result = await self.planner.plan(job.query, topic)
                except StoryboardPlanningError as exc:
                    return await self._mark_failed(
                        job_id,
                        current_status,
                        "PLANNING_FAILED",
                        "A safe, validated storyboard could not be produced.",
                        attempts=list(exc.attempts),
                    )

                storyboard = planning_result.storyboard
                job = await self.repository.transition(
                    job_id,
                    JobStatus.PLANNING,
                    JobStatus.VALIDATING_PLAN,
                    progress_pct=30,
                    attempts=list(planning_result.attempts),
                    used_fallback=planning_result.used_fallback,
                    storyboard=storyboard,
                )
                current_status = JobStatus.VALIDATING_PLAN

                await self.artifact_store.save_storyboard(job_id, storyboard)

                job = await self.repository.transition(
                    job_id,
                    JobStatus.VALIDATING_PLAN,
                    JobStatus.SYNTHESIZING_AUDIO,
                    progress_pct=50,
                )
                current_status = JobStatus.SYNTHESIZING_AUDIO

                workspace = await self.artifact_store.workspace(job_id)
                audio_directory = workspace / "audio"
                await asyncio.to_thread(
                    audio_directory.mkdir, parents=True, exist_ok=True
                )
                audio_paths: list[Path] = []
                for position, scene in enumerate(storyboard.scenes, start=1):
                    requested_path = audio_directory / f"scene-{position:02d}.mp3"
                    produced_path = await self.speech_synthesizer.synthesize(
                        scene.narration, requested_path
                    )
                    audio_paths.append(Path(produced_path))

                job = await self.repository.transition(
                    job_id,
                    JobStatus.SYNTHESIZING_AUDIO,
                    JobStatus.RENDERING_VISUALS,
                    progress_pct=75,
                )
                current_status = JobStatus.RENDERING_VISUALS

                staged_path = await self.artifact_store.staging_path(job_id)
                render_result = await asyncio.to_thread(
                    self.renderer.render,
                    storyboard,
                    audio_paths,
                    staged_path,
                )
                rendered_path = Path(getattr(render_result, "output_path", staged_path))
                expected_duration = float(render_result.duration_seconds)
                if not math.isfinite(expected_duration) or expected_duration <= 0:
                    raise ValueError("renderer returned an invalid duration")

                job = await self.repository.transition(
                    job_id,
                    JobStatus.RENDERING_VISUALS,
                    JobStatus.VERIFYING_ARTIFACT,
                    progress_pct=92,
                )
                current_status = JobStatus.VERIFYING_ARTIFACT

                verification = await asyncio.to_thread(
                    self.media_verifier,
                    rendered_path,
                    expected_duration,
                )
                if not verification.passed:
                    return await self._mark_failed(
                        job_id,
                        current_status,
                        "MEDIA_VERIFICATION_FAILED",
                        "The rendered video did not pass the media quality checks.",
                    )

                final_path = await self.artifact_store.finalize_video(
                    job_id, rendered_path
                )
                metadata = await self.artifact_store.metadata(
                    job_id, FINAL_VIDEO_FILENAME
                )
                duration_seconds = (
                    verification.duration_seconds
                    if verification.duration_seconds is not None
                    else expected_duration
                )
                if duration_seconds <= 0 or metadata.size_bytes <= 0:
                    raise ValueError("verified artifact metadata is incomplete")

                artifact_url = self.artifact_url_builder(job_id)
                if not isinstance(artifact_url, str) or not artifact_url.strip():
                    raise ValueError("artifact_url_builder returned an invalid URL")
                if Path(final_path).name != FINAL_VIDEO_FILENAME:
                    raise ValueError("artifact store returned a non-canonical video")

                completed = await self.repository.transition(
                    job_id,
                    JobStatus.VERIFYING_ARTIFACT,
                    JobStatus.COMPLETED,
                    progress_pct=100,
                    artifact_url=artifact_url,
                    duration_seconds=float(duration_seconds),
                    file_size_bytes=metadata.size_bytes,
                    completed_at=utc_now(),
                    error_code=None,
                    error_message=None,
                )
                current_status = JobStatus.COMPLETED

                # Cleanup is best effort and cannot invalidate a committed,
                # verified artifact.
                try:
                    await self.artifact_store.remove_intermediates(job_id)
                except Exception:
                    pass
                return completed

            except asyncio.CancelledError:
                if current_status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                    try:
                        await asyncio.shield(
                            self._mark_failed(
                                job_id,
                                current_status,
                                "WORKER_INTERRUPTED",
                                "Video generation was interrupted before completion.",
                            )
                        )
                    except (JobNotFoundError, JobStateConflictError):
                        pass
                raise
            except (JobNotFoundError, JobStateConflictError):
                raise
            except Exception:
                code, message = self._safe_stage_error(current_status)
                return await self._mark_failed(job_id, current_status, code, message)

    # Compatibility-friendly verb for worker loops.
    run = execute

    def _match_topic(self, query: str, subject: str) -> TopicDefinition | None:
        matcher = self.topic_matcher
        match_method = getattr(matcher, "match", None)
        if callable(match_method):
            return match_method(query, subject=subject)
        if callable(matcher):
            try:
                signature = inspect.signature(matcher)
            except (TypeError, ValueError):
                signature = None
            accepts_subject = signature is not None and (
                "subject" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_subject:
                return matcher(query, subject=subject)
            return matcher(query)
        raise TypeError("topic_matcher must be callable or provide match()")

    async def _mark_failed(
        self,
        job_id: str,
        expected_status: JobStatus,
        error_code: str,
        error_message: str,
        **changes: Any,
    ) -> VideoJob:
        changes.update(
            error_code=error_code,
            error_message=error_message,
            artifact_url=None,
            duration_seconds=None,
            file_size_bytes=None,
            completed_at=utc_now(),
        )
        return await self.repository.transition(
            job_id,
            expected_status,
            JobStatus.FAILED,
            **changes,
        )

    @staticmethod
    def _safe_stage_error(status: JobStatus) -> tuple[str, str]:
        if status in (JobStatus.PLANNING, JobStatus.VALIDATING_PLAN):
            return (
                "PLANNING_FAILED",
                "A safe, validated storyboard could not be produced.",
            )
        if status == JobStatus.SYNTHESIZING_AUDIO:
            return "TTS_UNAVAILABLE", "Narration audio could not be generated."
        if status == JobStatus.RENDERING_VISUALS:
            return "RENDER_FAILED", "The video could not be rendered."
        if status == JobStatus.VERIFYING_ARTIFACT:
            return (
                "MEDIA_VERIFICATION_FAILED",
                "The rendered video could not be safely published.",
            )
        return "PIPELINE_FAILED", "Video generation could not be completed."


__all__ = ["VideoGenerationPipeline"]
