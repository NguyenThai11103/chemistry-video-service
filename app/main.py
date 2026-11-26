"""FastAPI entry point for the chemistry video generation service.

The HTTP layer deliberately stays thin: job records remain the source of truth,
while :class:`VideoGenerationPipeline` owns all lifecycle transitions after a
queued record has been persisted.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from .artifacts import ArtifactStore, LocalArtifactStore
from .models import (
    CreateVideoRequest,
    JobStatus,
    VideoJob,
    VideoJobDetail,
    VideoJobSummary,
)
from .pipeline import VideoGenerationPipeline
from .store import JobRepository, JsonFileJobRepository


logger = logging.getLogger(__name__)


class CreateVideoResponse(VideoJobSummary):
    """Queued job metadata plus the canonical polling URL."""

    status_url: StrictStr


class HealthResponse(BaseModel):
    """Readiness of the process and its required local media tools."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    ffmpeg_available: StrictBool
    ffprobe_available: StrictBool


def _summary(job: VideoJob) -> VideoJobSummary:
    return VideoJobSummary.model_validate(job)


def _detail(job: VideoJob) -> VideoJobDetail:
    return VideoJobDetail.model_validate(job)


def create_app(
    repo: JobRepository | None = None,
    artifacts: ArtifactStore | None = None,
    pipeline: VideoGenerationPipeline | None = None,
) -> FastAPI:
    """Build an application with optional storage and pipeline test doubles."""

    repository = repo if repo is not None else JsonFileJobRepository()
    artifact_store = artifacts if artifacts is not None else LocalArtifactStore()
    video_pipeline = (
        pipeline
        if pipeline is not None
        else VideoGenerationPipeline(repository, artifact_store)
    )
    background_tasks: set[asyncio.Task[None]] = set()

    async def run_pipeline(job_id: str) -> None:
        try:
            await video_pipeline.execute(job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # The pipeline is responsible for persisting a safe terminal failure.
            # This guard prevents an unexpected implementation error from becoming
            # an unobserved task exception in the ASGI server.
            logger.exception("Unhandled video pipeline error for job %s", job_id)

    def schedule_pipeline(job_id: str) -> None:
        task = asyncio.create_task(
            run_pipeline(job_id), name=f"video-generation-{job_id}"
        )
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            remaining = tuple(background_tasks)
            for task in remaining:
                task.cancel()
            if remaining:
                await asyncio.gather(*remaining, return_exceptions=True)

    application = FastAPI(
        title="Chemistry Video Service",
        summary="Generate short, validated chemistry explainer videos.",
        description=(
            "Submit a supported chemistry question, poll the durable job record, "
            "and download the verified MP4 when generation completes."
        ),
        version="1.0.0",
        openapi_tags=[
            {
                "name": "health",
                "description": "Process readiness and required media tooling.",
            },
            {
                "name": "videos",
                "description": "Create, inspect, list, and download video jobs.",
            },
        ],
        lifespan=lifespan,
    )
    application.state.repository = repository
    application.state.artifacts = artifact_store
    application.state.pipeline = video_pipeline
    application.state.background_tasks = background_tasks

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
        summary="Check service readiness",
    )
    async def health(response: Response) -> HealthResponse:
        ffmpeg_available = shutil.which("ffmpeg") is not None
        ffprobe_available = shutil.which("ffprobe") is not None
        ready = ffmpeg_available and ffprobe_available
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if ready else "degraded",
            ffmpeg_available=ffmpeg_available,
            ffprobe_available=ffprobe_available,
        )

    @application.post(
        "/api/v1/videos",
        response_model=CreateVideoResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["videos"],
        summary="Queue a chemistry explainer video",
    )
    async def create_video(
        payload: CreateVideoRequest, request: Request
    ) -> CreateVideoResponse:
        job = await repository.create(
            VideoJob(query=payload.question, subject=payload.subject)
        )
        schedule_pipeline(job.job_id)
        summary = _summary(job)
        return CreateVideoResponse(
            **summary.model_dump(),
            status_url=str(request.url_for("get_video_job", job_id=job.job_id)),
        )

    @application.get(
        "/api/v1/videos",
        response_model=list[VideoJobSummary],
        tags=["videos"],
        summary="List video jobs",
    )
    async def list_video_jobs(
        job_status: JobStatus | None = Query(
            default=None,
            alias="status",
            description="Return only jobs in this lifecycle state.",
        ),
    ) -> list[VideoJobSummary]:
        jobs = await repository.list()
        if job_status is not None:
            jobs = [job for job in jobs if job.status == job_status]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return [_summary(job) for job in jobs]

    @application.get(
        "/api/v1/videos/{job_id}",
        response_model=VideoJobDetail,
        tags=["videos"],
        summary="Get a video job",
        name="get_video_job",
    )
    async def get_video_job(job_id: str) -> VideoJobDetail:
        job = await repository.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video job not found.",
            )
        return _detail(job)

    @application.get(
        "/api/v1/videos/{job_id}/video",
        response_class=FileResponse,
        responses={
            status.HTTP_200_OK: {"content": {"video/mp4": {}}},
            status.HTTP_404_NOT_FOUND: {"description": "Job or video not found."},
            status.HTTP_409_CONFLICT: {"description": "The job is not completed."},
        },
        tags=["videos"],
        summary="Download a completed video",
    )
    async def download_video(job_id: str) -> FileResponse:
        job = await repository.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video job not found.",
            )
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "Video is not available until the job is completed.",
                    "status": job.status.value,
                },
            )

        video_path = await artifact_store.get_video_path(job_id)
        if video_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Completed video artifact not found.",
            )
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    return application


app = create_app()
