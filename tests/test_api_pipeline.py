"""Deterministic orchestration and HTTP contract tests.

These tests deliberately replace speech, rendering, media inspection, and the
background worker with local fakes.  They exercise lifecycle and API behavior
without relying on the network or installed FFmpeg binaries.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
from typing import Any

import httpx
import pytest

import app.main as main_module
from app.artifacts import FINAL_VIDEO_FILENAME, LocalArtifactStore
from app.main import create_app
from app.models import (
    AttemptRecord,
    JobStatus,
    SceneSpec,
    Storyboard,
    VideoJob,
    VisualTemplate,
)
from app.pipeline import VideoGenerationPipeline
from app.planner import PlanningResult
from app.renderer import RenderResult
from app.store import InMemoryJobRepository
from app.validator import MediaVerificationResult


SUPPORTED_QUERY = "How does the pH scale work?"


@pytest.fixture
def test_tmp_path(request: pytest.FixtureRequest) -> Path:
    """Use a repository-local temp root when the OS temp ACL is unavailable."""

    root = Path(".pytest-api-pipeline") / request.node.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _storyboard(query: str = SUPPORTED_QUERY) -> Storyboard:
    return Storyboard(
        topic_id="ph_scale",
        title="How the pH scale works",
        learner_query=query,
        scenes=[
            SceneSpec(
                scene_index=index,
                heading=f"pH scene {index}",
                narration=f"Deterministic narration for pH scene {index}.",
                caption=f"pH caption {index}",
                visual_template=VisualTemplate.PH_SCALE_SPECTRUM,
                visual_params={"target_ph": index + 1},
            )
            for index in range(1, 4)
        ],
    )


class RecordingRepository(InMemoryJobRepository):
    """In-memory repository that retains successful transition snapshots."""

    def __init__(self, jobs: list[VideoJob] | None = None) -> None:
        super().__init__(jobs)
        self.transitions: list[tuple[JobStatus, int]] = []

    async def transition(
        self,
        job_id: str,
        expected_status: Any,
        new_status: Any,
        **changes: Any,
    ) -> VideoJob:
        updated = await super().transition(
            job_id, expected_status, new_status, **changes
        )
        self.transitions.append((updated.status, updated.progress_pct))
        return updated


class FakePlanner:
    async def plan(self, query: str, topic: Any) -> PlanningResult:
        assert topic.topic_id == "ph_scale"
        return PlanningResult(
            storyboard=_storyboard(query),
            attempts=(
                AttemptRecord(
                    attempt_number=1,
                    provider="fake_planner",
                    passed_validation=True,
                    duration_ms=1,
                ),
            ),
            used_fallback=False,
        )


class ConcurrencyProbePlanner(FakePlanner):
    def __init__(self) -> None:
        self.active = 0
        self.peak_active = 0
        self.calls = 0
        self.two_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def plan(self, query: str, topic: Any) -> PlanningResult:
        self.calls += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        if self.active == 2:
            self.two_entered.set()
        try:
            await self.release.wait()
            return await super().plan(query, topic)
        finally:
            self.active -= 1


class FakeSynthesizer:
    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        assert text
        path = Path(output_path).with_suffix(".wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-wave-data")
        return path


class FakeRenderer:
    def render(
        self,
        storyboard: Storyboard,
        audio_paths: list[Path],
        output_path: str | Path,
    ) -> RenderResult:
        assert len(audio_paths) == len(storyboard.scenes)
        assert all(path.is_file() for path in audio_paths)
        destination = Path(output_path)
        destination.write_bytes(b"deterministic-fake-mp4")
        scene_durations = tuple(4.0 for _ in storyboard.scenes)
        return RenderResult(
            output_path=destination,
            duration_seconds=sum(scene_durations),
            scene_durations=scene_durations,
            scene_count=len(scene_durations),
            frame_count=240,
            size_bytes=destination.stat().st_size,
        )


def _passing_verifier(
    path: str | Path, expected_duration: float
) -> MediaVerificationResult:
    assert Path(path).is_file()
    return MediaVerificationResult(
        passed=True,
        issues=(),
        file_size_bytes=Path(path).stat().st_size,
        duration_seconds=expected_duration,
        video_codec="h264",
        width=1280,
        height=720,
        audio_codec="aac",
    )


def _failing_verifier(
    path: str | Path, expected_duration: float
) -> MediaVerificationResult:
    assert Path(path).is_file()
    assert expected_duration > 0
    return MediaVerificationResult(
        passed=False,
        issues=("audio_stream: ffprobe did not report an audio stream.",),
    )


def _pipeline(
    repository: InMemoryJobRepository,
    artifact_store: LocalArtifactStore,
    *,
    verifier: Any = _passing_verifier,
) -> VideoGenerationPipeline:
    return VideoGenerationPipeline(
        repository=repository,
        artifact_store=artifact_store,
        planner=FakePlanner(),
        speech_synthesizer=FakeSynthesizer(),
        renderer=FakeRenderer(),
        media_verifier=verifier,
    )


@pytest.mark.asyncio
async def test_pipeline_persists_lifecycle_progress_and_completed_artifact(
    test_tmp_path: Path,
) -> None:
    repository = RecordingRepository()
    artifacts = LocalArtifactStore(test_tmp_path / "artifacts")
    queued = await repository.create(VideoJob(query=SUPPORTED_QUERY))

    completed = await _pipeline(repository, artifacts).execute(queued.job_id)

    assert repository.transitions == [
        (JobStatus.PLANNING, 15),
        (JobStatus.VALIDATING_PLAN, 30),
        (JobStatus.SYNTHESIZING_AUDIO, 50),
        (JobStatus.RENDERING_VISUALS, 75),
        (JobStatus.VERIFYING_ARTIFACT, 92),
        (JobStatus.COMPLETED, 100),
    ]
    assert completed.status == JobStatus.COMPLETED
    assert completed.topic_id == "ph_scale"
    assert completed.started_at is not None
    assert completed.completed_at is not None
    assert completed.artifact_url == f"/api/v1/videos/{queued.job_id}/video"
    assert completed.duration_seconds == 12.0
    assert completed.file_size_bytes == len(b"deterministic-fake-mp4")
    assert await artifacts.get_video_path(queued.job_id) is not None


@pytest.mark.asyncio
async def test_pipeline_limits_concurrent_jobs_to_two(
    test_tmp_path: Path,
) -> None:
    repository = InMemoryJobRepository()
    artifacts = LocalArtifactStore(test_tmp_path / "artifacts")
    planner = ConcurrencyProbePlanner()
    pipeline = VideoGenerationPipeline(
        repository=repository,
        artifact_store=artifacts,
        planner=planner,
        speech_synthesizer=FakeSynthesizer(),
        renderer=FakeRenderer(),
        media_verifier=_passing_verifier,
    )
    jobs = [
        await repository.create(
            VideoJob(job_id=f"concurrent-{index}", query=SUPPORTED_QUERY)
        )
        for index in range(3)
    ]
    tasks = [asyncio.create_task(pipeline.execute(job.job_id)) for job in jobs]

    await asyncio.wait_for(planner.two_entered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert planner.calls == 2
    assert planner.active == 2
    assert planner.peak_active == 2

    planner.release.set()
    completed = await asyncio.gather(*tasks)

    assert planner.calls == 3
    assert planner.peak_active == 2
    assert all(job.status == JobStatus.COMPLETED for job in completed)


@pytest.mark.asyncio
async def test_unsupported_topic_ends_in_stable_terminal_failure(
    test_tmp_path: Path,
) -> None:
    repository = RecordingRepository()
    queued = await repository.create(
        VideoJob(query="Explain orbital hybridization in xenon compounds.")
    )

    failed = await _pipeline(
        repository, LocalArtifactStore(test_tmp_path / "artifacts")
    ).execute(queued.job_id)

    assert repository.transitions == [
        (JobStatus.PLANNING, 15),
        (JobStatus.FAILED, 15),
    ]
    assert failed.status == JobStatus.FAILED
    assert failed.error_code == "UNSUPPORTED_TOPIC"
    assert failed.error_message
    assert failed.completed_at is not None
    assert failed.artifact_url is None


@pytest.mark.asyncio
async def test_failed_media_verification_never_publishes_video(
    test_tmp_path: Path,
) -> None:
    repository = RecordingRepository()
    artifacts = LocalArtifactStore(test_tmp_path / "artifacts")
    queued = await repository.create(VideoJob(query=SUPPORTED_QUERY))

    failed = await _pipeline(repository, artifacts, verifier=_failing_verifier).execute(
        queued.job_id
    )

    assert failed.status == JobStatus.FAILED
    assert failed.error_code == "MEDIA_VERIFICATION_FAILED"
    assert failed.artifact_url is None
    assert failed.duration_seconds is None
    assert failed.file_size_bytes is None
    assert await artifacts.get_video_path(queued.job_id) is None
    assert repository.transitions[-2:] == [
        (JobStatus.VERIFYING_ARTIFACT, 92),
        (JobStatus.FAILED, 92),
    ]


class NoOpPipeline:
    def __init__(self) -> None:
        self.executed_job_ids: list[str] = []

    async def execute(self, job_id: str) -> None:
        self.executed_job_ids.append(job_id)


@pytest.mark.asyncio
async def test_post_returns_202_with_polling_url_and_queued_record(
    test_tmp_path: Path,
) -> None:
    repository = InMemoryJobRepository()
    fake_pipeline = NoOpPipeline()
    application = create_app(
        repo=repository,
        artifacts=LocalArtifactStore(test_tmp_path / "artifacts"),
        pipeline=fake_pipeline,  # type: ignore[arg-type]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/videos", json={"question": SUPPORTED_QUERY}
        )
        await asyncio.sleep(0)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["progress_pct"] == 0
    assert body["status_url"] == (f"http://testserver/api/v1/videos/{body['job_id']}")
    assert fake_pipeline.executed_job_ids == [body["job_id"]]
    persisted = await repository.get(body["job_id"])
    assert persisted is not None
    assert persisted.query == SUPPORTED_QUERY


@pytest.mark.asyncio
async def test_list_is_newest_first_filters_status_and_detail_is_complete(
    test_tmp_path: Path,
) -> None:
    oldest_time = datetime(2026, 1, 1, tzinfo=UTC)
    oldest = VideoJob(
        job_id="oldest",
        query=SUPPORTED_QUERY,
        created_at=oldest_time,
        updated_at=oldest_time,
    )
    failed_time = oldest_time + timedelta(minutes=1)
    failed = VideoJob(
        job_id="failed",
        query="Unsupported chemistry question",
        status=JobStatus.FAILED,
        progress_pct=15,
        error_code="UNSUPPORTED_TOPIC",
        error_message="Unsupported topic.",
        created_at=failed_time,
        updated_at=failed_time,
        completed_at=failed_time,
    )
    newest_time = oldest_time + timedelta(minutes=2)
    newest = VideoJob(
        job_id="newest",
        query=SUPPORTED_QUERY,
        topic_id="ph_scale",
        status=JobStatus.COMPLETED,
        progress_pct=100,
        attempts=[
            AttemptRecord(
                attempt_number=1,
                provider="fake_planner",
                passed_validation=True,
                duration_ms=1,
            )
        ],
        storyboard=_storyboard(),
        artifact_url="/api/v1/videos/newest/video",
        duration_seconds=12.0,
        file_size_bytes=123,
        created_at=newest_time,
        updated_at=newest_time,
        started_at=newest_time,
        completed_at=newest_time,
    )
    repository = InMemoryJobRepository([oldest, newest, failed])
    application = create_app(
        repo=repository,
        artifacts=LocalArtifactStore(test_tmp_path / "artifacts"),
        pipeline=NoOpPipeline(),  # type: ignore[arg-type]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        listed = await client.get("/api/v1/videos")
        filtered = await client.get("/api/v1/videos", params={"status": "failed"})
        detail = await client.get("/api/v1/videos/newest")
        missing = await client.get("/api/v1/videos/missing")

    assert listed.status_code == 200
    assert [job["job_id"] for job in listed.json()] == [
        "newest",
        "failed",
        "oldest",
    ]
    assert filtered.status_code == 200
    assert [job["job_id"] for job in filtered.json()] == ["failed"]
    assert detail.status_code == 200
    assert detail.json()["storyboard"]["topic_id"] == "ph_scale"
    assert detail.json()["attempts"][0]["provider"] == "fake_planner"
    assert detail.json()["artifact_url"] == "/api/v1/videos/newest/video"
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_video_download_returns_404_409_and_completed_file(
    test_tmp_path: Path,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    queued = VideoJob(job_id="queued", query=SUPPORTED_QUERY)
    completed = VideoJob(
        job_id="completed",
        query=SUPPORTED_QUERY,
        status=JobStatus.COMPLETED,
        progress_pct=100,
        artifact_url="/api/v1/videos/completed/video",
        duration_seconds=12.0,
        file_size_bytes=14,
        created_at=now,
        updated_at=now,
        completed_at=now,
    )
    missing_artifact = VideoJob(
        job_id="missing-artifact",
        query=SUPPORTED_QUERY,
        status=JobStatus.COMPLETED,
        progress_pct=100,
        artifact_url="/api/v1/videos/missing-artifact/video",
        duration_seconds=12.0,
        file_size_bytes=14,
        created_at=now,
        updated_at=now,
        completed_at=now,
    )
    repository = InMemoryJobRepository([queued, completed, missing_artifact])
    artifacts = LocalArtifactStore(test_tmp_path / "artifacts")
    video_bytes = b"fake-mp4-bytes"
    await artifacts.write_bytes("completed", FINAL_VIDEO_FILENAME, video_bytes)
    application = create_app(
        repo=repository,
        artifacts=artifacts,
        pipeline=NoOpPipeline(),  # type: ignore[arg-type]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        unknown = await client.get("/api/v1/videos/unknown/video")
        not_ready = await client.get("/api/v1/videos/queued/video")
        absent = await client.get("/api/v1/videos/missing-artifact/video")
        downloaded = await client.get("/api/v1/videos/completed/video")

    assert unknown.status_code == 404
    assert not_ready.status_code == 409
    assert not_ready.json()["detail"]["status"] == "queued"
    assert absent.status_code == 404
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "video/mp4"
    assert 'filename="completed.mp4"' in downloaded.headers["content-disposition"]
    assert downloaded.content == video_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("available_tools", "expected_status", "expected_code"),
    [
        ({"ffmpeg", "ffprobe"}, "ok", 200),
        ({"ffmpeg"}, "degraded", 503),
    ],
)
async def test_health_reports_tool_availability(
    monkeypatch: pytest.MonkeyPatch,
    available_tools: set[str],
    expected_status: str,
    expected_code: int,
) -> None:
    monkeypatch.setattr(
        main_module.shutil,
        "which",
        lambda executable: (
            f"/fake/{executable}" if executable in available_tools else None
        ),
    )
    application = create_app(
        repo=InMemoryJobRepository(),
        artifacts=LocalArtifactStore(Path(".pytest-api-health-artifacts")),
        pipeline=NoOpPipeline(),  # type: ignore[arg-type]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == expected_code
    assert response.json() == {
        "status": expected_status,
        "ffmpeg_available": "ffmpeg" in available_tools,
        "ffprobe_available": "ffprobe" in available_tools,
    }
