"""Black-box HTTP coverage for the real video-generation pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import httpx
import pytest

# Keep direct invocations such as ``pytest tests/test_api.py`` working on Windows.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artifacts import LocalArtifactStore
from app.main import create_app
from app.models import Storyboard
from app.pipeline import VideoGenerationPipeline
from app.planner import CuratedChemistryPlanner, ResilientPlannerOrchestrator
from app.renderer import RenderResult
from app.store import InMemoryJobRepository


SUPPORTED_QUESTION = "How does the pH scale work?"
UNSUPPORTED_QUESTION = "Explain the causes of the French Revolution."
TERMINAL_STATUSES = {"completed", "failed"}


@pytest.fixture
def api_tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Avoid the locked system temp directory used by this Windows checkout."""

    root = Path(".pytest-api") / request.node.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class GateableAudioStub:
    """Offline audio double that holds the supported job in a known state."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        assert text
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            await self.release.wait()

        path = Path(output_path).with_suffix(".offline-audio")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"deterministic offline audio stub")
        return path


class FastMp4Renderer:
    """Create a short real H.264/AAC artifact without rendering scene frames."""

    duration_seconds = 12.0
    frame_rate = 2

    def __init__(self) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required for the black-box API test")
        self.ffmpeg = executable

    def render(
        self,
        storyboard: Storyboard,
        audio_paths: list[Path],
        output_path: str | Path,
    ) -> RenderResult:
        assert len(audio_paths) == len(storyboard.scenes)
        assert all(path.is_file() for path in audio_paths)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            (
                "testsrc2=size=1280x720:rate="
                f"{self.frame_rate}:duration={self.duration_seconds:g}"
            ),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{self.duration_seconds:g}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "38",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "32k",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            shell=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert destination.is_file()

        scene_duration = self.duration_seconds / len(storyboard.scenes)
        return RenderResult(
            output_path=destination,
            duration_seconds=self.duration_seconds,
            scene_durations=tuple(scene_duration for _ in storyboard.scenes),
            scene_count=len(storyboard.scenes),
            frame_count=int(self.duration_seconds * self.frame_rate),
            size_bytes=destination.stat().st_size,
        )


async def _poll_terminal_job(
    client: httpx.AsyncClient, job_id: str, *, timeout: float = 10.0
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    last_payload: dict[str, Any] | None = None
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/v1/videos/{job_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] in TERMINAL_STATUSES:
            return last_payload
        await asyncio.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish before the deadline: {last_payload}")


def _build_test_application(
    root: Path,
) -> tuple[Any, GateableAudioStub]:
    repository = InMemoryJobRepository()
    artifacts = LocalArtifactStore(root / "artifacts")
    audio = GateableAudioStub()
    planner = ResilientPlannerOrchestrator(
        primary=CuratedChemistryPlanner(),
        max_attempts=1,
        attempt_timeout_seconds=None,
    )
    pipeline = VideoGenerationPipeline(
        repository=repository,
        artifact_store=artifacts,
        planner=planner,
        speech_synthesizer=audio,
        renderer=FastMp4Renderer(),
        # Deliberately use the production ffprobe-based media verifier.
    )
    return (
        create_app(repo=repository, artifacts=artifacts, pipeline=pipeline),
        audio,
    )


@pytest.mark.asyncio
async def test_health_returns_200(api_tmp_path: Path) -> None:
    application, _ = _build_test_application(api_tmp_path)

    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ffmpeg_available": True,
        "ffprobe_available": True,
    }


@pytest.mark.asyncio
async def test_video_job_lifecycle_and_download(api_tmp_path: Path) -> None:
    application, audio = _build_test_application(api_tmp_path)

    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                "/api/v1/videos",
                json={"question": SUPPORTED_QUESTION},
            )
            assert created.status_code == 202
            created_payload = created.json()
            assert created_payload["status"] == "queued"
            job_id = created_payload["job_id"]
            assert created_payload["status_url"].endswith(f"/api/v1/videos/{job_id}")

            await asyncio.wait_for(audio.entered.wait(), timeout=2)
            in_progress = await client.get(f"/api/v1/videos/{job_id}")
            assert in_progress.status_code == 200
            assert in_progress.json()["status"] == "synthesizing_audio"

            collection = await client.get("/api/v1/videos")
            assert collection.status_code == 200
            assert job_id in {job["job_id"] for job in collection.json()}

            unavailable = await client.get(f"/api/v1/videos/{job_id}/video")
            assert unavailable.status_code == 409
            assert unavailable.json()["detail"]["status"] == "synthesizing_audio"

            audio.release.set()
            completed = await _poll_terminal_job(client, job_id)
            assert completed["status"] == "completed", completed
            assert completed["progress_pct"] == 100
            assert completed["artifact_url"] == f"/api/v1/videos/{job_id}/video"
            assert completed["duration_seconds"] == pytest.approx(12.0, abs=0.05)
            assert completed["file_size_bytes"] >= 30 * 1024

            video = await client.get(f"/api/v1/videos/{job_id}/video")
            assert video.status_code == 200
            assert video.headers["content-type"] == "video/mp4"
            assert video.content[4:8] == b"ftyp"
            assert len(video.content) == completed["file_size_bytes"]


@pytest.mark.asyncio
async def test_unsupported_query_transitions_to_failed(
    api_tmp_path: Path,
) -> None:
    application, _ = _build_test_application(api_tmp_path)

    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://testserver",
        ) as client:
            rejected = await client.post(
                "/api/v1/videos",
                json={"question": UNSUPPORTED_QUESTION},
            )
            assert rejected.status_code == 202
            rejected_id = rejected.json()["job_id"]
            failed = await _poll_terminal_job(client, rejected_id, timeout=2)
            assert failed["status"] == "failed"
            assert failed["error_code"] == "UNSUPPORTED_TOPIC"
            assert failed["artifact_url"] is None
