"""Job-scoped local artifact storage.

Only service-generated job IDs and relative keys are accepted.  All resolved
paths are checked against the artifact root to prevent accidental access to
another job (or to files outside runtime storage).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import Storyboard


DEFAULT_ARTIFACT_ROOT = Path("data/runtime_jobs/artifacts")
STORYBOARD_FILENAME = "storyboard.json"
FINAL_VIDEO_FILENAME = "explanation.mp4"
DEFAULT_STAGED_VIDEO_FILENAME = "explanation.staged.mp4"
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    key: str
    size_bytes: int
    media_type: str
    sha256: str
    path: Path


@runtime_checkable
class ArtifactStore(Protocol):
    async def workspace(self, job_id: str) -> Path: ...

    async def staging_path(
        self, job_id: str, name: str = DEFAULT_STAGED_VIDEO_FILENAME
    ) -> Path: ...

    async def write_bytes(self, job_id: str, key: str, data: bytes) -> Path: ...

    async def read_bytes(self, job_id: str, key: str) -> bytes: ...

    async def exists(self, job_id: str, key: str) -> bool: ...

    async def metadata(self, job_id: str, key: str) -> ArtifactMetadata: ...

    async def save_storyboard(self, job_id: str, storyboard: Storyboard) -> Path: ...

    async def load_storyboard(self, job_id: str) -> Storyboard | None: ...

    async def finalize_video(
        self, job_id: str, staged_path: str | Path | None = None
    ) -> Path: ...

    async def get_video_path(self, job_id: str) -> Path | None: ...

    async def remove_intermediates(self, job_id: str) -> None: ...


def _jsonable(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "json"):
        return json.loads(model.json())
    if isinstance(model, dict):
        return dict(model)
    raise TypeError(f"Cannot serialize {type(model).__name__}")


def _parse_storyboard(payload: dict[str, Any]) -> Storyboard:
    if hasattr(Storyboard, "model_validate"):
        return Storyboard.model_validate(payload)
    if hasattr(Storyboard, "parse_obj"):
        return Storyboard.parse_obj(payload)
    return Storyboard(**payload)


class LocalArtifactStore:
    def __init__(self, root: str | Path = DEFAULT_ARTIFACT_ROOT) -> None:
        self.root = Path(root)

    async def workspace(self, job_id: str) -> Path:
        workspace = self._job_path(job_id)
        await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
        # Re-resolve after creation to catch a pre-existing escaping symlink.
        self._ensure_within(workspace.resolve(), self.root.resolve())
        return workspace

    async def staging_path(
        self, job_id: str, name: str = DEFAULT_STAGED_VIDEO_FILENAME
    ) -> Path:
        await self.workspace(job_id)
        path = self._artifact_path(job_id, name)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        return path

    async def write_bytes(self, job_id: str, key: str, data: bytes) -> Path:
        if not isinstance(data, bytes):
            raise TypeError("Artifact data must be bytes")
        await self.workspace(job_id)
        path = self._artifact_path(job_id, key)
        await asyncio.to_thread(self._atomic_write_bytes, path, data)
        return path

    async def read_bytes(self, job_id: str, key: str) -> bytes:
        path = self._artifact_path(job_id, key)
        return await asyncio.to_thread(path.read_bytes)

    async def exists(self, job_id: str, key: str) -> bool:
        path = self._artifact_path(job_id, key)
        return await asyncio.to_thread(path.is_file)

    async def metadata(self, job_id: str, key: str) -> ArtifactMetadata:
        path = self._artifact_path(job_id, key)
        size, digest = await asyncio.to_thread(self._size_and_checksum, path)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return ArtifactMetadata(
            key=key,
            size_bytes=size,
            media_type=media_type,
            sha256=digest,
            path=path,
        )

    async def save_storyboard(self, job_id: str, storyboard: Storyboard) -> Path:
        payload = (
            json.dumps(
                _jsonable(storyboard),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        return await self.write_bytes(job_id, STORYBOARD_FILENAME, payload)

    async def load_storyboard(self, job_id: str) -> Storyboard | None:
        path = self._artifact_path(job_id, STORYBOARD_FILENAME)
        try:
            data = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return None
        return _parse_storyboard(json.loads(data))

    async def finalize_video(
        self, job_id: str, staged_path: str | Path | None = None
    ) -> Path:
        """Atomically promote a staged job-local MP4 to ``explanation.mp4``."""

        workspace = await self.workspace(job_id)
        source = (
            self._artifact_path(job_id, DEFAULT_STAGED_VIDEO_FILENAME)
            if staged_path is None
            else Path(staged_path)
        )
        if not source.is_absolute():
            # A bare key is interpreted within this job's namespace.  A path
            # already rooted at the workspace is preserved as supplied.
            candidate = source.resolve()
            if not self._is_within(candidate, workspace.resolve()):
                source = workspace / source
        source = source.resolve()
        self._ensure_within(source, workspace.resolve())
        destination = self._artifact_path(job_id, FINAL_VIDEO_FILENAME)

        def promote() -> None:
            if not source.is_file():
                raise FileNotFoundError(source)
            if source == destination.resolve():
                return
            os.replace(source, destination)

        await asyncio.to_thread(promote)
        return destination

    async def get_video_path(self, job_id: str) -> Path | None:
        path = self._artifact_path(job_id, FINAL_VIDEO_FILENAME)
        return path if await asyncio.to_thread(path.is_file) else None

    async def remove_intermediates(self, job_id: str) -> None:
        """Remove job-local files except the canonical storyboard and video."""

        workspace = self._job_path(job_id)
        if not await asyncio.to_thread(workspace.is_dir):
            return
        keep = {STORYBOARD_FILENAME, FINAL_VIDEO_FILENAME}

        def remove() -> None:
            for child in workspace.iterdir():
                if child.name in keep:
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)

        await asyncio.to_thread(remove)

    def _job_path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError("Invalid job_id")
        if job_id in {".", ".."}:
            raise ValueError("Invalid job_id")
        root = self.root.resolve()
        candidate = (root / job_id).resolve()
        self._ensure_within(candidate, root)
        return candidate

    def _artifact_path(self, job_id: str, key: str) -> Path:
        if not isinstance(key, str) or not key:
            raise ValueError("Artifact key must be a non-empty relative path")
        relative = Path(key)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("Artifact key must be a safe relative path")
        workspace = self._job_path(job_id)
        candidate = (workspace / relative).resolve()
        self._ensure_within(candidate, workspace.resolve())
        return candidate

    @staticmethod
    def _is_within(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    @classmethod
    def _ensure_within(cls, path: Path, parent: Path) -> None:
        if not cls._is_within(path, parent):
            raise ValueError("Artifact path escapes its job workspace")

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _size_and_checksum(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        return size, digest.hexdigest()
