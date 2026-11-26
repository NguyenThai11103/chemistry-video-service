"""Persistence abstractions for video jobs.

The JSON implementation is intentionally small and single-process.  Its lock
protects read/modify/write sequences within the process, while ``os.replace``
prevents readers from observing a partially written repository file.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import VideoJob


DEFAULT_JOBS_PATH = Path("data/runtime_jobs/jobs.json")
_TERMINAL_STATUSES = frozenset({"completed", "failed"})


class DuplicateJobError(ValueError):
    """Raised when a job with the same ID already exists."""


class JobNotFoundError(KeyError):
    """Raised when an update targets a job that does not exist."""


class JobStateConflictError(RuntimeError):
    """Raised when a compare-and-set update sees an unexpected status."""


@runtime_checkable
class JobRepository(Protocol):
    async def create(self, job: VideoJob) -> VideoJob: ...

    async def get(self, job_id: str) -> VideoJob | None: ...

    async def list(self) -> list[VideoJob]: ...

    async def list_non_terminal(self) -> list[VideoJob]: ...

    async def update(
        self,
        job_id: str,
        changes: Mapping[str, Any] | None = None,
        *,
        expected_status: Any | None = None,
        **fields: Any,
    ) -> VideoJob: ...

    async def save(
        self, job: VideoJob, *, expected_status: Any | None = None
    ) -> VideoJob: ...

    async def transition(
        self,
        job_id: str,
        expected_status: Any,
        new_status: Any,
        **changes: Any,
    ) -> VideoJob: ...


def _job_id(job: VideoJob) -> str:
    value = getattr(job, "job_id", None)
    if value is None:
        # This fallback keeps the storage boundary usable with older domain
        # records while the public service model uses ``job_id``.
        value = getattr(job, "id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("VideoJob must have a non-empty job_id")
    return value


def _jsonable(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "json"):
        return json.loads(model.json())
    if isinstance(model, Mapping):
        return dict(model)
    raise TypeError(f"Cannot serialize {type(model).__name__}")


def _parse_job(payload: Mapping[str, Any]) -> VideoJob:
    if hasattr(VideoJob, "model_validate"):
        return VideoJob.model_validate(payload)
    if hasattr(VideoJob, "parse_obj"):
        return VideoJob.parse_obj(payload)
    return VideoJob(**payload)


def _copy_job(job: VideoJob) -> VideoJob:
    if hasattr(job, "model_copy"):
        return job.model_copy(deep=True)
    if hasattr(job, "copy"):
        return job.copy(deep=True)
    return _parse_job(_jsonable(job))


def _status_value(job_or_status: Any) -> Any:
    value = getattr(job_or_status, "status", job_or_status)
    return getattr(value, "value", value)


def _apply_changes(job: VideoJob, changes: Mapping[str, Any]) -> VideoJob:
    payload = _jsonable(job)
    payload.update(changes)

    field_names = set(payload)
    model_fields = getattr(type(job), "model_fields", None)
    if model_fields is None:
        model_fields = getattr(type(job), "__fields__", None)
    if model_fields is not None:
        field_names.update(model_fields)
    if "updated_at" in field_names and "updated_at" not in changes:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    return _parse_job(payload)


class JsonFileJobRepository:
    """Store all runtime jobs in one atomically replaced JSON document."""

    def __init__(self, path: str | Path = DEFAULT_JOBS_PATH) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def create(self, job: VideoJob) -> VideoJob:
        job = _copy_job(job)
        identifier = _job_id(job)
        async with self._lock:
            jobs = await asyncio.to_thread(self._read_records)
            if identifier in jobs:
                raise DuplicateJobError(f"Job {identifier!r} already exists")
            jobs[identifier] = _jsonable(job)
            await asyncio.to_thread(self._write_records, jobs)
        return _copy_job(job)

    async def get(self, job_id: str) -> VideoJob | None:
        async with self._lock:
            jobs = await asyncio.to_thread(self._read_records)
            payload = jobs.get(job_id)
        return None if payload is None else _parse_job(payload)

    async def list(self) -> list[VideoJob]:
        async with self._lock:
            jobs = await asyncio.to_thread(self._read_records)
        return [_parse_job(payload) for payload in jobs.values()]

    async def list_non_terminal(self) -> list[VideoJob]:
        return [
            job
            for job in await self.list()
            if _status_value(job) not in _TERMINAL_STATUSES
        ]

    async def update(
        self,
        job_id: str,
        changes: Mapping[str, Any] | None = None,
        *,
        expected_status: Any | None = None,
        **fields: Any,
    ) -> VideoJob:
        merged = dict(changes or {})
        merged.update(fields)
        async with self._lock:
            jobs = await asyncio.to_thread(self._read_records)
            payload = jobs.get(job_id)
            if payload is None:
                raise JobNotFoundError(job_id)
            current = _parse_job(payload)
            self._check_expected_status(current, expected_status)
            updated = _apply_changes(current, merged)
            if _job_id(updated) != job_id:
                raise ValueError("A job update cannot change job_id")
            jobs[job_id] = _jsonable(updated)
            await asyncio.to_thread(self._write_records, jobs)
        return _copy_job(updated)

    async def save(
        self, job: VideoJob, *, expected_status: Any | None = None
    ) -> VideoJob:
        """Insert or replace a complete record, with optional status CAS."""

        job = _copy_job(job)
        identifier = _job_id(job)
        async with self._lock:
            jobs = await asyncio.to_thread(self._read_records)
            if identifier in jobs and expected_status is not None:
                self._check_expected_status(
                    _parse_job(jobs[identifier]), expected_status
                )
            elif identifier not in jobs and expected_status is not None:
                raise JobNotFoundError(identifier)
            jobs[identifier] = _jsonable(job)
            await asyncio.to_thread(self._write_records, jobs)
        return _copy_job(job)

    async def transition(
        self,
        job_id: str,
        expected_status: Any,
        new_status: Any,
        **changes: Any,
    ) -> VideoJob:
        """Convenience compare-and-set operation for lifecycle transitions."""

        changes["status"] = new_status
        return await self.update(job_id, changes, expected_status=expected_status)

    @staticmethod
    def _check_expected_status(job: VideoJob, expected_status: Any | None) -> None:
        if expected_status is None:
            return
        actual = _status_value(job)
        expected = _status_value(expected_status)
        if actual != expected:
            raise JobStateConflictError(
                f"Expected status {expected!r}, found {actual!r}"
            )

    def _read_records(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if isinstance(document, dict) and "jobs" in document:
            records = document["jobs"]
        else:
            records = document
        if not isinstance(records, dict):
            raise ValueError(f"Invalid jobs repository at {self.path}")
        return {str(key): dict(value) for key, value in records.items()}

    def _write_records(self, jobs: Mapping[str, Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"jobs": jobs},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


class InMemoryJobRepository:
    """Lock-protected repository fake that does not leak mutable records."""

    def __init__(self, jobs: list[VideoJob] | None = None) -> None:
        self._jobs = {_job_id(job): _copy_job(job) for job in (jobs or [])}
        self._lock = asyncio.Lock()

    async def create(self, job: VideoJob) -> VideoJob:
        job = _copy_job(job)
        identifier = _job_id(job)
        async with self._lock:
            if identifier in self._jobs:
                raise DuplicateJobError(f"Job {identifier!r} already exists")
            self._jobs[identifier] = job
        return _copy_job(job)

    async def get(self, job_id: str) -> VideoJob | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else _copy_job(job)

    async def list(self) -> list[VideoJob]:
        async with self._lock:
            return [_copy_job(job) for job in self._jobs.values()]

    async def list_non_terminal(self) -> list[VideoJob]:
        return [
            job
            for job in await self.list()
            if _status_value(job) not in _TERMINAL_STATUSES
        ]

    async def update(
        self,
        job_id: str,
        changes: Mapping[str, Any] | None = None,
        *,
        expected_status: Any | None = None,
        **fields: Any,
    ) -> VideoJob:
        merged = dict(changes or {})
        merged.update(fields)
        async with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise JobNotFoundError(job_id)
            JsonFileJobRepository._check_expected_status(current, expected_status)
            updated = _apply_changes(current, merged)
            if _job_id(updated) != job_id:
                raise ValueError("A job update cannot change job_id")
            self._jobs[job_id] = _copy_job(updated)
        return _copy_job(updated)

    async def save(
        self, job: VideoJob, *, expected_status: Any | None = None
    ) -> VideoJob:
        job = _copy_job(job)
        identifier = _job_id(job)
        async with self._lock:
            current = self._jobs.get(identifier)
            if current is not None and expected_status is not None:
                JsonFileJobRepository._check_expected_status(current, expected_status)
            elif current is None and expected_status is not None:
                raise JobNotFoundError(identifier)
            self._jobs[identifier] = job
        return _copy_job(job)

    async def transition(
        self,
        job_id: str,
        expected_status: Any,
        new_status: Any,
        **changes: Any,
    ) -> VideoJob:
        changes["status"] = new_status
        return await self.update(job_id, changes, expected_status=expected_status)
