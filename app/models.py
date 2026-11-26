"""Persistence-friendly domain and API models.

All timestamps are timezone-aware and default to UTC.  Calling
``model_dump(mode="json")`` produces a document that can be written directly to
the JSON job repository described in :mod:`ARCHITECTURE.md`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated
from uuid import uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp suitable for persisted job records."""

    return datetime.now(UTC)


class JobStatus(str, Enum):
    """Canonical job lifecycle states."""

    QUEUED = "queued"
    PLANNING = "planning"
    VALIDATING_PLAN = "validating_plan"
    SYNTHESIZING_AUDIO = "synthesizing_audio"
    RENDERING_VISUALS = "rendering_visuals"
    VERIFYING_ARTIFACT = "verifying_artifact"
    COMPLETED = "completed"
    FAILED = "failed"


class VisualTemplate(str, Enum):
    """Renderer templates accepted from a validated storyboard."""

    PH_SCALE_SPECTRUM = "ph_scale_spectrum"
    COVALENT_ELECTRON_SHARING = "covalent_electron_sharing"
    IONIC_VS_COVALENT_SPLIT = "ionic_vs_covalent_split"
    CONCEPT_SUMMARY_CARD = "concept_summary_card"


class DomainModel(BaseModel):
    """Shared validation policy for persisted and externally visible models."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
        from_attributes=True,
    )


NonEmptyText = Annotated[StrictStr, Field(min_length=1)]


class SceneSpec(DomainModel):
    """One narrated scene and its renderer-owned visual instructions."""

    scene_index: Annotated[StrictInt, Field(ge=0, le=5)]
    heading: Annotated[StrictStr, Field(min_length=1, max_length=120)]
    narration: Annotated[StrictStr, Field(min_length=1, max_length=1_500)]
    caption: Annotated[StrictStr, Field(min_length=1, max_length=300)]
    visual_template: VisualTemplate
    visual_params: dict[str, JsonValue] = Field(default_factory=dict)


class Storyboard(DomainModel):
    """Validated storyboard contract consumed by audio and visual stages."""

    topic_id: Annotated[
        StrictStr,
        Field(min_length=1, max_length=100, pattern=r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$"),
    ]
    subject: Annotated[StrictStr, Field(min_length=1, max_length=80)] = "chemistry"
    title: Annotated[StrictStr, Field(min_length=1, max_length=160)]
    learner_query: Annotated[StrictStr, Field(min_length=3, max_length=500)]
    scenes: Annotated[list[SceneSpec], Field(min_length=3, max_length=5)]

    @model_validator(mode="after")
    def validate_scene_order(self) -> Storyboard:
        """Require stable, unique scene indexes in presentation order.

        Both zero-based and one-based indexes are accepted at the API boundary;
        either convention must remain contiguous throughout one storyboard.
        """

        indexes = [scene.scene_index for scene in self.scenes]
        first = indexes[0]
        if first not in (0, 1) or indexes != list(range(first, first + len(indexes))):
            raise ValueError(
                "scene_index values must be contiguous, ordered, and start at 0 or 1"
            )
        return self


class AttemptRecord(DomainModel):
    """Result metadata for one planner candidate."""

    attempt_number: Annotated[StrictInt, Field(ge=1, le=3)]
    provider: Annotated[StrictStr, Field(min_length=1, max_length=100)]
    passed_validation: StrictBool
    issues: list[NonEmptyText] = Field(default_factory=list, max_length=100)
    duration_ms: Annotated[StrictInt, Field(ge=0)]

    @model_validator(mode="after")
    def validate_issues(self) -> AttemptRecord:
        if self.passed_validation and self.issues:
            raise ValueError("a passing attempt cannot contain validation issues")
        return self


class VideoJob(DomainModel):
    """Durable source-of-truth record for a video generation job."""

    job_id: Annotated[StrictStr, Field(min_length=1, max_length=100)] = Field(
        default_factory=lambda: str(uuid4())
    )
    query: Annotated[StrictStr, Field(min_length=3, max_length=500)]
    subject: Annotated[StrictStr, Field(min_length=1, max_length=80)] = "chemistry"
    topic_id: Annotated[StrictStr, Field(min_length=1, max_length=100)] | None = None
    status: JobStatus = JobStatus.QUEUED
    progress_pct: Annotated[StrictInt, Field(ge=0, le=100)] = 0
    attempts: list[AttemptRecord] = Field(default_factory=list, max_length=3)
    used_fallback: StrictBool = False
    error_code: Annotated[StrictStr, Field(min_length=1, max_length=100)] | None = None
    error_message: Annotated[StrictStr, Field(min_length=1, max_length=500)] | None = (
        None
    )
    storyboard: Storyboard | None = None
    artifact_url: Annotated[StrictStr, Field(min_length=1, max_length=2_048)] | None = (
        None
    )
    duration_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    file_size_bytes: Annotated[StrictInt, Field(gt=0)] | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None

    @field_validator("topic_id")
    @classmethod
    def validate_topic_id(cls, value: str | None) -> str | None:
        if value is not None and not all(
            part and part.replace("-", "").replace("_", "").isalnum()
            for part in [value]
        ):
            raise ValueError(
                "topic_id must contain only letters, numbers, hyphens, or underscores"
            )
        return value

    @model_validator(mode="after")
    def validate_record_consistency(self) -> VideoJob:
        if (self.error_code is None) != (self.error_message is None):
            raise ValueError(
                "error_code and error_message must either both be set or both be null"
            )
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at cannot be earlier than created_at")
        if self.completed_at is not None:
            if self.completed_at < self.created_at:
                raise ValueError("completed_at cannot be earlier than created_at")
            if self.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                raise ValueError("completed_at is only valid for a terminal job")
        return self


class CreateVideoRequest(DomainModel):
    """Request body accepted by ``POST /jobs``."""

    question: Annotated[StrictStr, Field(min_length=3, max_length=500)]
    subject: Annotated[StrictStr, Field(min_length=1, max_length=80)] = "chemistry"


class VideoJobSummary(DomainModel):
    """Compact job representation suitable for creation and polling responses."""

    job_id: StrictStr
    query: StrictStr
    subject: StrictStr
    topic_id: StrictStr | None = None
    status: JobStatus
    progress_pct: Annotated[StrictInt, Field(ge=0, le=100)]
    used_fallback: StrictBool
    error_code: StrictStr | None = None
    error_message: StrictStr | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class VideoJobDetail(VideoJobSummary):
    """Full safe API representation returned by ``GET /jobs/{job_id}``."""

    attempts: list[AttemptRecord] = Field(default_factory=list)
    storyboard: Storyboard | None = None
    artifact_url: StrictStr | None = None
    duration_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    file_size_bytes: Annotated[StrictInt, Field(gt=0)] | None = None
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
