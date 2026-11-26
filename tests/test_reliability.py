"""Deterministic reliability tests for storyboard planning and validation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
import sys

import pytest

# The standalone ``pytest.exe`` launcher on Windows does not always prepend the
# current working directory.  Keep this test file runnable directly either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import SceneSpec, Storyboard, VisualTemplate
from app.planner import (
    CuratedChemistryPlanner,
    ResilientPlannerOrchestrator,
    StoryboardPlanningError,
)
from app.topics import TOPIC_REGISTRY, TopicDefinition
from app.validator import validate_storyboard


CURATED_QUERIES = (
    "How does the pH scale work?",
    "Why do atoms form covalent bonds?",
    "What is the difference between ionic and covalent bonding?",
)


def _concept_incomplete_storyboard(query: str, topic: TopicDefinition) -> Storyboard:
    narration = (
        "Chemistry uses measured quantities and diagrams to describe observations "
        "in a clear way."
    )
    return Storyboard(
        topic_id=topic.topic_id,
        subject=topic.subject,
        title=topic.title,
        learner_query=query,
        scenes=[
            SceneSpec(
                scene_index=index,
                heading=f"Generic chemistry scene {index}",
                narration=narration,
                caption=f"Generic chemistry summary {index}",
                visual_template=VisualTemplate.PH_SCALE_SPECTRUM,
                visual_params={"state": "generic", "target_ph": 7},
            )
            for index in range(1, 4)
        ],
    )


def _structurally_bad_storyboard(storyboard: Storyboard) -> Storyboard:
    payload = storyboard.model_dump(mode="python")
    for index, scene in enumerate(payload["scenes"]):
        scene["scene_index"] = index
    payload["scenes"][0]["heading"] = "# Raw Markdown is not render-ready"
    return Storyboard.model_validate(payload)


class FlakyPlanner:
    provider_name = "flaky_mock"

    def __init__(self, invalid: Storyboard, corrected: Storyboard) -> None:
        self.invalid = invalid
        self.corrected = corrected
        self.feedback_received: list[list[str] | None] = []

    async def plan(
        self,
        query: str,
        topic: TopicDefinition,
        feedback_errors: Sequence[str] | None = None,
    ) -> Storyboard:
        del query, topic
        self.feedback_received.append(
            None if feedback_errors is None else list(feedback_errors)
        )
        return self.invalid if len(self.feedback_received) == 1 else self.corrected


class BrokenPlanner:
    provider_name = "broken_mock"

    def __init__(self, candidates: Sequence[Storyboard]) -> None:
        self.candidates = tuple(candidates)
        self.feedback_received: list[list[str] | None] = []

    async def plan(
        self,
        query: str,
        topic: TopicDefinition,
        feedback_errors: Sequence[str] | None = None,
    ) -> Storyboard:
        del query, topic
        self.feedback_received.append(
            None if feedback_errors is None else list(feedback_errors)
        )
        index = min(len(self.feedback_received) - 1, len(self.candidates) - 1)
        return self.candidates[index]


class TimeoutPlanner:
    provider_name = "timeout_mock"

    def __init__(self) -> None:
        self.feedback_received: list[list[str] | None] = []

    async def plan(
        self,
        query: str,
        topic: TopicDefinition,
        feedback_errors: Sequence[str] | None = None,
    ) -> Storyboard:
        del query, topic
        self.feedback_received.append(
            None if feedback_errors is None else list(feedback_errors)
        )
        await asyncio.Event().wait()
        raise AssertionError("the timeout should cancel this planner call")


class UnexpectedFallbackPlanner:
    provider_name = "unexpected_fallback"

    def __init__(self) -> None:
        self.calls = 0

    async def plan(
        self,
        query: str,
        topic: TopicDefinition,
        feedback_errors: Sequence[str] | None = None,
    ) -> Storyboard:
        del query, topic, feedback_errors
        self.calls += 1
        raise AssertionError("fallback must not run after a corrected primary plan")


@pytest.mark.asyncio
@pytest.mark.parametrize("query", CURATED_QUERIES)
async def test_curated_storyboards_validate_without_issues(query: str) -> None:
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    storyboard = await CuratedChemistryPlanner().plan(query, topic)

    assert validate_storyboard(storyboard, topic) == []


@pytest.mark.asyncio
async def test_flaky_planner_receives_exact_errors_then_succeeds() -> None:
    query = CURATED_QUERIES[0]
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    invalid = _concept_incomplete_storyboard(query, topic)
    corrected = await CuratedChemistryPlanner().plan(query, topic)
    expected_errors = validate_storyboard(invalid, topic)
    assert expected_errors
    assert all(error.startswith("missing_concept:") for error in expected_errors)

    primary = FlakyPlanner(invalid, corrected)
    fallback = UnexpectedFallbackPlanner()
    result = await ResilientPlannerOrchestrator(
        primary=primary,
        fallback=fallback,
        max_attempts=2,
        attempt_timeout_seconds=None,
    ).plan(query, topic)

    assert primary.feedback_received == [None, expected_errors]
    assert fallback.calls == 0
    assert result.storyboard == corrected
    assert result.used_fallback is False
    assert len(result.attempts) == 2
    assert result.attempts[0].passed_validation is False
    assert result.attempts[0].issues == expected_errors
    assert result.attempts[1].passed_validation is True
    assert result.attempts[1].issues == []


@pytest.mark.asyncio
async def test_broken_planner_falls_back_and_records_every_failed_attempt() -> None:
    query = CURATED_QUERIES[0]
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    incomplete = _concept_incomplete_storyboard(query, topic)
    bad = _structurally_bad_storyboard(incomplete)
    primary = BrokenPlanner((incomplete, bad))

    result = await ResilientPlannerOrchestrator(
        primary=primary,
        max_attempts=2,
        attempt_timeout_seconds=None,
    ).plan(query, topic)

    assert result.used_fallback is True
    assert validate_storyboard(result.storyboard, topic) == []
    assert len(primary.feedback_received) == 2
    assert primary.feedback_received[0] is None
    assert primary.feedback_received[1] == result.attempts[0].issues
    assert len(result.attempts) == 3
    assert [attempt.provider for attempt in result.attempts] == [
        "broken_mock",
        "broken_mock",
        "curated_chemistry",
    ]
    assert all(
        not attempt.passed_validation and attempt.issues
        for attempt in result.attempts[:2]
    )
    assert any(
        issue.startswith("scene_index:") or issue.startswith("raw_markdown:")
        for issue in result.attempts[1].issues
    )
    assert result.attempts[2].passed_validation is True


@pytest.mark.asyncio
async def test_timeout_planner_falls_back_and_records_every_failed_attempt() -> None:
    query = CURATED_QUERIES[1]
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    primary = TimeoutPlanner()

    result = await ResilientPlannerOrchestrator(
        primary=primary,
        max_attempts=2,
        attempt_timeout_seconds=0.01,
    ).plan(query, topic)

    assert result.used_fallback is True
    assert validate_storyboard(result.storyboard, topic) == []
    assert len(primary.feedback_received) == 2
    assert primary.feedback_received[0] is None
    assert primary.feedback_received[1] == result.attempts[0].issues
    assert len(result.attempts) == 3
    assert all(
        attempt.provider == "timeout_mock"
        and not attempt.passed_validation
        and len(attempt.issues) == 1
        and attempt.issues[0].startswith("planner_timeout:")
        for attempt in result.attempts[:2]
    )
    assert result.attempts[2].provider == "curated_chemistry"
    assert result.attempts[2].passed_validation is True


@pytest.mark.asyncio
async def test_invalid_fallback_is_rejected_instead_of_returned() -> None:
    query = CURATED_QUERIES[0]
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    incomplete = _concept_incomplete_storyboard(query, topic)

    with pytest.raises(StoryboardPlanningError) as caught:
        await ResilientPlannerOrchestrator(
            primary=BrokenPlanner((incomplete,)),
            fallback=BrokenPlanner((incomplete,)),
            max_attempts=2,
            attempt_timeout_seconds=None,
        ).plan(query, topic)

    assert len(caught.value.attempts) == 3
    assert all(
        not attempt.passed_validation and attempt.issues
        for attempt in caught.value.attempts
    )


@pytest.mark.asyncio
async def test_validator_catches_ph_misconception() -> None:
    query = CURATED_QUERIES[0]
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    storyboard = await CuratedChemistryPlanner().plan(query, topic)

    for claim in ("pH 2 is alkaline.", "pH < 7 is basic."):
        payload = storyboard.model_dump(mode="python")
        payload["scenes"][0]["caption"] += f" {claim}"
        misconception = Storyboard.model_validate(payload)

        issues = validate_storyboard(misconception, topic)

        assert any(issue.startswith("forbidden_misconception:") for issue in issues)


@pytest.mark.asyncio
async def test_validator_catches_ionic_equal_sharing_misconception() -> None:
    query = CURATED_QUERIES[2]
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    storyboard = await CuratedChemistryPlanner().plan(query, topic)
    payload = storyboard.model_dump(mode="python")
    payload["scenes"][0]["narration"] += (
        " Ionic bonding is equal sharing of electrons between atoms."
    )
    misconception = Storyboard.model_validate(payload)

    issues = validate_storyboard(misconception, topic)

    assert any(
        issue.startswith("forbidden_misconception:")
        and "equal electron sharing" in issue
        for issue in issues
    )


@pytest.mark.asyncio
async def test_validator_rejects_invalid_template_parameters() -> None:
    query = CURATED_QUERIES[0]
    topic = TOPIC_REGISTRY.require_match(query, subject="chemistry")
    storyboard = await CuratedChemistryPlanner().plan(query, topic)
    storyboard.scenes[0].visual_params["target_ph"] = "not-a-number"

    issues = validate_storyboard(storyboard, topic)

    assert any(
        issue.startswith("visual_params:") and "target_ph" in issue for issue in issues
    )
