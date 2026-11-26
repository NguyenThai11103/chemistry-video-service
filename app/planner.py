"""Storyboard planning adapters and resilient retry orchestration.

Planner output is deliberately treated as untrusted.  The orchestrator is the
only entry point that returns a plan for rendering: every remote or curated
candidate must first pass :func:`app.validator.validate_storyboard`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
import re
from time import monotonic
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

import httpx

from app.models import AttemptRecord, SceneSpec, Storyboard, VisualTemplate
from app.topics import TopicDefinition
from app.validator import validate_storyboard


FeedbackErrors = list[str] | None


@runtime_checkable
class StoryboardPlanner(Protocol):
    """Turn a supported learner query into an untrusted storyboard candidate."""

    async def plan(
        self,
        query: str,
        topic: TopicDefinition,
        feedback_errors: FeedbackErrors = None,
    ) -> Storyboard:
        """Create one candidate, using prior validation errors when supplied."""


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """A validated storyboard and the complete audit trail used to obtain it."""

    storyboard: Storyboard
    attempts: tuple[AttemptRecord, ...]
    used_fallback: bool


# ``PlanningOutcome`` reads naturally at call sites and is retained as an alias
# so integrations can choose either term without creating two result shapes.
PlanningOutcome = PlanningResult


class StoryboardPlanningError(RuntimeError):
    """Raised when no candidate, including the curated fallback, is valid."""

    def __init__(self, message: str, attempts: Sequence[AttemptRecord]) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


# Short alias useful to callers that do not need the domain-specific prefix.
PlanningError = StoryboardPlanningError


class CuratedChemistryPlanner:
    """Reviewed deterministic storyboards for the prototype chemistry topics."""

    provider_name = "curated_chemistry"

    async def plan(
        self,
        query: str,
        topic: TopicDefinition,
        feedback_errors: FeedbackErrors = None,
    ) -> Storyboard:
        del feedback_errors  # The reviewed plan is deterministic by design.
        try:
            scenes = _CURATED_SCENES[topic.topic_id]()
        except KeyError as exc:
            raise LookupError(
                f"no curated storyboard is registered for topic {topic.qualified_id}"
            ) from exc

        return Storyboard(
            topic_id=topic.topic_id,
            subject=topic.subject,
            title=topic.title,
            learner_query=query,
            scenes=scenes,
        )


def _ph_scale_scenes() -> list[SceneSpec]:
    return [
        SceneSpec(
            scene_index=1,
            heading="Reading an acidic value",
            narration=(
                "The pH scale is a logarithmic scale: pH is the negative log base "
                "10 of hydrogen ion concentration, written pH = -log10[H+]."
            ),
            caption="pH = -log10[H+]",
            visual_template=VisualTemplate.PH_SCALE_SPECTRUM,
            visual_params={
                "state": "acidic",
                "ph": 2,
                "target_ph": 2,
                "marker_label": "Acidic: pH 2",
                "scale_min": 0,
                "scale_max": 14,
                "color": "#D94B4B",
            },
        ),
        SceneSpec(
            scene_index=2,
            heading="Why one unit matters",
            narration=(
                "A one-unit step represents a tenfold change in H+ concentration. "
                "For example, pH 2 has 10 times the hydrogen ion concentration of pH 3."
            ),
            caption="1 pH unit = a factor of ten in [H+]",
            visual_template=VisualTemplate.PH_SCALE_SPECTRUM,
            visual_params={
                "state": "acidic_tenfold_comparison",
                "ph": 2,
                "target_ph": 2,
                "comparison_ph": 3,
                "concentration_ratio": 10,
                "highlight_tenfold": True,
                "color": "#E26A45",
            },
        ),
        SceneSpec(
            scene_index=3,
            heading="The neutral reference",
            narration=(
                "Neutral water is pH 7 at 25 C. Below 7 is acidic and above 7 is "
                "basic, so the direction tracks decreasing hydrogen-ion concentration."
            ),
            caption="At about 25 C: acidic < 7, neutral = 7, basic > 7",
            visual_template=VisualTemplate.PH_SCALE_SPECTRUM,
            visual_params={
                "state": "neutral",
                "ph": 7,
                "target_ph": 7,
                "marker_label": "Neutral water: pH 7",
                "temperature_c": 25,
                "color": "#4FAE74",
            },
        ),
        SceneSpec(
            scene_index=4,
            heading="A basic example and the big picture",
            narration=(
                "A pH 12 solution is basic and has much less H+ than neutral water. "
                "Concentrated solutions can also extend beyond the familiar classroom range."
            ),
            caption="Basic example: pH 12; pH values can extend beyond 0 to 14",
            visual_template=VisualTemplate.PH_SCALE_SPECTRUM,
            visual_params={
                "state": "basic",
                "ph": 12,
                "target_ph": 12,
                "marker_label": "Basic: pH 12",
                "takeaways": [
                    "logarithmic H+ measure",
                    "tenfold per pH unit",
                    "pH 7 neutral near 25 C",
                ],
                "color": "#5669C9",
            },
        ),
    ]


def _covalent_bond_scenes() -> list[SceneSpec]:
    return [
        SceneSpec(
            scene_index=1,
            heading="Atoms approach",
            narration=(
                "As two hydrogen atoms approach, each nucleus attracts the other atom's "
                "electron, so attraction between nuclei and shared electrons begins to "
                "outweigh repulsions."
            ),
            caption="Two H atoms approach: attractions and repulsions change",
            visual_template=VisualTemplate.COVALENT_ELECTRON_SHARING,
            visual_params={
                "state": "approaching_h_atoms",
                "atoms": ["H", "H"],
                "separation": "approaching",
                "shared_electrons": 0,
            },
        ),
        SceneSpec(
            scene_index=2,
            heading="A shared H2 pair",
            narration=(
                "At a suitable distance, the atoms share electrons as shared electron "
                "density between them, creating H2 in a lower energy arrangement than "
                "separated atoms."
            ),
            caption="H-H: a shared electron pair lowers the system energy",
            visual_template=VisualTemplate.COVALENT_ELECTRON_SHARING,
            visual_params={
                "state": "shared_h2_pair",
                "molecule": "H2",
                "atoms": ["H", "H"],
                "shared_electrons": 2,
                "energy": "lower",
            },
        ),
        SceneSpec(
            scene_index=3,
            heading="Sharing in water",
            narration=(
                "In H2O, oxygen shares electron density with two hydrogens. The two "
                "O-H bonds illustrate that covalent sharing can build larger, shaped molecules."
            ),
            caption="H-O-H: two covalent bonds in a bent molecule",
            visual_template=VisualTemplate.COVALENT_ELECTRON_SHARING,
            visual_params={
                "state": "h2o",
                "molecule": "H2O",
                "central_atom": "O",
                "outer_atoms": ["H", "H"],
                "bond_angle_degrees": 104.5,
                "shared_pairs": 2,
            },
        ),
        SceneSpec(
            scene_index=4,
            heading="A useful rule, not the cause",
            narration=(
                "The octet rule is a heuristic for organizing many main-group structures, "
                "but it has exceptions; lower energy and electrostatic interactions provide "
                "the deeper explanation."
            ),
            caption="Octet guideline: useful, with exceptions",
            visual_template=VisualTemplate.CONCEPT_SUMMARY_CARD,
            visual_params={
                "state": "covalent_summary",
                "takeaways": [
                    "shared electron density",
                    "balance of attractions and repulsions",
                    "lower-energy bonded arrangement",
                    "octet heuristic has exceptions",
                ],
            },
        ),
    ]


def _ionic_vs_covalent_scenes() -> list[SceneSpec]:
    return [
        SceneSpec(
            scene_index=1,
            heading="Ionic model: Na to Cl",
            narration=(
                "In the ionic model, an electron transfers from Na to Cl, producing "
                "positive and negative ions held together by electrostatic attraction "
                "between opposite charges."
            ),
            caption="Na -> Cl electron transfer creates Na+ and Cl-",
            visual_template=VisualTemplate.IONIC_VS_COVALENT_SPLIT,
            visual_params={
                "state": "na_to_cl_transfer",
                "from_atom": "Na",
                "to_atom": "Cl",
                "electron_count": 1,
                "products": ["Na+", "Cl-"],
            },
        ),
        SceneSpec(
            scene_index=2,
            heading="Ions form an extended solid",
            narration=(
                "Many ions repeat through an extended ionic lattice rather than existing "
                "as isolated NaCl molecules, with each ion attracted to several oppositely "
                "charged neighbors."
            ),
            caption="NaCl forms an extended ionic lattice",
            visual_template=VisualTemplate.IONIC_VS_COVALENT_SPLIT,
            visual_params={
                "state": "ionic_lattice",
                "formula_unit": "NaCl",
                "ions": ["Na+", "Cl-"],
                "arrangement": "alternating_extended_lattice",
            },
        ),
        SceneSpec(
            scene_index=3,
            heading="Covalent model: a shared Cl2 pair",
            narration=(
                "In Cl2, the atoms share a shared electron pair, concentrating shared "
                "electron density between the nuclei instead of forming separate positive "
                "and negative ions."
            ),
            caption="Cl-Cl: one shared covalent electron pair",
            visual_template=VisualTemplate.IONIC_VS_COVALENT_SPLIT,
            visual_params={
                "state": "shared_cl2_pair",
                "molecule": "Cl2",
                "atoms": ["Cl", "Cl"],
                "shared_electrons": 2,
            },
        ),
        SceneSpec(
            scene_index=4,
            heading="Compare without absolutes",
            narration=(
                "Bonding is a continuum, so real bonds may have partial ionic and covalent "
                "character. Observable properties depend on structure, and exceptions to "
                "property trends are important."
            ),
            caption="Bond character is continuous; properties depend on structure",
            visual_template=VisualTemplate.CONCEPT_SUMMARY_CARD,
            visual_params={
                "state": "comparison_summary",
                "ionic_emphasis": "ions and electrostatic attraction",
                "covalent_emphasis": "shared electron density",
                "continuum": True,
                "properties": "structure_dependent_trends",
            },
        ),
    ]


_CURATED_SCENES = {
    "ph_scale": _ph_scale_scenes,
    "covalent_bonds": _covalent_bond_scenes,
    "ionic_vs_covalent": _ionic_vs_covalent_scenes,
}


class LLMStoryboardPlanner:
    """Direct JSON-schema HTTP adapter for Gemini or OpenAI.

    The adapter owns and closes a client it creates.  A caller-supplied client
    remains caller-owned, which keeps test transports and shared connection
    pools usable after this planner is closed.
    """

    OPENAI_BASE_URL = "https://api.openai.com"
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
    OPENAI_MODEL = "gpt-5-mini"
    GEMINI_MODEL = "gemini-3.8-flash"

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        provider_key = (provider or _provider_from_environment()).strip().casefold()
        if provider_key not in {"gemini", "openai"}:
            raise ValueError("provider must be 'gemini' or 'openai'")

        env_prefix = provider_key.upper()
        resolved_key = api_key or os.getenv(f"{env_prefix}_API_KEY")
        if not resolved_key:
            raise ValueError(f"{env_prefix}_API_KEY is required for {provider_key}")

        default_model = (
            self.GEMINI_MODEL if provider_key == "gemini" else self.OPENAI_MODEL
        )
        default_base_url = (
            self.GEMINI_BASE_URL if provider_key == "gemini" else self.OPENAI_BASE_URL
        )
        self.provider = provider_key
        self.provider_name = provider_key
        self.api_key = resolved_key
        self.model = model or os.getenv(f"{env_prefix}_MODEL") or default_model
        self.base_url = (
            base_url or os.getenv(f"{env_prefix}_BASE_URL") or default_base_url
        ).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None
        self._closed = False

    async def __aenter__(self) -> LLMStoryboardPlanner:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close only an internally created HTTP client."""

        if self._owns_client and self._client is not None and not self._closed:
            await self._client.aclose()
        self._closed = True

    async def plan(
        self,
        query: str,
        topic: TopicDefinition,
        feedback_errors: FeedbackErrors = None,
    ) -> Storyboard:
        if self._closed:
            raise RuntimeError("LLMStoryboardPlanner is closed")

        prompt = _storyboard_prompt(query, topic, feedback_errors)
        schema = _schema_without_defaults(Storyboard.model_json_schema())
        client = self._get_client()

        if self.provider == "gemini":
            raw_text = await self._request_gemini(client, prompt, schema)
        else:
            raw_text = await self._request_openai(client, prompt, schema)

        payload = _parse_json_object(raw_text)
        # Provider output cannot select another rule pack or alter the learner's
        # source question.  Those values come from trusted orchestration input.
        payload["topic_id"] = topic.topic_id
        payload["subject"] = topic.subject
        payload["learner_query"] = query
        return Storyboard.model_validate(payload)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        return self._client

    async def _request_gemini(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        schema: dict[str, Any],
    ) -> str:
        model_path = quote(self.model, safe="-._")
        url = _versioned_url(
            self.base_url, "v1beta", f"models/{model_path}:generateContent"
        )
        response = await client.post(
            url,
            headers={"x-goog-api-key": self.api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        try:
            parts = body["candidates"][0]["content"]["parts"]
            text_parts = [
                part["text"]
                for part in parts
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            ]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Gemini response did not contain candidate text") from exc
        if not text_parts:
            raise ValueError("Gemini response contained empty candidate text")
        return "".join(text_parts)

    async def _request_openai(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        schema: dict[str, Any],
    ) -> str:
        url = _versioned_url(self.base_url, "v1", "responses")
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": prompt,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "storyboard",
                        # visual_params intentionally accepts template-specific
                        # JSON keys. Provider-strict schemas require every
                        # object to be closed, so Pydantic plus the local
                        # storyboard validator remain the authoritative gates.
                        "schema": schema,
                        "strict": False,
                    }
                },
            },
        )
        response.raise_for_status()
        body = response.json()

        text_parts: list[str] = []
        for output in body.get("output", []):
            if not isinstance(output, Mapping):
                continue
            for content in output.get("content", []):
                if (
                    isinstance(content, Mapping)
                    and content.get("type") == "output_text"
                    and isinstance(content.get("text"), str)
                ):
                    text_parts.append(content["text"])
        # Some compatible endpoints expose this convenience field directly.
        if not text_parts and isinstance(body.get("output_text"), str):
            text_parts.append(body["output_text"])
        if not text_parts:
            raise ValueError("OpenAI response did not contain output_text")
        return "".join(text_parts)


class ResilientPlannerOrchestrator:
    """Validate bounded primary attempts, then validate a curated fallback."""

    def __init__(
        self,
        primary: StoryboardPlanner | None = None,
        fallback: StoryboardPlanner | None = None,
        *,
        max_attempts: int = 2,
        attempt_timeout_seconds: float | None = 45.0,
    ) -> None:
        if not 1 <= max_attempts <= 2:
            raise ValueError("max_attempts must be 1 or 2")
        if attempt_timeout_seconds is not None and attempt_timeout_seconds <= 0:
            raise ValueError("attempt_timeout_seconds must be positive or None")

        self.primary = primary if primary is not None else default_storyboard_planner()
        self.fallback = fallback if fallback is not None else CuratedChemistryPlanner()
        self.max_attempts = max_attempts
        self.attempt_timeout_seconds = attempt_timeout_seconds

    async def plan(self, query: str, topic: TopicDefinition) -> PlanningResult:
        attempts: list[AttemptRecord] = []
        feedback_errors: list[str] | None = None

        for attempt_number in range(1, self.max_attempts + 1):
            candidate, issues, duration_ms = await self._attempt(
                self.primary,
                query,
                topic,
                feedback_errors,
            )
            attempts.append(
                AttemptRecord(
                    attempt_number=attempt_number,
                    provider=_planner_provider_name(self.primary),
                    passed_validation=not issues,
                    issues=issues,
                    duration_ms=duration_ms,
                )
            )
            if candidate is not None and not issues:
                return PlanningResult(candidate, tuple(attempts), used_fallback=False)

            # Copy without altering, prefixing, sorting, or deduplicating.  The
            # next planner receives the validator's exact issue strings.
            feedback_errors = list(issues)

        fallback_number = self.max_attempts + 1
        candidate, issues, duration_ms = await self._attempt(
            self.fallback,
            query,
            topic,
            feedback_errors,
        )
        attempts.append(
            AttemptRecord(
                attempt_number=fallback_number,
                provider=_planner_provider_name(self.fallback),
                passed_validation=not issues,
                issues=issues,
                duration_ms=duration_ms,
            )
        )
        if candidate is not None and not issues:
            return PlanningResult(candidate, tuple(attempts), used_fallback=True)

        detail = "; ".join(issues) if issues else "unknown validation failure"
        raise StoryboardPlanningError(
            f"curated fallback failed planning or validation for "
            f"{topic.qualified_id}: {detail}",
            attempts,
        )

    async def _attempt(
        self,
        planner: StoryboardPlanner,
        query: str,
        topic: TopicDefinition,
        feedback_errors: FeedbackErrors,
    ) -> tuple[Storyboard | None, list[str], int]:
        started = monotonic()
        candidate: Storyboard | None = None
        try:
            operation = planner.plan(query, topic, feedback_errors)
            if self.attempt_timeout_seconds is None:
                candidate = await operation
            else:
                candidate = await asyncio.wait_for(
                    operation, timeout=self.attempt_timeout_seconds
                )
            issues = list(validate_storyboard(candidate, topic))
        except TimeoutError as exc:
            timeout = self.attempt_timeout_seconds
            if timeout is None:
                issues = [f"planner_exception: {type(exc).__name__}: {exc}"]
            else:
                issues = [f"planner_timeout: attempt exceeded {timeout:g} seconds"]
        except Exception as exc:  # Provider/schema failures are retryable here.
            issues = [f"planner_exception: {type(exc).__name__}: {exc}"]

        duration_ms = max(0, int((monotonic() - started) * 1000))
        return candidate, issues, duration_ms


def default_storyboard_planner() -> StoryboardPlanner:
    """Select a configured remote planner, otherwise use curated plans directly."""

    if os.getenv("GEMINI_API_KEY"):
        return LLMStoryboardPlanner(provider="gemini")
    if os.getenv("OPENAI_API_KEY"):
        return LLMStoryboardPlanner(provider="openai")
    return CuratedChemistryPlanner()


# Compatibility-friendly name for dependency-injection setup code.
build_default_planner = default_storyboard_planner


def _provider_from_environment() -> str:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise ValueError("set GEMINI_API_KEY or OPENAI_API_KEY to use an LLM planner")


def _planner_provider_name(planner: StoryboardPlanner) -> str:
    value = getattr(planner, "provider_name", None) or getattr(
        planner, "provider", None
    )
    if not isinstance(value, str) or not value.strip():
        value = type(planner).__name__
    return value.strip()[:100]


def _versioned_url(base_url: str, version: str, path: str) -> str:
    """Join provider bases whether or not they already include the API version."""

    base = base_url.rstrip("/")
    if base.casefold().endswith(f"/{version.casefold()}"):
        return f"{base}/{path.lstrip('/')}"
    return f"{base}/{version}/{path.lstrip('/')}"


def _storyboard_prompt(
    query: str,
    topic: TopicDefinition,
    feedback_errors: FeedbackErrors,
) -> str:
    concept_groups = [
        {"concept_id": group.concept_id, "accepted_phrases": list(group.synonyms)}
        for group in topic.required_concept_groups
    ]
    prompt = (
        "Create a pedagogically clear chemistry storyboard. Return only JSON that "
        "matches the supplied Storyboard JSON schema. Use 3 to 5 ordered scenes; "
        "each narration must contain 12 to 50 words. Do not emit Markdown.\n"
        f"Learner query: {query}\n"
        f"Required topic_id: {topic.topic_id}\n"
        f"Required subject: {topic.subject}\n"
        f"Title: {topic.title}\n"
        "Allowed visual_template values: "
        f"{json.dumps(sorted(item.value for item in topic.allowed_visual_templates))}\n"
        "Across the narration, include at least one exact accepted phrase for every "
        f"concept group: {json.dumps(concept_groups, ensure_ascii=False)}\n"
        "Never state these forbidden misconceptions: "
        f"{json.dumps(list(topic.forbidden_misconceptions), ensure_ascii=False)}"
    )
    if feedback_errors is not None:
        # JSON preserves every issue exactly, including punctuation and locations.
        prompt += (
            "\nThe previous candidate failed validation. Fix every issue exactly as "
            "reported here: " + json.dumps(list(feedback_errors), ensure_ascii=False)
        )
    return prompt


def _schema_without_defaults(value: Any) -> Any:
    """Recursively remove Pydantic defaults rejected by strict schema providers."""

    if isinstance(value, Mapping):
        return {
            key: _schema_without_defaults(item)
            for key, item in value.items()
            if key != "default"
        }
    if isinstance(value, list):
        return [_schema_without_defaults(item) for item in value]
    return value


_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```", flags=re.IGNORECASE | re.DOTALL
)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from plain, fenced, or lightly wrapped model output."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("planner response was empty")

    candidates = [match.group("body") for match in _FENCED_JSON_RE.finditer(raw_text)]
    candidates.append(raw_text.strip())
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        else:
            if isinstance(parsed, dict):
                return parsed

        # Providers occasionally surround otherwise valid JSON with a sentence.
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError(
        "planner response did not contain a valid JSON object"
    ) from last_error


__all__ = [
    "CuratedChemistryPlanner",
    "LLMStoryboardPlanner",
    "PlanningError",
    "PlanningOutcome",
    "PlanningResult",
    "ResilientPlannerOrchestrator",
    "StoryboardPlanner",
    "StoryboardPlanningError",
    "build_default_planner",
    "default_storyboard_planner",
]
