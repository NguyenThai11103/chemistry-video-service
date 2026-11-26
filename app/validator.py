"""Deterministic storyboard and rendered-media quality gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from .models import SceneSpec, Storyboard, VisualTemplate
from .topics import TopicDefinition, normalize_question


MIN_VIDEO_SIZE_BYTES = 30 * 1024
MAX_VIDEO_SIZE_BYTES = 6 * 1024 * 1024
MIN_VIDEO_DURATION_SECONDS = 12.0
MAX_DURATION_DELTA_SECONDS = 3.5
FFPROBE_TIMEOUT_SECONDS = 10.0

_WORD_RE = re.compile(r"[^\W_]+(?:[\u2019'-][^\W_]+)*", re.UNICODE)
_MARKDOWN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code fence", re.compile(r"```|~~~")),
    ("heading", re.compile(r"(?m)^\s{0,3}#{1,6}(?:\s|$)")),
    ("list", re.compile(r"(?m)^\s{0,3}(?:[-+*]|\d+[.)])\s+")),
    ("blockquote", re.compile(r"(?m)^\s{0,3}>\s?")),
    ("link or image", re.compile(r"!?\[[^\]\r\n]+\]\([^\)\r\n]+\)")),
    ("emphasis", re.compile(r"(?:\*\*|__|~~)[^\r\n]+?(?:\*\*|__|~~)")),
    ("inline code", re.compile(r"`[^`\r\n]+`")),
)


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_question(phrase)
    return bool(normalized_phrase) and (
        f" {normalized_phrase} " in f" {normalized_text} "
    )


def _word_count(value: str) -> int:
    return len(_WORD_RE.findall(value))


def _markdown_kind(value: str) -> str | None:
    for label, pattern in _MARKDOWN_RULES:
        if pattern.search(value):
            return label
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _visual_param_issues(scene: SceneSpec, location: str) -> list[str]:
    """Validate the bounded renderer contract for known template parameters."""

    params = scene.visual_params
    issues: list[str] = []
    state = params.get("state")
    if state is not None and (
        not isinstance(state, str) or not state.strip() or len(state) > 64
    ):
        issues.append(
            f"visual_params: {location}.visual_params.state must be a non-empty "
            "string of at most 64 characters."
        )

    marker_label = params.get("marker_label")
    if marker_label is not None and (
        not isinstance(marker_label, str) or len(marker_label.strip()) > 48
    ):
        issues.append(
            f"visual_params: {location}.visual_params.marker_label must be a "
            "string of at most 48 characters."
        )

    if scene.visual_template == VisualTemplate.PH_SCALE_SPECTRUM:
        target_key = "target_ph" if "target_ph" in params else "ph"
        target = params.get(target_key)
        number = _finite_number(target)
        if number is None or not 0 <= number <= 14:
            issues.append(
                f"visual_params: {location}.visual_params.{target_key} must be "
                "a finite number from 0 through 14."
            )
        for key in ("comparison_ph",):
            if key in params:
                comparison = _finite_number(params[key])
                if comparison is None or not 0 <= comparison <= 14:
                    issues.append(
                        f"visual_params: {location}.visual_params.{key} must be "
                        "a finite number from 0 through 14."
                    )
        if "concentration_ratio" in params:
            ratio = _finite_number(params["concentration_ratio"])
            if ratio is None or ratio <= 0:
                issues.append(
                    f"visual_params: {location}.visual_params.concentration_ratio "
                    "must be a positive finite number."
                )
        if "highlight_tenfold" in params and not isinstance(
            params["highlight_tenfold"], bool
        ):
            issues.append(
                f"visual_params: {location}.visual_params.highlight_tenfold must "
                "be a boolean."
            )

    elif scene.visual_template == VisualTemplate.COVALENT_ELECTRON_SHARING:
        for key in ("atoms", "outer_atoms"):
            if key not in params:
                continue
            atoms = params[key]
            if (
                not isinstance(atoms, list)
                or len(atoms) != 2
                or any(
                    not isinstance(atom, str) or not atom.strip() or len(atom) > 4
                    for atom in atoms
                )
            ):
                issues.append(
                    f"visual_params: {location}.visual_params.{key} must contain "
                    "exactly two short atom labels."
                )
        if "shared_electrons" in params:
            count = params["shared_electrons"]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 8
            ):
                issues.append(
                    f"visual_params: {location}.visual_params.shared_electrons "
                    "must be an integer from 0 through 8."
                )
        if "bond_angle_degrees" in params:
            angle = _finite_number(params["bond_angle_degrees"])
            if angle is None or not 0 < angle < 180:
                issues.append(
                    f"visual_params: {location}.visual_params.bond_angle_degrees "
                    "must be between 0 and 180."
                )

    elif scene.visual_template == VisualTemplate.IONIC_VS_COVALENT_SPLIT:
        if "electron_count" in params:
            count = params["electron_count"]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 8
            ):
                issues.append(
                    f"visual_params: {location}.visual_params.electron_count must "
                    "be an integer from 0 through 8."
                )
        for key in ("products", "ions"):
            if key not in params:
                continue
            labels = params[key]
            if (
                not isinstance(labels, list)
                or not 1 <= len(labels) <= 12
                or any(
                    not isinstance(label, str) or not label.strip() or len(label) > 12
                    for label in labels
                )
            ):
                issues.append(
                    f"visual_params: {location}.visual_params.{key} must contain "
                    "1 to 12 short ion labels."
                )

    elif scene.visual_template == VisualTemplate.CONCEPT_SUMMARY_CARD:
        if "takeaways" in params:
            takeaways = params["takeaways"]
            if (
                not isinstance(takeaways, list)
                or not 1 <= len(takeaways) <= 4
                or any(
                    not isinstance(item, str) or not item.strip() or len(item) > 90
                    for item in takeaways
                )
            ):
                issues.append(
                    f"visual_params: {location}.visual_params.takeaways must contain "
                    "1 to 4 non-empty strings of at most 90 characters."
                )

    return issues


def _semantic_misconception(text: str) -> str | None:
    """Recognize common paraphrases not covered by literal topic phrases."""

    normalized = normalize_question(text)

    below_seven = (
        r"(?:ph(?: values?)?\s+)?(?:below|under|less than|lower than)\s+(?:a\s+)?7"
    )
    basic = r"(?:basic|alkaline)"
    below_match = re.search(
        rf"\b{below_seven}\b(?P<middle>(?:\s+\w+){{0,5}})\s+\b{basic}\b",
        normalized,
    )
    if below_match and "not" not in below_match.group("middle").split():
        return "pH below 7 is described as basic or alkaline"
    below_match = re.search(
        rf"\b{basic}\b(?P<middle>(?:\s+\w+){{0,5}})\s+\b{below_seven}\b",
        normalized,
    )
    if below_match and "not" not in below_match.group("middle").split():
        return "pH below 7 is described as basic or alkaline"

    symbolic_ph_claim = re.compile(
        r"\bph\s*(?P<operator><=?|≤|>=?|≥)\s*7\s*"
        r"(?:is|means|indicates|represents|would\s+be|is\s+called|"
        r"is\s+classified\s+as)\s+(?:an?\s+)?"
        r"(?P<label>basic|alkaline|acidic)\b",
        flags=re.IGNORECASE,
    )
    symbolic_match = symbolic_ph_claim.search(text)
    if symbolic_match:
        operator = symbolic_match.group("operator")
        label = symbolic_match.group("label").casefold()
        if operator in {"<", "<=", "≤"} and label in {"basic", "alkaline"}:
            return "pH below 7 is described as basic or alkaline"
        if operator in {">", ">=", "≥"} and label == "acidic":
            return "pH above 7 is described as acidic"

    # Requiring a linking verb after the number avoids treating ranges such as
    # "pH 2 to 8 runs from acidic to alkaline" as claims about pH 2.
    numeric_ph_claim = re.compile(
        r"\bph\s*(?:value\s*)?(?:of\s*|=\s*)?"
        r"(?P<value>-?\d+(?:\.\d+)?)\s*"
        r"(?:solution\s+|sample\s+|substance\s+|value\s+)?"
        r"(?:is|means|indicates|represents|would\s+be|is\s+called|"
        r"is\s+classified\s+as)\s+(?:an?\s+)?"
        r"(?P<label>basic|alkaline|acidic)\b",
        flags=re.IGNORECASE,
    )
    for claim in numeric_ph_claim.finditer(text):
        value = float(claim.group("value"))
        label = claim.group("label").casefold()
        if value < 7 and label in {"basic", "alkaline"}:
            return f"pH {value:g} is described as basic or alkaline"
        if value > 7 and label == "acidic":
            return f"pH {value:g} is described as acidic"

    ionic_subject = r"ionic(?:\s+(?:bond|bonds|bonding|model|interaction))?"
    equal_sharing = (
        r"(?:equal(?:ly)?\s+(?:share|shared|sharing)(?:\s+of)?"
        r"|(?:electrons?|electron density)\s+(?:is|are)\s+shared\s+equally)"
    )
    relation = r"(?:is|are|means?|involves?|uses?|requires?|occurs? through|works? by)"
    if re.search(
        rf"\b{ionic_subject}\b\s+(?:\w+\s+){{0,2}}{relation}\s+(?:the\s+)?{equal_sharing}\b",
        normalized,
    ):
        return "ionic bonding is described as equal electron sharing"
    if re.search(
        rf"\b{equal_sharing}\b\s+(?:\w+\s+){{0,2}}(?:defines?|describes?|forms?|is)\s+{ionic_subject}\b",
        normalized,
    ):
        return "ionic bonding is described as equal electron sharing"

    return None


def validate_storyboard(storyboard: Storyboard, topic: TopicDefinition) -> list[str]:
    """Return stable, actionable issues for a planner-produced storyboard."""

    issues: list[str] = []
    scenes = storyboard.scenes
    scene_count = len(scenes)

    if not 3 <= scene_count <= 5:
        issues.append(
            "scene_count: storyboard.scenes must contain 3 to 5 scenes; "
            f"found {scene_count}."
        )

    indexes = [scene.scene_index for scene in scenes]
    expected_indexes = list(range(1, scene_count + 1))
    if indexes != expected_indexes:
        issues.append(
            "scene_index: storyboard.scenes must use ordered one-based indexes "
            f"{expected_indexes}; found {indexes}."
        )

    allowed_templates = topic.allowed_visual_templates
    allowed_names = ", ".join(sorted(template.value for template in allowed_templates))
    combined_parts: list[str] = []

    for position, scene in enumerate(scenes, start=1):
        location = f"scenes[{position - 1}]"
        narration_words = _word_count(scene.narration)
        if not 12 <= narration_words <= 50:
            issues.append(
                f"narration_word_count: {location}.narration must contain 12 to 50 "
                f"words; found {narration_words}."
            )
        if len(scene.heading) > 70:
            issues.append(
                f"heading_length: {location}.heading must be at most 70 characters; "
                f"found {len(scene.heading)}."
            )
        if len(scene.caption) > 115:
            issues.append(
                f"caption_length: {location}.caption must be at most 115 characters; "
                f"found {len(scene.caption)}."
            )

        for field_name in ("heading", "narration", "caption"):
            markdown_kind = _markdown_kind(getattr(scene, field_name))
            if markdown_kind is not None:
                issues.append(
                    f"raw_markdown: {location}.{field_name} contains a Markdown "
                    f"{markdown_kind}; use plain text only."
                )

        if scene.visual_template not in allowed_templates:
            template_name = getattr(
                scene.visual_template, "value", str(scene.visual_template)
            )
            issues.append(
                f"visual_template: {location}.visual_template {template_name!r} is not "
                f"allowed for topic {topic.topic_id!r}; use one of: {allowed_names}."
            )

        issues.extend(_visual_param_issues(scene, location))

        combined_parts.extend((scene.narration, scene.caption))

    combined_text = "\n".join(combined_parts)
    normalized_combined = normalize_question(combined_text)
    for group in topic.required_concept_groups:
        if not any(
            _contains_phrase(normalized_combined, synonym) for synonym in group.synonyms
        ):
            examples = "; ".join(group.synonyms)
            issues.append(
                f"missing_concept: narration/captions must cover concept "
                f"{group.concept_id!r}; accepted wording includes: {examples}."
            )

    for misconception in topic.forbidden_misconceptions:
        if _contains_phrase(normalized_combined, misconception):
            issues.append(
                "forbidden_misconception: narration/captions contain the rejected "
                f"claim {misconception!r}; correct or remove it."
            )

    semantic_issue = _semantic_misconception(combined_text)
    if semantic_issue is not None:
        issues.append(
            "forbidden_misconception: narration/captions imply that "
            f"{semantic_issue}; correct the scientific relationship."
        )

    return issues


@dataclass(frozen=True, slots=True)
class MediaVerificationResult:
    """Structured outcome of the non-throwing rendered-video quality gate."""

    passed: bool
    issues: tuple[str, ...]
    file_size_bytes: int | None = None
    duration_seconds: float | None = None
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None
    audio_codec: str | None = None

    @property
    def valid(self) -> bool:
        """Alias useful to callers that express quality gates as validity."""

        return self.passed

    @property
    def errors(self) -> tuple[str, ...]:
        """Alias for consumers that call failed checks errors."""

        return self.issues


def _media_result(
    issues: list[str],
    *,
    file_size_bytes: int | None = None,
    duration_seconds: float | None = None,
    video_codec: str | None = None,
    width: int | None = None,
    height: int | None = None,
    audio_codec: str | None = None,
) -> MediaVerificationResult:
    return MediaVerificationResult(
        passed=not issues,
        issues=tuple(issues),
        file_size_bytes=file_size_bytes,
        duration_seconds=duration_seconds,
        video_codec=video_codec,
        width=width,
        height=height,
        audio_codec=audio_codec,
    )


def _positive_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def verify_rendered_video(
    video_path: str | Path, expected_duration_sec: float
) -> MediaVerificationResult:
    """Inspect an MP4 with ffprobe and return all ordinary verification failures."""

    expected_duration = _positive_finite_float(expected_duration_sec)
    if expected_duration is None:
        raise ValueError("expected_duration_sec must be a positive finite number")

    path = Path(video_path)
    issues: list[str] = []
    try:
        if not path.is_file():
            return _media_result(
                [f"file_missing: rendered video does not exist: {path}."]
            )
        file_size = path.stat().st_size
    except OSError as exc:
        return _media_result(
            [f"file_unreadable: cannot inspect rendered video {path}: {exc}."]
        )

    if not MIN_VIDEO_SIZE_BYTES <= file_size <= MAX_VIDEO_SIZE_BYTES:
        issues.append(
            "file_size: rendered video must be between "
            f"{MIN_VIDEO_SIZE_BYTES} and {MAX_VIDEO_SIZE_BYTES} bytes inclusive; "
            f"found {file_size}."
        )

    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=FFPROBE_TIMEOUT_SECONDS,
            shell=False,
        )
    except FileNotFoundError:
        issues.append("ffprobe_unavailable: ffprobe was not found on PATH.")
        return _media_result(issues, file_size_bytes=file_size)
    except subprocess.TimeoutExpired:
        issues.append(
            f"ffprobe_timeout: ffprobe exceeded {FFPROBE_TIMEOUT_SECONDS:g} seconds."
        )
        return _media_result(issues, file_size_bytes=file_size)
    except OSError as exc:
        issues.append(f"ffprobe_error: ffprobe could not be started: {exc}.")
        return _media_result(issues, file_size_bytes=file_size)

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().replace("\r", " ").replace("\n", " ")
        if len(detail) > 240:
            detail = detail[:237] + "..."
        suffix = f" Details: {detail}" if detail else ""
        issues.append(
            f"ffprobe_failed: ffprobe exited with code {completed.returncode}.{suffix}"
        )
        return _media_result(issues, file_size_bytes=file_size)

    try:
        probe = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        issues.append("ffprobe_json: ffprobe returned malformed JSON.")
        return _media_result(issues, file_size_bytes=file_size)
    if not isinstance(probe, dict):
        issues.append("ffprobe_json: ffprobe JSON root must be an object.")
        return _media_result(issues, file_size_bytes=file_size)

    streams = probe.get("streams")
    if not isinstance(streams, list):
        streams = []
    video_streams = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]

    video = next(
        (
            stream
            for stream in video_streams
            if stream.get("codec_name") == "h264"
            and _integer(stream.get("width")) == 1280
            and _integer(stream.get("height")) == 720
        ),
        None,
    )
    if video is None and video_streams:
        video = next(
            (stream for stream in video_streams if stream.get("codec_name") == "h264"),
            video_streams[0],
        )
    video_codec = (
        str(video.get("codec_name")) if video and video.get("codec_name") else None
    )
    width = _integer(video.get("width")) if video else None
    height = _integer(video.get("height")) if video else None
    if video is None:
        issues.append("video_stream: ffprobe did not report a video stream.")
    else:
        if video_codec != "h264":
            issues.append(
                f"video_codec: video codec must be h264; found {video_codec or 'unknown'}."
            )
        if (width, height) != (1280, 720):
            dimensions = (
                f"{width}x{height}"
                if width is not None and height is not None
                else "unknown"
            )
            issues.append(
                f"video_dimensions: video must be 1280x720; found {dimensions}."
            )

    audio = next(
        (stream for stream in audio_streams if stream.get("codec_name") == "aac"),
        audio_streams[0] if audio_streams else None,
    )
    audio_codec = (
        str(audio.get("codec_name")) if audio and audio.get("codec_name") else None
    )
    if audio is None:
        issues.append("audio_stream: ffprobe did not report an audio stream.")
    elif audio_codec != "aac":
        issues.append(
            f"audio_codec: audio codec must be aac; found {audio_codec or 'unknown'}."
        )

    format_data = probe.get("format")
    duration = (
        _positive_finite_float(format_data.get("duration"))
        if isinstance(format_data, dict)
        else None
    )
    if duration is None:
        issues.append(
            "duration_missing: ffprobe did not report a positive finite container duration."
        )
    else:
        if duration < MIN_VIDEO_DURATION_SECONDS:
            issues.append(
                f"duration_minimum: video duration must be at least "
                f"{MIN_VIDEO_DURATION_SECONDS:g} seconds; found {duration:.3f}."
            )
        delta = abs(duration - expected_duration)
        if delta > MAX_DURATION_DELTA_SECONDS:
            issues.append(
                f"duration_delta: video duration must be within "
                f"{MAX_DURATION_DELTA_SECONDS:g} seconds of expected "
                f"{expected_duration:.3f}; found {duration:.3f} (delta {delta:.3f})."
            )

    return _media_result(
        issues,
        file_size_bytes=file_size,
        duration_seconds=duration,
        video_codec=video_codec,
        width=width,
        height=height,
        audio_codec=audio_codec,
    )


__all__ = [
    "MediaVerificationResult",
    "validate_storyboard",
    "verify_rendered_video",
]
