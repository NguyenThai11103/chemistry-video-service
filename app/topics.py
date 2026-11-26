"""Extensible topic definitions and deterministic question matching.

The registry is intentionally independent of the orchestration layer.  A new
subject can register its own :class:`TopicDefinition` instances without
changing job processing code; for example, ``registry.register(physics_topic)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Iterable, Iterator

from app.models import VisualTemplate


_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question(value: str) -> str:
    """Return a stable representation used for all topic matching.

    NFKC normalization handles visually equivalent Unicode input, while
    punctuation-to-space conversion makes ``ionic-vs-covalent`` equivalent to
    ``ionic vs covalent``.  Matching is phrase based and never relies on the
    process-randomized value of Python hashes.
    """

    if not isinstance(value, str):
        raise TypeError("question must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _NON_WORD_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_question(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _normalize_values(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_question(value)
        if not item:
            raise ValueError(f"{field_name} cannot contain an empty value")
        if item not in seen:
            normalized.append(item)
            seen.add(item)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ConceptSynonymGroup:
    """Alternate language that satisfies one required scientific concept."""

    concept_id: str
    synonyms: tuple[str, ...]

    def __post_init__(self) -> None:
        concept_id = normalize_question(self.concept_id).replace(" ", "_")
        if not concept_id:
            raise ValueError("concept_id cannot be empty")
        object.__setattr__(self, "concept_id", concept_id)
        object.__setattr__(
            self,
            "synonyms",
            _normalize_values(self.synonyms, field_name=f"{concept_id}.synonyms"),
        )


@dataclass(frozen=True, slots=True)
class TopicMatcher:
    """Declarative keyword and learner-intent rules for one topic.

    Every keyword group must match at least one phrase.  At least one intent
    phrase must also match when ``require_intent`` is true.  Exclusions are
    checked first.  A higher priority wins before the other deterministic score
    components, making a narrow comparison intent outrank a broad topic.
    """

    keyword_groups: tuple[tuple[str, ...], ...]
    intent_phrases: tuple[str, ...]
    excluded_phrases: tuple[str, ...] = ()
    require_intent: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.keyword_groups:
            raise ValueError("keyword_groups cannot be empty")
        groups = tuple(
            _normalize_values(group, field_name="keyword group")
            for group in self.keyword_groups
        )
        intents = _normalize_values(self.intent_phrases, field_name="intent_phrases")
        exclusions = tuple(
            dict.fromkeys(
                normalize_question(value)
                for value in self.excluded_phrases
                if value.strip()
            )
        )
        object.__setattr__(self, "keyword_groups", groups)
        object.__setattr__(self, "intent_phrases", intents)
        object.__setattr__(self, "excluded_phrases", exclusions)

    def score(self, normalized_question: str) -> tuple[int, int, int, int] | None:
        """Return a sortable deterministic match score, or ``None``."""

        if any(
            _contains_phrase(normalized_question, phrase)
            for phrase in self.excluded_phrases
        ):
            return None

        matched_keywords: list[str] = []
        for group in self.keyword_groups:
            matches = [
                phrase
                for phrase in group
                if _contains_phrase(normalized_question, phrase)
            ]
            if not matches:
                return None
            matched_keywords.extend(matches)

        matched_intents = [
            phrase
            for phrase in self.intent_phrases
            if _contains_phrase(normalized_question, phrase)
        ]
        if self.require_intent and not matched_intents:
            return None

        # Longer matched phrases are more specific.  The tuple contains no
        # unordered data, so equal input and registry contents always agree.
        specificity = sum(len(phrase.split()) for phrase in matched_keywords)
        return self.priority, len(matched_intents), specificity, len(matched_keywords)


@dataclass(frozen=True, slots=True)
class TopicDefinition:
    """Everything a supported topic contributes to the application."""

    subject: str
    topic_id: str
    title: str
    matcher: TopicMatcher
    allowed_visual_templates: frozenset[VisualTemplate]
    required_concept_groups: tuple[ConceptSynonymGroup, ...]
    forbidden_misconceptions: tuple[str, ...]

    def __post_init__(self) -> None:
        subject = normalize_question(self.subject).replace(" ", "_")
        topic_id = normalize_question(self.topic_id).replace(" ", "_")
        if not subject or not topic_id or not self.title.strip():
            raise ValueError("subject, topic_id, and title are required")
        if not self.allowed_visual_templates:
            raise ValueError("allowed_visual_templates cannot be empty")
        if not self.required_concept_groups:
            raise ValueError("required_concept_groups cannot be empty")
        if not self.forbidden_misconceptions:
            raise ValueError("forbidden_misconceptions cannot be empty")
        if len({group.concept_id for group in self.required_concept_groups}) != len(
            self.required_concept_groups
        ):
            raise ValueError("required concept IDs must be unique within a topic")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "topic_id", topic_id)
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(
            self,
            "forbidden_misconceptions",
            _normalize_values(
                self.forbidden_misconceptions,
                field_name="forbidden_misconceptions",
            ),
        )

    @property
    def qualified_id(self) -> str:
        return f"{self.subject}.{self.topic_id}"


@dataclass(slots=True)
class TopicRegistry:
    """Topic catalog indexed first by subject, then by topic ID."""

    _subjects: dict[str, dict[str, TopicDefinition]] = field(default_factory=dict)

    def register(self, topic: TopicDefinition, *, replace: bool = False) -> None:
        subject_topics = self._subjects.setdefault(topic.subject, {})
        if topic.topic_id in subject_topics and not replace:
            raise ValueError(f"topic already registered: {topic.qualified_id}")
        subject_topics[topic.topic_id] = topic

    def get(self, topic_id: str, *, subject: str = "chemistry") -> TopicDefinition:
        subject_key = normalize_question(subject).replace(" ", "_")
        topic_key = normalize_question(topic_id).replace(" ", "_")
        try:
            return self._subjects[subject_key][topic_key]
        except KeyError as exc:
            raise KeyError(f"unknown topic: {subject_key}.{topic_key}") from exc

    def match(
        self, question: str, *, subject: str | None = None
    ) -> TopicDefinition | None:
        normalized = normalize_question(question)
        if not normalized:
            return None

        candidates: Iterator[TopicDefinition]
        if subject is None:
            candidates = (
                topic
                for subject_key in sorted(self._subjects)
                for topic in self.list_topics(subject_key)
            )
        else:
            candidates = iter(self.list_topics(subject))

        matches: list[tuple[tuple[int, int, int, int], str, TopicDefinition]] = []
        for topic in candidates:
            score = topic.matcher.score(normalized)
            if score is not None:
                matches.append((score, topic.qualified_id, topic))
        if not matches:
            return None

        # qualified_id is an explicit, stable final tie breaker. Reverse only
        # the score; lexical ascending wins ties for predictable registration-
        # order-independent behavior.
        best_score = max(match_[0] for match_ in matches)
        return min(
            (match_[2] for match_ in matches if match_[0] == best_score),
            key=lambda topic: topic.qualified_id,
        )

    def require_match(
        self, question: str, *, subject: str | None = None
    ) -> TopicDefinition:
        topic = self.match(question, subject=subject)
        if topic is None:
            scope = subject or "any subject"
            raise LookupError(f"no supported topic matched question in {scope}")
        return topic

    def list_subjects(self) -> tuple[str, ...]:
        return tuple(sorted(self._subjects))

    def list_topics(self, subject: str | None = None) -> tuple[TopicDefinition, ...]:
        if subject is None:
            return tuple(
                self._subjects[subject_key][topic_id]
                for subject_key in sorted(self._subjects)
                for topic_id in sorted(self._subjects[subject_key])
            )
        subject_key = normalize_question(subject).replace(" ", "_")
        topics = self._subjects.get(subject_key, {})
        return tuple(topics[topic_id] for topic_id in sorted(topics))


def _concept(concept_id: str, *synonyms: str) -> ConceptSynonymGroup:
    return ConceptSynonymGroup(concept_id=concept_id, synonyms=tuple(synonyms))


CHEMISTRY_TOPICS: tuple[TopicDefinition, ...] = (
    TopicDefinition(
        subject="chemistry",
        topic_id="ph_scale",
        title="How the pH scale works",
        matcher=TopicMatcher(
            keyword_groups=(("ph scale", "ph"),),
            intent_phrases=("how", "work", "explain", "what is", "mean"),
            priority=20,
        ),
        allowed_visual_templates=frozenset(
            {
                VisualTemplate.PH_SCALE_SPECTRUM,
                VisualTemplate.CONCEPT_SUMMARY_CARD,
            }
        ),
        required_concept_groups=(
            _concept(
                "logarithmic_definition",
                "negative log base 10",
                "-log10",
                "logarithmic scale",
            ),
            _concept(
                "hydrogen_ion_concentration",
                "hydrogen ion concentration",
                "h+ concentration",
                "proton concentration",
            ),
            _concept("tenfold_change", "tenfold change", "factor of ten", "10 times"),
            _concept(
                "neutral_at_25c",
                "neutral water is ph 7 at 25 c",
                "ph 7 is neutral at about 25 c",
            ),
            _concept(
                "acidic_and_basic_direction",
                "below 7 is acidic and above 7 is basic",
                "lower is acidic and higher is basic",
            ),
        ),
        forbidden_misconceptions=(
            "ph is always between 0 and 14",
            "the ph scale only ranges from 0 to 14",
            "each ph unit is a linear change",
            "higher ph means more hydrogen ions",
        ),
    ),
    TopicDefinition(
        subject="chemistry",
        topic_id="covalent_bonds",
        title="Why atoms form covalent bonds",
        matcher=TopicMatcher(
            keyword_groups=(("covalent",), ("bond", "bonds", "bonding")),
            intent_phrases=("why", "form", "share", "how", "what is"),
            excluded_phrases=("ionic", "difference", "compare", "versus", "vs"),
            priority=30,
        ),
        allowed_visual_templates=frozenset(
            {
                VisualTemplate.COVALENT_ELECTRON_SHARING,
                VisualTemplate.CONCEPT_SUMMARY_CARD,
            }
        ),
        required_concept_groups=(
            _concept(
                "shared_electron_density",
                "shared electron density",
                "share electrons",
                "shared electrons",
            ),
            _concept(
                "lower_energy",
                "lower energy arrangement",
                "more stable lower energy",
                "reduces the system energy",
            ),
            _concept(
                "nuclear_attraction",
                "nuclei attract shared electrons",
                "attraction between nuclei and shared electrons",
            ),
            _concept(
                "octet_is_heuristic",
                "octet rule is a heuristic",
                "octet rule has exceptions",
                "octet guideline with exceptions",
            ),
        ),
        forbidden_misconceptions=(
            "atoms bond because they want a full shell",
            "the octet rule is a universal law",
            "covalent electrons are always shared equally",
            "covalent bonds involve no electrostatic attraction",
        ),
    ),
    TopicDefinition(
        subject="chemistry",
        topic_id="ionic_vs_covalent",
        title="Ionic versus covalent bonding",
        matcher=TopicMatcher(
            keyword_groups=(("ionic",), ("covalent",)),
            intent_phrases=(
                "difference",
                "different",
                "compare",
                "comparison",
                "versus",
                "vs",
                "contrast",
            ),
            require_intent=True,
            priority=100,
        ),
        allowed_visual_templates=frozenset(
            {
                VisualTemplate.IONIC_VS_COVALENT_SPLIT,
                VisualTemplate.CONCEPT_SUMMARY_CARD,
            }
        ),
        required_concept_groups=(
            _concept(
                "ionic_particles",
                "positive and negative ions",
                "cations and anions",
                "oppositely charged ions",
            ),
            _concept(
                "electrostatic_attraction",
                "electrostatic attraction",
                "attraction between opposite charges",
            ),
            _concept(
                "ionic_lattice",
                "extended ionic lattice",
                "crystal lattice",
                "ionic lattice",
            ),
            _concept(
                "covalent_sharing",
                "shared electron density",
                "share electrons",
                "shared electron pair",
            ),
            _concept(
                "bonding_continuum",
                "bonding is a continuum",
                "continuum of bond character",
                "partial ionic and covalent character",
            ),
            _concept(
                "properties_are_not_universal",
                "properties depend on structure",
                "properties are trends not universal rules",
                "exceptions to property trends",
            ),
        ),
        forbidden_misconceptions=(
            "all ionic compounds dissolve in water",
            "all covalent substances have low melting points",
            "ionic bonds are complete electron transfer with no covalent character",
            "ionic compounds exist only as individual molecules",
        ),
    ),
)


TOPIC_REGISTRY = TopicRegistry()
for _topic in CHEMISTRY_TOPICS:
    TOPIC_REGISTRY.register(_topic)

# Lower-case alias is convenient for dependency injection and interactive use.
topic_registry = TOPIC_REGISTRY


def match_topic(query: str, subject: str = "chemistry") -> TopicDefinition | None:
    """Match a learner query against the default subject-scoped registry."""

    return TOPIC_REGISTRY.match(query, subject=subject)


__all__ = [
    "CHEMISTRY_TOPICS",
    "TOPIC_REGISTRY",
    "ConceptSynonymGroup",
    "TopicDefinition",
    "TopicMatcher",
    "TopicRegistry",
    "match_topic",
    "normalize_question",
    "topic_registry",
]
