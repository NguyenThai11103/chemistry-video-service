"""Fast, deterministic checks for the offline media pipeline."""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import pytest
from PIL import ImageChops

from app.models import SceneSpec, Storyboard, VisualTemplate
from app.renderer import HEIGHT, WIDTH, Renderer, RendererError
from app.tts import (
    DeterministicWavSynthesizer,
    ResilientSpeechSynthesizer,
    probe_audio_duration,
)


def test_pure_python_fallback_wav_is_nonempty_and_probeable(
    tmp_path: Path,
) -> None:
    output = asyncio.run(
        DeterministicWavSynthesizer().synthesize(
            "A covalent bond shares electron density.",
            tmp_path / "narration.mp3",
        )
    )

    assert output == tmp_path / "narration.wav"
    assert output.stat().st_size > 128
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() > 0
        assert wav_file.getnframes() > 0

    duration = probe_audio_duration(output)
    assert 0.5 < duration < 20.1


class _FailingPrimary:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        self.calls += 1
        raise self.error


@pytest.mark.parametrize("primary_error", [RuntimeError("offline"), TimeoutError()])
def test_resilient_synthesizer_falls_back_when_primary_fails(
    tmp_path: Path,
    primary_error: Exception,
) -> None:
    primary = _FailingPrimary(primary_error)
    synthesizer = ResilientSpeechSynthesizer(
        primary=primary,
        offline=DeterministicWavSynthesizer(),
    )

    output = asyncio.run(
        asyncio.wait_for(
            synthesizer.synthesize(
                "Neutral water is pH seven.", tmp_path / "scene.mp3"
            ),
            timeout=3,
        )
    )

    assert primary.calls == 1
    assert output.suffix == ".wav"
    assert output.is_file()
    assert probe_audio_duration(output) > 0


def _storyboard_covering_all_templates() -> Storyboard:
    common = {
        "narration": "This deterministic sentence supplies enough narration for a preview frame.",
    }
    return Storyboard(
        topic_id="media_test",
        title="Deterministic chemistry renderer preview",
        learner_query="Show every renderer template in a deterministic preview.",
        scenes=[
            SceneSpec(
                scene_index=1,
                heading="pH scale spectrum",
                caption="Acidic below seven, neutral at seven, and basic above seven.",
                visual_template=VisualTemplate.PH_SCALE_SPECTRUM,
                visual_params={"ph": 2, "marker_label": "Acidic: pH 2"},
                **common,
            ),
            SceneSpec(
                scene_index=2,
                heading="Covalent electron sharing",
                caption="A shared electron pair forms between the two atoms.",
                visual_template=VisualTemplate.COVALENT_ELECTRON_SHARING,
                visual_params={
                    "state": "shared_h2_pair",
                    "atoms": ["H", "H"],
                    "shared_electrons": 2,
                },
                **common,
            ),
            SceneSpec(
                scene_index=3,
                heading="Ionic and covalent comparison",
                caption="Electron transfer and electron sharing are shown side by side.",
                visual_template=VisualTemplate.IONIC_VS_COVALENT_SPLIT,
                visual_params={"state": "na_to_cl_transfer"},
                **common,
            ),
            SceneSpec(
                scene_index=4,
                heading="Concept summary",
                caption="The final cards keep each takeaway within the safe layout area.",
                visual_template=VisualTemplate.CONCEPT_SUMMARY_CARD,
                visual_params={
                    "takeaways": [
                        "logarithmic scale",
                        "shared electron density",
                        "opposite ions attract",
                        "structure affects properties",
                    ]
                },
                **common,
            ),
        ],
    )


def test_renderer_preview_covers_every_template_without_layout_errors() -> None:
    storyboard = _storyboard_covering_all_templates()
    renderer = Renderer()

    assert {scene.visual_template for scene in storyboard.scenes} == set(VisualTemplate)
    for scene_position in range(len(storyboard.scenes)):
        first = renderer.render_frame(storyboard, scene_position, 0.0)
        middle = renderer.render_frame(storyboard, scene_position, 0.5)
        last = renderer.render_frame(storyboard, scene_position, 1.0)

        for frame in (first, middle, last):
            assert frame.size == (WIDTH, HEIGHT)
            assert frame.mode == "RGB"
            assert frame.getbbox() == (0, 0, WIDTH, HEIGHT)
        assert ImageChops.difference(first, last).getbbox() is not None


def test_renderer_wraps_invalid_visual_parameters() -> None:
    storyboard = _storyboard_covering_all_templates()
    storyboard.scenes[0].visual_params["ph"] = "not-a-number"

    with pytest.raises(RendererError) as caught:
        Renderer().render_frame(storyboard, 0, 0.5)

    assert caught.value.code == "invalid_visual_params"
