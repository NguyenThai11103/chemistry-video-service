"""Resilient, per-scene speech synthesis and audio inspection.

The public synthesizer contract deliberately returns the path it actually
created.  This matters when the online MP3 provider fails over to an offline
WAV provider: callers must mux the returned file rather than assume the suffix
of the path they requested.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Protocol, runtime_checkable


DEFAULT_EDGE_VOICE = "en-US-AriaNeural"
EDGE_TIMEOUT_SECONDS = 12.0
OFFLINE_TIMEOUT_SECONDS = 12.0
FFPROBE_TIMEOUT_SECONDS = 10.0
SCENE_PADDING_SECONDS = 0.4

_MIN_AUDIO_BYTES = 128
_WAV_SAMPLE_RATE = 22_050
_MIN_FALLBACK_DURATION_SECONDS = 0.8
_MAX_FALLBACK_DURATION_SECONDS = 20.0


class SpeechSynthesisError(RuntimeError):
    """Raised when a speech provider cannot produce a usable audio file."""


class AudioProbeError(RuntimeError):
    """Raised when ffprobe cannot determine a positive audio duration."""


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Provider-independent, asynchronous per-scene speech interface."""

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        """Synthesize one narration and return the path actually produced."""


def _validated_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Narration text must be a non-empty string")
    return text.strip()


def _destination(output_path: str | Path, suffix: str) -> Path:
    path = Path(output_path)
    if not path.name or path.name in {".", ".."}:
        raise ValueError("Audio output path must name a file")
    return path.with_suffix(suffix)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    # Providers are more consistent when creating, rather than overwriting,
    # their output file.  mkstemp still gives us a collision-resistant name.
    Path(name).unlink(missing_ok=True)
    return Path(name)


def _promote_if_usable(temporary: Path, destination: Path) -> Path:
    try:
        if not temporary.is_file() or temporary.stat().st_size < _MIN_AUDIO_BYTES:
            raise SpeechSynthesisError("Speech provider produced no usable audio")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


class EdgeTTSSynthesizer:
    """Natural online speech using edge-tts, bounded per scene."""

    def __init__(
        self,
        voice: str = DEFAULT_EDGE_VOICE,
        *,
        rate: str = "+0%",
        timeout_seconds: float = EDGE_TIMEOUT_SECONDS,
    ) -> None:
        if not voice:
            raise ValueError("voice must be non-empty")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be positive and finite")
        self.voice = voice
        self.rate = rate
        self.timeout_seconds = float(timeout_seconds)

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        narration = _validated_text(text)
        destination = _destination(output_path, ".mp3")
        temporary = _temporary_path(destination)
        try:
            try:
                import edge_tts
            except ImportError as exc:
                raise SpeechSynthesisError("edge-tts is unavailable") from exc

            communicator = edge_tts.Communicate(
                text=narration,
                voice=self.voice,
                rate=self.rate,
            )
            await asyncio.wait_for(
                communicator.save(str(temporary)), timeout=self.timeout_seconds
            )
            return _promote_if_usable(temporary, destination)
        except TimeoutError as exc:
            temporary.unlink(missing_ok=True)
            raise SpeechSynthesisError("Online speech synthesis timed out") from exc
        except asyncio.CancelledError:
            temporary.unlink(missing_ok=True)
            raise
        except SpeechSynthesisError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            # Provider exception messages can contain request details.  Keep
            # this boundary deliberately generic so narration never leaks.
            raise SpeechSynthesisError("Online speech synthesis failed") from exc


class WindowsSAPISynthesizer:
    """Offline Windows speech through the built-in System.Speech assembly."""

    _SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$output = $env:CHEMISTRY_TTS_OUTPUT
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SetOutputToWaveFile($output)
    $narration = [Console]::In.ReadToEnd()
    $speaker.Speak($narration)
} finally {
    $speaker.Dispose()
}
"""

    def __init__(self, *, timeout_seconds: float = OFFLINE_TIMEOUT_SECONDS) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be positive and finite")
        self.timeout_seconds = float(timeout_seconds)

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        narration = _validated_text(text)
        destination = _destination(output_path, ".wav")
        temporary = _temporary_path(destination)
        try:
            await asyncio.to_thread(self._synthesize_sync, narration, temporary)
            return _promote_if_usable(temporary, destination)
        except asyncio.CancelledError:
            temporary.unlink(missing_ok=True)
            raise
        except SpeechSynthesisError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise SpeechSynthesisError(
                "Windows offline speech synthesis failed"
            ) from exc

    def _synthesize_sync(self, narration: str, temporary: Path) -> None:
        if sys.platform != "win32":
            raise SpeechSynthesisError("Windows offline speech is unavailable")
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if executable is None:
            raise SpeechSynthesisError("Windows offline speech is unavailable")

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        environment = os.environ.copy()
        environment["CHEMISTRY_TTS_OUTPUT"] = str(temporary)
        try:
            result = subprocess.run(
                [
                    executable,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    self._SCRIPT,
                ],
                input=narration,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
                creationflags=creation_flags,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise SpeechSynthesisError(
                "Windows offline speech synthesis timed out"
            ) from exc
        if result.returncode != 0:
            raise SpeechSynthesisError("Windows offline speech synthesis failed")


class DeterministicWavSynthesizer:
    """Dependency-free, deterministic, non-silent last-resort narration.

    The waveform is intentionally voice-like rather than a single test tone:
    short voiced units use harmonics, pitch drift, amplitude envelopes, and
    brief gaps.  It is not intended to replace intelligible TTS, but it keeps
    CI and degraded deployments capable of producing valid timed media.
    """

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        narration = _validated_text(text)
        destination = _destination(output_path, ".wav")
        temporary = _temporary_path(destination)
        try:
            await asyncio.to_thread(self._write_wav, narration, temporary)
            return _promote_if_usable(temporary, destination)
        except asyncio.CancelledError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if isinstance(exc, SpeechSynthesisError):
                raise
            raise SpeechSynthesisError("Fallback speech synthesis failed") from exc

    @staticmethod
    def estimated_duration(text: str) -> float:
        """Return the bounded duration used by the deterministic renderer."""

        narration = _validated_text(text)
        words = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", narration)
        punctuation = len(re.findall(r"[,.!?;:]", narration))
        estimate = 0.38 * max(1, len(words)) + 0.06 * punctuation + 0.35
        return min(
            _MAX_FALLBACK_DURATION_SECONDS,
            max(_MIN_FALLBACK_DURATION_SECONDS, estimate),
        )

    @classmethod
    def _write_wav(cls, narration: str, temporary: Path) -> None:
        duration = cls.estimated_duration(narration)
        sample_count = max(1, round(duration * _WAV_SAMPLE_RATE))
        digest = hashlib.sha256(narration.encode("utf-8")).digest()
        unit_seconds = 0.105
        unit_samples = max(1, round(unit_seconds * _WAV_SAMPLE_RATE))
        gap_samples = max(1, round(0.018 * _WAV_SAMPLE_RATE))

        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(_WAV_SAMPLE_RATE)

            chunk: list[bytes] = []
            for index in range(sample_count):
                within_unit = index % unit_samples
                unit = index // unit_samples
                if within_unit >= unit_samples - gap_samples:
                    value = 0
                else:
                    phase_position = within_unit / (unit_samples - gap_samples)
                    envelope = min(1.0, phase_position / 0.10) * min(
                        1.0, (1.0 - phase_position) / 0.16
                    )
                    seed = digest[unit % len(digest)]
                    base_frequency = 105.0 + float(seed % 75)
                    drift = 1.0 + 0.035 * math.sin(2.0 * math.pi * phase_position)
                    phase = (
                        2.0
                        * math.pi
                        * base_frequency
                        * drift
                        * (index / _WAV_SAMPLE_RATE)
                    )
                    # A changing harmonic mix loosely resembles voiced formants.
                    signal = (
                        math.sin(phase)
                        + 0.42 * math.sin(2.0 * phase + 0.3)
                        + 0.20 * math.sin(3.0 * phase + seed / 255.0)
                    )
                    value = int(8_500 * envelope * signal / 1.62)
                    value = max(-32_768, min(32_767, value))
                chunk.append(struct.pack("<h", value))
                if len(chunk) >= 4_096:
                    output.writeframesraw(b"".join(chunk))
                    chunk.clear()
            if chunk:
                output.writeframesraw(b"".join(chunk))


class OfflineSpeechSynthesizer:
    """Use Windows SAPI when possible, then the portable WAV generator."""

    def __init__(
        self,
        sapi: SpeechSynthesizer | None = None,
        portable: SpeechSynthesizer | None = None,
    ) -> None:
        self.sapi = sapi if sapi is not None else WindowsSAPISynthesizer()
        self.portable = (
            portable if portable is not None else DeterministicWavSynthesizer()
        )

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        _validated_text(text)
        if sys.platform == "win32":
            try:
                return await self.sapi.synthesize(text, output_path)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        return await self.portable.synthesize(text, output_path)


class ResilientSpeechSynthesizer:
    """Try natural online speech first and always retain an offline route."""

    def __init__(
        self,
        primary: SpeechSynthesizer | None = None,
        offline: SpeechSynthesizer | None = None,
    ) -> None:
        self.primary = primary if primary is not None else EdgeTTSSynthesizer()
        self.offline = offline if offline is not None else OfflineSpeechSynthesizer()

    async def synthesize(self, text: str, output_path: str | Path) -> Path:
        _validated_text(text)
        try:
            return await self.primary.synthesize(text, output_path)
        except asyncio.CancelledError:
            raise
        except Exception:
            return await self.offline.synthesize(text, output_path)


def probe_audio_duration(path: str | Path) -> float:
    """Return the actual media duration reported by ffprobe, without padding."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioProbeError("Audio file does not exist")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise AudioProbeError("ffprobe is unavailable") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProbeError("ffprobe timed out") from exc
    except OSError as exc:
        raise AudioProbeError("ffprobe could not be started") from exc

    if result.returncode != 0:
        raise AudioProbeError("ffprobe could not inspect the audio file")
    try:
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioProbeError("ffprobe returned no valid audio duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise AudioProbeError("ffprobe returned no valid audio duration")
    return duration


def padded_scene_duration(
    path: str | Path, padding_seconds: float = SCENE_PADDING_SECONDS
) -> float:
    """Return measured audio duration plus the renderer's scene-tail padding."""

    if padding_seconds < 0 or not math.isfinite(padding_seconds):
        raise ValueError("padding_seconds must be non-negative and finite")
    return probe_audio_duration(path) + padding_seconds


# Explicit, discoverable names for callers that prefer provider terminology.
PurePythonSpeechSynthesizer = DeterministicWavSynthesizer
PurePythonWavSynthesizer = DeterministicWavSynthesizer
FailoverSpeechSynthesizer = ResilientSpeechSynthesizer


__all__ = [
    "AudioProbeError",
    "DEFAULT_EDGE_VOICE",
    "DeterministicWavSynthesizer",
    "EDGE_TIMEOUT_SECONDS",
    "EdgeTTSSynthesizer",
    "FailoverSpeechSynthesizer",
    "OfflineSpeechSynthesizer",
    "PurePythonSpeechSynthesizer",
    "PurePythonWavSynthesizer",
    "ResilientSpeechSynthesizer",
    "SCENE_PADDING_SECONDS",
    "SpeechSynthesisError",
    "SpeechSynthesizer",
    "WindowsSAPISynthesizer",
    "padded_scene_duration",
    "probe_audio_duration",
]
