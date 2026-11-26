"""Deterministic Pillow/FFmpeg renderer for validated chemistry storyboards.

Frames are written to FFmpeg as RGB24 bytes.  No frame images or user-derived
command strings are created on disk.  Every scene is encoded in a private
temporary directory and the concatenated movie is atomically promoted only
after all subprocesses have completed successfully.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .models import SceneSpec, Storyboard, VisualTemplate


WIDTH = 1280
HEIGHT = 720
FPS = 20
SCENE_PADDING_SECONDS = 0.4
FINAL_VIDEO_FILENAME = "explanation.mp4"

BACKGROUND = "#0F172A"
PANEL = "#17233B"
PANEL_ALT = "#1D2B46"
TEXT = "#F8FAFC"
MUTED = "#A8B5C8"
GRID = "#334155"
CYAN = "#22D3EE"
BLUE = "#60A5FA"
RED = "#FB7185"
AMBER = "#FBBF24"
GREEN = "#34D399"
PURPLE = "#A78BFA"


class RendererError(RuntimeError):
    """A safe rendering failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# Concise alias for callers that use the module name as context.
RenderError = RendererError


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Metadata for one successfully and atomically rendered explainer."""

    output_path: Path
    duration_seconds: float
    scene_durations: tuple[float, ...]
    scene_count: int
    frame_count: int
    size_bytes: int

    @property
    def path(self) -> Path:
        """Compatibility shorthand for integrations that only need the path."""

        return self.output_path


def _ease(progress: float) -> float:
    value = min(1.0, max(0.0, progress))
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: float, b: float, progress: float) -> float:
    return a + (b - a) * progress


def _hex_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _continuous_gradient(width: int, height: int, colors: Sequence[str]) -> Image.Image:
    """Create a reusable, pixel-continuous horizontal scientific spectrum."""

    image = Image.new("RGB", (width, height))
    pixels = image.load()
    stops = tuple(_hex_rgb(color) for color in colors)
    for x in range(width):
        position = x / max(1, width - 1) * (len(stops) - 1)
        stop_index = min(len(stops) - 2, int(position))
        fraction = position - stop_index
        first, second = stops[stop_index], stops[stop_index + 1]
        color = tuple(round(_lerp(a, b, fraction)) for a, b in zip(first, second))
        for y in range(height):
            pixels[x, y] = color
    return image


def _font_candidates(bold: bool) -> tuple[str, ...]:
    if os.name == "nt":
        windows = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        names = (
            ("segoeuib.ttf", "arialbd.ttf")
            if bold
            else (
                "segoeui.ttf",
                "arial.ttf",
            )
        )
        return tuple(str(windows / name) for name in names)
    return (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )


class _Fonts:
    def __init__(self) -> None:
        self._cache: dict[
            tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont
        ] = {}

    def get(
        self, size: int, *, bold: bool = False
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (size, bold)
        if key not in self._cache:
            for candidate in _font_candidates(bold):
                try:
                    self._cache[key] = ImageFont.truetype(candidate, size)
                    break
                except OSError:
                    continue
            else:
                self._cache[key] = ImageFont.load_default(size=size)
        return self._cache[key]


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: Any) -> float:
    return float(draw.textlength(text, font=font))


def _wrap_lines(
    draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int
) -> list[str]:
    """Pixel-measured wrapping that also breaks pathological long tokens."""

    paragraphs = str(text).replace("\r", "").split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if _text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if _text_width(draw, word, font) <= max_width:
                current = word
                continue
            fragment = ""
            for character in word:
                candidate = fragment + character
                if fragment and _text_width(draw, candidate, font) > max_width:
                    lines.append(fragment)
                    fragment = character
                else:
                    fragment = candidate
            current = fragment
        if current:
            lines.append(current)
    return lines


def _fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int,
    bold: bool,
    fonts: _Fonts,
) -> tuple[Any, list[str], int]:
    for size in range(start_size, min_size - 1, -1):
        font = fonts.get(size, bold=bold)
        lines = _wrap_lines(draw, text, font, max_width)
        spacing = max(4, size // 5)
        total_height = len(lines) * size + max(0, len(lines) - 1) * spacing
        if total_height <= max_height:
            return font, lines, spacing
    font = fonts.get(min_size, bold=bold)
    lines = _wrap_lines(draw, text, font, max_width)
    max_lines = max(1, max_height // (min_size + max(4, min_size // 5)))
    lines = lines[:max_lines]
    if lines and len(" ".join(lines)) < len(text):
        suffix = "..."
        last = lines[-1]
        while last and _text_width(draw, last + suffix, font) > max_width:
            last = last[:-1]
        lines[-1] = last.rstrip() + suffix
    return font, lines, max(4, min_size // 5)


def _centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    fill: str,
    *,
    anchor: str = "mm",
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 15
    left = (
        end[0] - size * math.cos(angle - 0.55),
        end[1] - size * math.sin(angle - 0.55),
    )
    right = (
        end[0] - size * math.cos(angle + 0.55),
        end[1] - size * math.sin(angle + 0.55),
    )
    draw.polygon((end, left, right), fill=color)


def _draw_atom(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    label: str,
    radius: int,
    color: str,
    fonts: _Fonts,
    *,
    charge: str = "",
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline="#E2E8F0",
        width=3,
    )
    _centered_text(draw, (x, y + 1), label, fonts.get(30, bold=True), TEXT)
    if charge:
        _centered_text(
            draw,
            (x + radius * 0.72, y - radius * 0.72),
            charge,
            fonts.get(22, bold=True),
            TEXT,
        )


class Renderer:
    """Render a validated storyboard and one audio file per scene."""

    def __init__(
        self,
        *,
        ffmpeg_path: str | Path = "ffmpeg",
        width: int = WIDTH,
        height: int = HEIGHT,
        fps: int = FPS,
        padding_seconds: float = SCENE_PADDING_SECONDS,
        duration_probe: Callable[[str | Path], float] | None = None,
    ) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError("the chemistry renderer is fixed at 1280x720")
        if fps != FPS:
            raise ValueError("the chemistry renderer is fixed at 20 fps")
        if padding_seconds < 0:
            raise ValueError("padding_seconds cannot be negative")
        self.ffmpeg_path = str(ffmpeg_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.padding_seconds = float(padding_seconds)
        self._duration_probe = duration_probe
        self._fonts = _Fonts()
        self._ph_gradient = _continuous_gradient(
            710,
            67,
            (
                "#E11D48",
                "#F97316",
                "#FACC15",
                "#84CC16",
                "#10B981",
                "#06B6D4",
                "#3B82F6",
                "#7C3AED",
            ),
        )

    def render(
        self,
        storyboard: Storyboard,
        audio_paths: Sequence[str | Path],
        output_path: str | Path,
    ) -> RenderResult:
        """Render scenes, concatenate them, and atomically create explanation.mp4.

        ``output_path`` may be a directory or an explicit ``.mp4`` file.  A
        directory always receives the canonical ``explanation.mp4`` filename.
        """

        if not isinstance(storyboard, Storyboard):
            raise RendererError("invalid_storyboard", "storyboard must be a Storyboard")
        paths = self._validate_audio(storyboard, audio_paths)
        destination = self._resolve_destination(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._require_ffmpeg()

        durations: list[float] = []
        total_frames = 0
        try:
            with tempfile.TemporaryDirectory(
                prefix=".chem-render-", dir=destination.parent
            ) as temporary_name:
                temporary = Path(temporary_name)
                scene_paths: list[Path] = []
                for position, (scene, audio_path) in enumerate(
                    zip(storyboard.scenes, paths, strict=True)
                ):
                    audio_duration = self._probe_duration(audio_path)
                    scene_duration = audio_duration + self.padding_seconds
                    if not math.isfinite(scene_duration) or scene_duration <= 0:
                        raise RendererError(
                            "invalid_audio_duration",
                            f"scene {position + 1} has an invalid audio duration",
                        )
                    frame_count = max(1, math.ceil(scene_duration * self.fps))
                    scene_path = temporary / f"scene-{position + 1:02d}.mp4"
                    self._encode_scene(
                        storyboard,
                        scene,
                        position,
                        audio_path,
                        scene_path,
                        scene_duration,
                        frame_count,
                    )
                    durations.append(scene_duration)
                    total_frames += frame_count
                    scene_paths.append(scene_path)

                staged = temporary / "explanation.staged.mp4"
                self._concatenate(scene_paths, staged)
                if not staged.is_file() or staged.stat().st_size == 0:
                    raise RendererError(
                        "render_failed", "FFmpeg did not produce a final video"
                    )
                os.replace(staged, destination)
        except RendererError:
            raise
        except OSError as exc:
            raise RendererError(
                "render_failed", f"video rendering failed: {exc}"
            ) from exc

        return RenderResult(
            output_path=destination,
            duration_seconds=sum(durations),
            scene_durations=tuple(durations),
            scene_count=len(durations),
            frame_count=total_frames,
            size_bytes=destination.stat().st_size,
        )

    def render_frame(
        self,
        storyboard: Storyboard,
        scene_position: int,
        progress: float,
    ) -> Image.Image:
        """Render one frame; public for deterministic preview and unit tests."""

        if not 0 <= scene_position < len(storyboard.scenes):
            raise IndexError("scene_position is outside the storyboard")
        scene = storyboard.scenes[scene_position]
        progress = min(1.0, max(0.0, float(progress)))
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        self._draw_background(draw)
        self._draw_header(draw, storyboard, scene, scene_position)

        template = scene.visual_template
        try:
            if template == VisualTemplate.PH_SCALE_SPECTRUM:
                self._draw_ph(image, draw, scene, progress)
            elif template == VisualTemplate.COVALENT_ELECTRON_SHARING:
                self._draw_covalent(draw, scene, progress)
            elif template == VisualTemplate.IONIC_VS_COVALENT_SPLIT:
                self._draw_ionic_split(draw, scene, progress)
            elif template == VisualTemplate.CONCEPT_SUMMARY_CARD:
                self._draw_summary(image, scene, progress)
            else:  # Defensive even though VisualTemplate is an enum.
                raise RendererError(
                    "unsupported_visual_template", f"unsupported template: {template}"
                )
        except RendererError:
            raise
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise RendererError(
                "invalid_visual_params",
                f"scene {scene.scene_index} contains invalid visual parameters",
            ) from exc
        self._draw_subtitle(draw, scene.caption)
        return image

    def _validate_audio(
        self, storyboard: Storyboard, audio_paths: Sequence[str | Path]
    ) -> tuple[Path, ...]:
        if isinstance(audio_paths, (str, bytes, Path)):
            raise RendererError(
                "invalid_audio_files", "audio_paths must contain one file per scene"
            )
        try:
            supplied = tuple(Path(path).expanduser().resolve() for path in audio_paths)
        except (TypeError, OSError) as exc:
            raise RendererError(
                "invalid_audio_files", "audio_paths contains an invalid path"
            ) from exc
        if len(supplied) != len(storyboard.scenes):
            raise RendererError(
                "audio_count_mismatch",
                f"expected {len(storyboard.scenes)} audio files, found {len(supplied)}",
            )
        for index, path in enumerate(supplied, start=1):
            if not path.is_file():
                raise RendererError(
                    "audio_file_missing", f"audio file for scene {index} does not exist"
                )
            if path.stat().st_size <= 0:
                raise RendererError(
                    "audio_file_empty", f"audio file for scene {index} is empty"
                )
        return supplied

    @staticmethod
    def _resolve_destination(output_path: str | Path) -> Path:
        supplied = Path(output_path).expanduser()
        if supplied.suffix.casefold() == ".mp4":
            destination = supplied
        else:
            destination = supplied / FINAL_VIDEO_FILENAME
        return destination.resolve()

    def _require_ffmpeg(self) -> None:
        executable = self.ffmpeg_path
        if Path(executable).parent != Path("."):
            exists = Path(executable).is_file()
        else:
            exists = shutil.which(executable) is not None
        if not exists:
            raise RendererError("ffmpeg_unavailable", "FFmpeg is not available")

    def _probe_duration(self, audio_path: Path) -> float:
        probe = self._duration_probe
        if probe is None:
            try:
                from .tts import probe_audio_duration
            except (ImportError, AttributeError) as exc:
                raise RendererError(
                    "audio_probe_unavailable", "audio duration probing is unavailable"
                ) from exc
            probe = probe_audio_duration
        try:
            duration = float(probe(audio_path))
        except RendererError:
            raise
        except Exception as exc:
            raise RendererError(
                "audio_probe_failed", f"could not inspect audio file {audio_path.name}"
            ) from exc
        if not math.isfinite(duration) or duration <= 0:
            raise RendererError(
                "invalid_audio_duration", f"invalid duration for {audio_path.name}"
            )
        return duration

    def _encode_scene(
        self,
        storyboard: Storyboard,
        scene: SceneSpec,
        position: int,
        audio_path: Path,
        output_path: Path,
        duration: float,
        frame_count: int,
    ) -> None:
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-af",
            f"apad=pad_dur={self.padding_seconds:.6f}",
            "-t",
            f"{duration:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(self.fps),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except OSError as exc:
            raise RendererError(
                "ffmpeg_start_failed", "could not start FFmpeg"
            ) from exc

        write_error: BaseException | None = None
        try:
            assert process.stdin is not None
            for frame_index in range(frame_count):
                progress = 1.0 if frame_count == 1 else frame_index / (frame_count - 1)
                frame = self.render_frame(storyboard, position, progress)
                process.stdin.write(frame.tobytes())
        except BaseException as exc:
            write_error = exc
        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        if write_error is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if write_error is not None:
            if isinstance(write_error, RendererError):
                raise write_error
            if isinstance(write_error, (KeyboardInterrupt, SystemExit)):
                raise write_error
            detail = (
                stderr[-500:] if stderr else "frame generation or FFmpeg input failed"
            )
            raise RendererError(
                "scene_encode_failed",
                f"failed while writing scene {position + 1}: {detail}",
            ) from write_error
        if return_code != 0:
            detail = stderr[-500:] if stderr else "no FFmpeg diagnostic was returned"
            raise RendererError(
                "scene_encode_failed",
                f"FFmpeg failed while encoding scene {position + 1}: {detail}",
            )

    def _concatenate(self, scene_paths: Sequence[Path], output_path: Path) -> None:
        if not scene_paths:
            raise RendererError("render_failed", "there are no scenes to concatenate")
        manifest = output_path.parent / "concat.txt"
        lines = []
        for scene_path in scene_paths:
            # FFmpeg concat quoting uses '\'' for a literal single quote.
            normalized = scene_path.resolve().as_posix().replace("'", "'\\''")
            lines.append(f"file '{normalized}'")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
                text=True,
            )
        except OSError as exc:
            raise RendererError(
                "ffmpeg_start_failed", "could not start FFmpeg"
            ) from exc
        if completed.returncode != 0:
            detail = (
                completed.stderr.strip()[-500:] or "no FFmpeg diagnostic was returned"
            )
            raise RendererError(
                "concat_failed", f"FFmpeg could not concatenate scenes: {detail}"
            )

    def _draw_background(self, draw: ImageDraw.ImageDraw) -> None:
        for x in range(0, self.width, 64):
            draw.line((x, 100, x, 580), fill="#152039", width=1)
        for y in range(100, 581, 64):
            draw.line((0, y, self.width, y), fill="#152039", width=1)
        draw.ellipse((1050, 8, 1360, 318), fill="#111F39")
        draw.ellipse((-130, 445, 200, 775), fill="#101E35")

    def _draw_header(
        self,
        draw: ImageDraw.ImageDraw,
        storyboard: Storyboard,
        scene: SceneSpec,
        position: int,
    ) -> None:
        draw.rectangle((0, 0, self.width, 100), fill="#111C31")
        draw.rectangle((0, 97, self.width, 100), fill=CYAN)
        title_font, title_lines, _ = _fit_wrapped_text(
            draw,
            storyboard.title,
            max_width=760,
            max_height=34,
            start_size=30,
            min_size=20,
            bold=True,
            fonts=self._fonts,
        )
        draw.text((48, 22), title_lines[0], font=title_font, fill=TEXT)
        draw.text((48, 62), scene.heading, font=self._fonts.get(20), fill=MUTED)
        total = len(storyboard.scenes)
        badge = f"SCENE {position + 1} / {total}"
        draw.rounded_rectangle((1022, 22, 1228, 69), radius=18, fill="#20304D")
        _centered_text(
            draw,
            (1125, 46),
            badge,
            self._fonts.get(18, bold=True),
            CYAN,
        )
        bar_left, bar_right = 1022, 1228
        draw.rounded_rectangle((bar_left, 78, bar_right, 85), radius=3, fill=GRID)
        draw.rounded_rectangle(
            (
                bar_left,
                78,
                bar_left + (bar_right - bar_left) * (position + 1) / total,
                85,
            ),
            radius=3,
            fill=CYAN,
        )

    def _draw_subtitle(self, draw: ImageDraw.ImageDraw, caption: str) -> None:
        left, top, right, bottom = 70, 590, 1210, 690
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=24,
            fill="#111C31",
            outline="#3B4D69",
            width=2,
        )
        draw.rounded_rectangle(
            (left + 18, top + 18, left + 24, bottom - 18), 3, fill=CYAN
        )
        font, lines, spacing = _fit_wrapped_text(
            draw,
            caption,
            max_width=right - left - 96,
            max_height=bottom - top - 28,
            start_size=30,
            min_size=19,
            bold=True,
            fonts=self._fonts,
        )
        line_height = getattr(font, "size", 20)
        total_height = len(lines) * line_height + max(0, len(lines) - 1) * spacing
        y = top + (bottom - top - total_height) / 2
        for line in lines:
            draw.text((left + 50, y), line, font=font, fill=TEXT)
            y += line_height + spacing

    def _draw_ph(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        scene: SceneSpec,
        progress: float,
    ) -> None:
        params = scene.visual_params
        ph = float(params.get("target_ph", params.get("ph", 7)))
        if not math.isfinite(ph):
            raise ValueError("pH must be finite")
        ph = min(14.0, max(0.0, ph))
        state = str(params.get("state", ""))
        left, right = 90, 800
        top, bottom = 225, 292
        image.paste(self._ph_gradient, (left, top))
        draw.rounded_rectangle(
            (left, top, right, bottom), radius=16, outline=TEXT, width=3
        )
        for value in range(15):
            x = left + value * (right - left) / 14
            draw.line((x, bottom, x, bottom + 10), fill=TEXT, width=2)
            _centered_text(
                draw,
                (x, bottom + 29),
                str(value),
                self._fonts.get(16, bold=True),
                TEXT,
            )
        _centered_text(
            draw, (left, top - 26), "ACIDIC", self._fonts.get(17, bold=True), RED
        )
        _centered_text(
            draw,
            ((left + right) / 2, top - 26),
            "NEUTRAL",
            self._fonts.get(17, bold=True),
            GREEN,
        )
        _centered_text(
            draw, (right, top - 26), "BASIC", self._fonts.get(17, bold=True), BLUE
        )

        travel = _ease(min(1.0, progress / 0.55))
        start_ph = 7.0 if ph != 7 else 3.5
        marker_ph = _lerp(start_ph, ph, travel)
        marker_x = left + marker_ph * (right - left) / 14
        pulse = 0.5 + 0.5 * math.sin(progress * math.tau * 2)
        glow = 23 + int(7 * pulse)
        draw.ellipse(
            (marker_x - glow, top - glow - 12, marker_x + glow, top + glow - 12),
            outline="#FFFFFF",
            width=4,
        )
        draw.polygon(
            ((marker_x, top - 2), (marker_x - 12, top - 24), (marker_x + 12, top - 24)),
            fill=TEXT,
        )
        label = str(params.get("marker_label", f"pH {ph:g}"))
        label_font, label_lines, _ = _fit_wrapped_text(
            draw,
            label,
            max_width=246,
            max_height=23,
            start_size=19,
            min_size=14,
            bold=True,
            fonts=self._fonts,
        )
        label_text = label_lines[0]
        label_center = min(right - 130, max(left + 130, marker_x))
        draw.rounded_rectangle(
            (label_center - 128, top - 87, label_center + 128, top - 52),
            13,
            fill="#1D2B46",
            outline="#475569",
            width=2,
        )
        _centered_text(
            draw,
            (label_center, top - 69),
            label_text,
            label_font,
            TEXT,
        )

        # A beaker makes the H+/OH- balance visible without relying on prose.
        bx, by = 925, 205
        draw.line((bx, by, bx + 200, by), fill="#DCEAF5", width=6)
        draw.line((bx + 18, by, bx + 34, 455), fill="#DCEAF5", width=6)
        draw.line((bx + 182, by, bx + 166, 455), fill="#DCEAF5", width=6)
        draw.arc((bx + 34, 420, bx + 166, 481), 0, 180, fill="#DCEAF5", width=6)
        liquid_color = "#7DD3FC" if ph >= 7 else "#FDA4AF"
        draw.rounded_rectangle((bx + 38, 302, bx + 162, 448), 16, fill=liquid_color)
        acid_strength = max(0.0, (7 - ph) / 7)
        base_strength = max(0.0, (ph - 7) / 7)
        h_count = 3 + round(8 * acid_strength)
        oh_count = 3 + round(8 * base_strength)
        for index in range(max(h_count, oh_count)):
            # Integer congruences keep positions stable across frames; only a
            # small sinusoidal drift is animated.
            px = bx + 50 + ((index * 47 + 13) % 100)
            py = 326 + ((index * 71 + 19) % 100)
            drift = 4 * math.sin(progress * math.tau + index * 0.8)
            if index < h_count:
                draw.ellipse(
                    (px - 14, py + drift - 14, px + 14, py + drift + 14), fill=RED
                )
                _centered_text(
                    draw, (px, py + drift), "H+", self._fonts.get(13, bold=True), TEXT
                )
            if index < oh_count:
                ox = bx + 150 - ((index * 43 + 11) % 100)
                oy = 330 + ((index * 59 + 29) % 96)
                draw.ellipse(
                    (ox - 15, oy - drift - 15, ox + 15, oy - drift + 15), fill=CYAN
                )
                _centered_text(
                    draw,
                    (ox, oy - drift),
                    "OH-",
                    self._fonts.get(12, bold=True),
                    BACKGROUND,
                )
        _centered_text(
            draw,
            (bx + 100, 505),
            f"H+ {h_count}  :  {oh_count} OH-",
            self._fonts.get(18, bold=True),
            TEXT,
        )

        if (
            "tenfold" in state
            or "comparison_ph" in params
            or bool(params.get("highlight_tenfold"))
        ):
            alpha_progress = _ease(max(0.0, (progress - 0.3) / 0.45))
            box_right = left + 350 * alpha_progress
            draw.rounded_rectangle(
                (left, 380, box_right, 502), 20, fill="#21314D", outline=AMBER, width=3
            )
            if alpha_progress > 0.55:
                ratio = params.get("concentration_ratio", 10)
                _centered_text(
                    draw,
                    (left + 175, 416),
                    "LOG SCALE",
                    self._fonts.get(17, bold=True),
                    AMBER,
                )
                _centered_text(
                    draw,
                    (left + 175, 462),
                    f"1 pH step = {ratio}x [H+]",
                    self._fonts.get(25, bold=True),
                    TEXT,
                )

    def _draw_energy_curve(
        self, draw: ImageDraw.ImageDraw, progress: float, *, highlight: bool
    ) -> None:
        left, top, right, bottom = 800, 170, 1180, 485
        draw.rounded_rectangle(
            (left, top, right, bottom), 22, fill=PANEL, outline=GRID, width=2
        )
        draw.line((left + 48, top + 35, left + 48, bottom - 42), fill=MUTED, width=2)
        draw.line(
            (left + 48, bottom - 42, right - 28, bottom - 42), fill=MUTED, width=2
        )
        draw.text(
            (left + 14, top + 15),
            "Energy",
            font=self._fonts.get(15, bold=True),
            fill=MUTED,
        )
        draw.text(
            (right - 132, bottom - 34),
            "distance",
            font=self._fonts.get(15, bold=True),
            fill=MUTED,
        )
        points: list[tuple[float, float]] = []
        minimum = (0.0, 0.0)
        for index in range(101):
            x_norm = 0.12 + index / 100 * 2.9
            energy = 0.30 / (x_norm**5) - 1.2 / (x_norm**2)
            energy = min(1.1, max(-0.82, energy))
            px = left + 55 + index / 100 * (right - left - 92)
            py = top + 80 + (0.95 - energy) / 1.9 * (bottom - top - 145)
            points.append((px, py))
            if energy < minimum[1]:
                minimum = (px, py)
        visible = max(2, round(len(points) * _ease(progress)))
        draw.line(points[:visible], fill=CYAN, width=5, joint="curve")
        if highlight and progress > 0.48:
            mx, my = minimum
            draw.ellipse((mx - 9, my - 9, mx + 9, my + 9), fill=AMBER)
            draw.line((mx, my + 12, mx, bottom - 42), fill=AMBER, width=2)
            _centered_text(
                draw,
                (mx + 45, my + 27),
                "minimum",
                self._fonts.get(16, bold=True),
                AMBER,
            )

    def _draw_covalent(
        self, draw: ImageDraw.ImageDraw, scene: SceneSpec, progress: float
    ) -> None:
        params = scene.visual_params
        state = str(params.get("state", ""))
        molecule = str(params.get("molecule", ""))
        is_water = molecule.casefold() == "h2o" or state.casefold() == "h2o"
        diagram_left, diagram_right = 80, 760
        draw.rounded_rectangle(
            (diagram_left, 145, diagram_right, 530),
            24,
            fill=PANEL,
            outline=GRID,
            width=2,
        )
        if is_water:
            center = (420.0, 330.0)
            spread = _ease(min(1.0, progress / 0.65))
            left_h = (_lerp(420, 285, spread), _lerp(330, 420, spread))
            right_h = (_lerp(420, 555, spread), _lerp(330, 420, spread))
            for outer in (left_h, right_h):
                draw.line((center, outer), fill="#94A3B8", width=13)
                dx, dy = outer[0] - center[0], outer[1] - center[1]
                length = max(1.0, math.hypot(dx, dy))
                nx, ny = dx / length, dy / length
                for offset in (-10, 10):
                    ex = center[0] + nx * 78 - ny * offset
                    ey = center[1] + ny * 78 + nx * offset
                    draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=AMBER)
            _draw_atom(draw, center, "O", 62, RED, self._fonts)
            _draw_atom(draw, left_h, "H", 43, BLUE, self._fonts)
            _draw_atom(draw, right_h, "H", 43, BLUE, self._fonts)
            _centered_text(
                draw, (420, 185), "H2O  •  104.5°", self._fonts.get(24, bold=True), TEXT
            )
            _centered_text(
                draw,
                (420, 493),
                "two shared electron pairs",
                self._fonts.get(20, bold=True),
                CYAN,
            )
            self._draw_energy_curve(draw, progress, highlight=True)
            return

        atoms = params.get("atoms", ["H", "H"])
        labels = (
            [str(item) for item in atoms] if isinstance(atoms, list) else ["H", "H"]
        )
        labels = (labels + ["H", "H"])[:2]
        bonded = int(params.get("shared_electrons", 0)) >= 2 or "shared" in state
        movement = _ease(min(1.0, progress / 0.68))
        final_gap = 65 if bonded else 150
        half_gap = _lerp(250, final_gap, movement)
        centers = ((420 - half_gap, 325.0), (420 + half_gap, 325.0))
        shell_radius = 76
        shared_alpha = _ease(max(0.0, (progress - 0.32) / 0.42)) if bonded else 0.0
        if bonded:
            cloud_width = 52 + 40 * shared_alpha
            draw.ellipse(
                (420 - cloud_width, 267, 420 + cloud_width, 383),
                fill="#145D73",
                outline=CYAN,
                width=3,
            )
        for center, label in zip(centers, labels, strict=True):
            draw.ellipse(
                (
                    center[0] - shell_radius,
                    center[1] - shell_radius,
                    center[0] + shell_radius,
                    center[1] + shell_radius,
                ),
                outline=CYAN,
                width=3,
            )
            _draw_atom(
                draw, center, label, 39 if len(label) == 1 else 47, BLUE, self._fonts
            )
        electron_phase = progress * math.tau
        if bonded:
            for index, offset in enumerate((-17, 17)):
                ey = 325 + 8 * math.sin(electron_phase + index * math.pi)
                draw.ellipse(
                    (420 + offset - 9, ey - 9, 420 + offset + 9, ey + 9), fill=AMBER
                )
            _centered_text(
                draw,
                (420, 464),
                "shared electron pair",
                self._fonts.get(21, bold=True),
                CYAN,
            )
        else:
            for index, center in enumerate(centers):
                angle = electron_phase + index * math.pi
                ex = center[0] + math.cos(angle) * shell_radius
                ey = center[1] + math.sin(angle) * shell_radius
                draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=AMBER)
            _draw_arrow(draw, (330, 465), (510, 465), MUTED, width=3)
            _centered_text(
                draw,
                (420, 495),
                "atoms approach",
                self._fonts.get(19, bold=True),
                MUTED,
            )
        self._draw_energy_curve(draw, progress, highlight=bonded)

    def _draw_ionic_split(
        self, draw: ImageDraw.ImageDraw, scene: SceneSpec, progress: float
    ) -> None:
        params = scene.visual_params
        state = str(params.get("state", ""))
        draw.rounded_rectangle(
            (55, 135, 625, 525), 24, fill=PANEL, outline=RED, width=3
        )
        draw.rounded_rectangle(
            (655, 135, 1225, 525), 24, fill=PANEL, outline=CYAN, width=3
        )
        draw.rounded_rectangle((78, 153, 210, 193), 16, fill="#4A2334")
        draw.rounded_rectangle((678, 153, 840, 193), 16, fill="#164A59")
        _centered_text(draw, (144, 173), "IONIC", self._fonts.get(18, bold=True), RED)
        _centered_text(
            draw, (759, 173), "COVALENT", self._fonts.get(18, bold=True), CYAN
        )

        if state == "ionic_lattice":
            reveal = _ease(progress)
            for row in range(3):
                for column in range(4):
                    order = row * 4 + column
                    if reveal * 13 < order:
                        continue
                    x = 175 + column * 100
                    y = 260 + row * 90
                    sodium = (row + column) % 2 == 0
                    _draw_atom(
                        draw,
                        (x, y),
                        "Na" if sodium else "Cl",
                        30,
                        BLUE if sodium else GREEN,
                        self._fonts,
                        charge="+" if sodium else "-",
                    )
        else:
            transfer = _ease(min(1.0, progress / 0.72))
            na_x = _lerp(205, 235, transfer)
            cl_x = _lerp(480, 445, transfer)
            charged = transfer > 0.72
            _draw_atom(
                draw,
                (na_x, 330),
                "Na",
                57,
                BLUE,
                self._fonts,
                charge="+" if charged else "",
            )
            _draw_atom(
                draw,
                (cl_x, 330),
                "Cl",
                62,
                GREEN,
                self._fonts,
                charge="-" if charged else "",
            )
            arc_box = (na_x + 20, 205, cl_x - 15, 347)
            draw.arc(arc_box, 190, 350, fill=AMBER, width=5)
            angle = math.pi * (1.0 + transfer)
            ex = (na_x + cl_x) / 2 + (cl_x - na_x - 35) / 2 * math.cos(angle)
            ey = 276 + 70 * math.sin(angle)
            draw.ellipse((ex - 10, ey - 10, ex + 10, ey + 10), fill=AMBER)
            _centered_text(
                draw,
                (342, 440),
                "electron transfer",
                self._fonts.get(20, bold=True),
                AMBER,
            )
            if charged:
                _draw_arrow(draw, (300, 470), (385, 470), RED, width=4)
                _draw_arrow(draw, (385, 470), (300, 470), RED, width=4)

        # The comparison side remains visible in every split scene so the
        # distinction is continuously reinforced.
        shift = _ease(progress)
        left_cl = _lerp(820, 895, shift)
        right_cl = _lerp(1090, 1005, shift)
        _draw_atom(draw, (left_cl, 330), "Cl", 58, GREEN, self._fonts)
        _draw_atom(draw, (right_cl, 330), "Cl", 58, GREEN, self._fonts)
        draw.ellipse(
            (930 - 45, 275, 970 + 45, 385), fill="#145D73", outline=CYAN, width=3
        )
        for offset in (-14, 14):
            draw.ellipse((950 + offset - 9, 321, 950 + offset + 9, 339), fill=AMBER)
        _centered_text(
            draw, (950, 440), "shared pair", self._fonts.get(20, bold=True), CYAN
        )

        badge_y = 485
        draw.rounded_rectangle((115, badge_y, 565, badge_y + 31), 13, fill="#402334")
        draw.rounded_rectangle((715, badge_y, 1165, badge_y + 31), 13, fill="#164453")
        _centered_text(
            draw,
            (340, badge_y + 16),
            "opposite ions attract",
            self._fonts.get(16, bold=True),
            TEXT,
        )
        _centered_text(
            draw,
            (940, badge_y + 16),
            "electron density is shared",
            self._fonts.get(16, bold=True),
            TEXT,
        )

    def _draw_summary(
        self, image: Image.Image, scene: SceneSpec, progress: float
    ) -> None:
        params = scene.visual_params
        raw = params.get("takeaways")
        if isinstance(raw, list) and raw:
            items = [str(item) for item in raw[:4]]
        elif str(params.get("state", "")) == "comparison_summary":
            items = [
                str(params.get("ionic_emphasis", "ions attract")),
                str(params.get("covalent_emphasis", "electrons are shared")),
                "bond character is a continuum",
                "properties depend on structure",
            ]
        else:
            marker = str(params.get("marker_label", "Core idea"))
            items = [marker, "measure the trend", "connect structure to behavior"]
        state = str(params.get("state", ""))
        if state == "comparison_summary":
            formulas = ("Na+", "e-", "<->", "Δ")
        elif state == "covalent_summary":
            formulas = ("e-", "E↓", "H-H", "!")
        else:
            formulas = ("H+", "10x", "pH", "Σ")
        accents = (RED, CYAN, AMBER, PURPLE)
        positions = ((90, 175), (650, 175), (90, 360), (650, 360))
        for index, item in enumerate(items):
            reveal = _ease((progress - index * 0.14) / 0.32)
            if reveal <= 0:
                continue
            x, y = positions[index]
            card = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay = ImageDraw.Draw(card)
            alpha = round(235 * reveal)
            overlay.rounded_rectangle(
                (x, y + (1 - reveal) * 20, x + 535, y + 145 + (1 - reveal) * 20),
                radius=24,
                fill=(29, 43, 70, alpha),
                outline=accents[index],
                width=3,
            )
            icon_x, icon_y = x + 62, y + 72 + (1 - reveal) * 20
            overlay.ellipse(
                (icon_x - 38, icon_y - 38, icon_x + 38, icon_y + 38),
                fill=accents[index],
            )
            _centered_text(
                overlay,
                (icon_x, icon_y),
                formulas[index],
                self._fonts.get(18, bold=True),
                BACKGROUND,
            )
            font, lines, spacing = _fit_wrapped_text(
                overlay,
                item,
                max_width=390,
                max_height=94,
                start_size=24,
                min_size=17,
                bold=True,
                fonts=self._fonts,
            )
            line_height = getattr(font, "size", 18)
            total = len(lines) * line_height + max(0, len(lines) - 1) * spacing
            text_y = icon_y - total / 2
            for line in lines:
                overlay.text((x + 120, text_y), line, font=font, fill=TEXT)
                text_y += line_height + spacing
            image.alpha_composite(card) if image.mode == "RGBA" else image.paste(
                card.convert("RGB"), (0, 0), card
            )


def render_video(
    storyboard: Storyboard,
    audio_paths: Sequence[str | Path],
    output_path: str | Path,
    **renderer_options: Any,
) -> RenderResult:
    """Functional convenience API for the common one-shot render path."""

    return Renderer(**renderer_options).render(storyboard, audio_paths, output_path)


# A discoverable synonym for callers thinking in storyboard terminology.
render_storyboard = render_video


__all__ = [
    "FINAL_VIDEO_FILENAME",
    "FPS",
    "HEIGHT",
    "RenderError",
    "RenderResult",
    "Renderer",
    "RendererError",
    "SCENE_PADDING_SECONDS",
    "WIDTH",
    "render_storyboard",
    "render_video",
]
