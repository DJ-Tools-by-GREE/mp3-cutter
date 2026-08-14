"""audio_edits — DB-agnostic, position-driven audio-editing core.

Every public function takes positions in **seconds** and explicit input/output
paths. This module knows nothing about Engine DJ, djlib, or any database:
callers supply positions from wherever they like — a UI selection, djlib's
``cue_points``, or Engine hotcues via a thin adapter (see ``get_hotcues`` in
``engineDJ_cutByHotCues``, which stays the Engine-specific front-end).

It is the seconds-based facade over the sample-/frame-accurate primitives in
``engineDJ_cutByHotCues``:

    .mp3        frame-accurate  (walk MPEG frames, drop whole frames, no re-encode)
    .flac/.wav  sample-accurate (zero-crossing snap)
    .m4a        decode -> PCM cut -> AAC re-encode (the one lossy path)

Ranges are half-open ``[start_s, end_s)``. Output is always a *new* file; the
core refuses to overwrite the source. Callers own where the output goes
(djlib picks ``<music_root>/edits/``, beside the source, or a custom path).

Deps: numpy + soundfile (FLAC/WAV), mutagen (tags), ffmpeg on PATH (M4A only),
pedalboard (optional, reverb tail only), vocal_remover (optional, stems only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import engineDJ_cutByHotCues as _engine

__all__ = [
    "SUPPORTED_EXTS",
    "AudioInfo",
    "probe",
    "to_samples",
    "build_output_path",
    "cut_between",
    "cut_to_end",
    "insert_silence",
    "compress",
    "copy_between",
    "stem",
    "apply_edit",
]

SUPPORTED_EXTS = (".mp3", ".flac", ".wav", ".m4a")

# Extension -> sample-accurate/frame-accurate primitive (all take SAMPLE positions).
_CUT = {
    ".mp3": _engine.cut_mp3,
    ".flac": _engine.cut_flac,
    ".wav": _engine.cut_wav,
    ".m4a": _engine.cut_m4a,
}
_SILENCE = {
    ".mp3": _engine.insert_silence_mp3,
    ".flac": _engine.insert_silence_flac,
    ".wav": _engine.insert_silence_wav,
    ".m4a": _engine.insert_silence_m4a,
}
_COPY = {
    ".mp3": _engine.copy_beats_mp3,
    ".flac": _engine.copy_beats_flac,
    ".wav": _engine.copy_beats_wav,
    ".m4a": _engine.copy_beats_m4a,
}
_STEM = {
    ".mp3": _engine.stem_separation_mp3,
    ".flac": _engine.stem_separation_flac,
    ".wav": _engine.stem_separation_wav,
    ".m4a": _engine.stem_separation_m4a,
}

# CUT_TO_END reverb tuning still lives as module globals on the engine; map the
# public kwargs onto them so a reverb request is deterministic instead of picking
# up whatever the globals happen to be.
_REVERB_ATTR = {
    "room_size": "CUT_TO_END_REVERB_ROOM_SIZE",
    "damping": "CUT_TO_END_REVERB_DAMPING",
    "wet_level": "CUT_TO_END_REVERB_WET_LEVEL",
    "width": "CUT_TO_END_REVERB_WIDTH",
    "tail_secs": "CUT_TO_END_REVERB_TAIL_SECS",
    "blend_secs": "CUT_TO_END_REVERB_BLEND_SECS",
}


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def probe(path: str) -> AudioInfo:
    """Return sample rate, channel count, and total frame count for *path*."""
    ext = _ext(path)
    if ext in (".flac", ".wav"):
        import soundfile as sf

        info = sf.info(path)
        return AudioInfo(int(info.samplerate), int(info.channels), int(info.frames))
    if ext == ".mp3":
        from mutagen.mp3 import MP3

        info = MP3(path).info
        return AudioInfo(
            int(info.sample_rate),
            int(getattr(info, "channels", 2) or 2),
            int(round(info.length * info.sample_rate)),
        )
    if ext == ".m4a":
        from mutagen.mp4 import MP4

        info = MP4(path).info
        return AudioInfo(
            int(info.sample_rate),
            int(info.channels),
            int(round(info.length * info.sample_rate)),
        )
    raise ValueError(f"unsupported audio format: {ext!r} ({path})")


def to_samples(seconds: float, sample_rate: int) -> int:
    """Convert a position in seconds to a sample index (rounded)."""
    if seconds < 0:
        raise ValueError(f"position must be >= 0, got {seconds}")
    return int(round(seconds * sample_rate))


def build_output_path(input_path: str, out_dir: str, appendix: str = "", ext: str | None = None) -> str:
    """Compose ``<out_dir>/<stem> <appendix><ext>`` from *input_path*.

    Mirrors the standalone tool's OUTPUT_APPENDIX convention. *ext* overrides the
    source extension (e.g. ``.mp3`` for a COMPRESS output).
    """
    stem_name, cur_ext = os.path.splitext(os.path.basename(input_path))
    out_ext = ext or cur_ext
    name = f"{stem_name} {appendix}".rstrip() if appendix else stem_name
    return os.path.join(out_dir, f"{name}{out_ext}")


def _prepare(input_path: str, output_path: str) -> str:
    ext = _ext(input_path)
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"unsupported audio format: {ext!r} ({input_path})")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("output_path must differ from the source (never overwrite the original)")
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return ext


def _apply_reverb_settings(reverb: dict | None) -> None:
    if not reverb:
        return
    for key, attr in _REVERB_ATTR.items():
        if reverb.get(key) is not None:
            setattr(_engine, attr, reverb[key])


def _maybe_title(output_path: str, title: str | None) -> None:
    if title:
        _engine.update_track_title(output_path, title)


# ── Public operations ──────────────────────────────────────────────────────────

def cut_between(input_path: str, output_path: str, start_s: float, end_s: float, *, title: str | None = None) -> str:
    """Remove the audio in ``[start_s, end_s)`` and write the rest to *output_path*."""
    ext = _prepare(input_path, output_path)
    info = probe(input_path)
    a = to_samples(start_s, info.sample_rate)
    b = min(to_samples(end_s, info.sample_rate), info.frames)
    if b <= a:
        raise ValueError(f"end_s ({end_s}) must be after start_s ({start_s})")
    _CUT[ext](input_path, output_path, a, b, False)
    _maybe_title(output_path, title)
    return output_path


def cut_to_end(
    input_path: str,
    output_path: str,
    start_s: float,
    *,
    reverb_tail: bool = False,
    reverb: dict | None = None,
    title: str | None = None,
) -> str:
    """Remove everything from ``start_s`` to the end of the track.

    With ``reverb_tail=True`` a reverb decay is appended (requires ``pedalboard``);
    ``reverb`` may override room_size/damping/wet_level/width/tail_secs/blend_secs.
    """
    ext = _prepare(input_path, output_path)
    info = probe(input_path)
    a = to_samples(start_s, info.sample_rate)
    if a >= info.frames:
        raise ValueError(f"start_s ({start_s}) is at/after end of track ({info.duration_s:.3f}s)")
    if reverb_tail:
        _apply_reverb_settings(reverb)
    _CUT[ext](input_path, output_path, a, info.frames, bool(reverb_tail))
    _maybe_title(output_path, title)
    return output_path


def insert_silence(input_path: str, output_path: str, at_s: float, duration_s: float, *, title: str | None = None) -> str:
    """Insert ``duration_s`` seconds of silence at ``at_s``."""
    ext = _prepare(input_path, output_path)
    info = probe(input_path)
    at = to_samples(at_s, info.sample_rate)
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    _SILENCE[ext](input_path, output_path, at, float(duration_s))
    _maybe_title(output_path, title)
    return output_path


def compress(input_path: str, output_path: str, bitrate_kbps: int = 320, *, remove_artwork: bool = False, title: str | None = None) -> str:
    """Transcode/re-encode to MP3 at *bitrate_kbps* (output is always .mp3)."""
    if _ext(input_path) not in SUPPORTED_EXTS:
        raise ValueError(f"unsupported audio format: {_ext(input_path)!r} ({input_path})")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("output_path must differ from the source (never overwrite the original)")
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    _engine.compress_to_mp3(input_path, output_path, int(bitrate_kbps), bool(remove_artwork))
    _maybe_title(output_path, title)
    return output_path


def copy_between(
    input_path: str,
    output_path: str,
    src_start_s: float,
    src_end_s: float,
    dst_start_s: float,
    dst_end_s: float | None = None,
    *,
    repeat_count: int = 1,
    paste_mode: str = "ADD",
    remove_vocals: bool = False,
    source_track_path: str | None = None,
    title: str | None = None,
) -> str:
    """Copy ``[src_start_s, src_end_s)`` and mix/replace it onto ``dst_start_s``.

    ``source_track_path`` pulls the copied section from a *different* track
    (COPY_BEATS_BETWEEN_TRACKS); its rate/channels are conformed by the engine.
    """
    ext = _prepare(input_path, output_path)
    info = probe(input_path)
    sr = info.sample_rate
    sa = to_samples(src_start_s, sr)
    sb = to_samples(src_end_s, sr)
    if sb <= sa:
        raise ValueError(f"src_end_s ({src_end_s}) must be after src_start_s ({src_start_s})")
    da = to_samples(dst_start_s, sr)
    db = to_samples(dst_end_s, sr) if dst_end_s is not None else None
    mode = "REPLACE" if str(paste_mode).upper() == "REPLACE" else "ADD"
    _COPY[ext](
        input_path,
        output_path,
        sa,
        sb,
        da,
        db,
        max(1, int(repeat_count)),
        mode,
        bool(remove_vocals),
        source_track_path,
    )
    _maybe_title(output_path, title)
    return output_path


def stem(
    input_path: str,
    output_path: str,
    keep: str = "INSTRUMENTAL",
    start_s: float | None = None,
    end_s: float | None = None,
    *,
    title: str | None = None,
) -> str:
    """Replace the whole track (or ``[start_s, end_s)``) with an isolated stem.

    ``keep`` is ``"INSTRUMENTAL"`` (vocals removed) or ``"VOCALS"``. Requires the
    optional ``vocal_remover`` dependency.
    """
    ext = _prepare(input_path, output_path)
    info = probe(input_path)
    keep_norm = "VOCALS" if str(keep).upper() == "VOCALS" else "INSTRUMENTAL"
    ss = to_samples(start_s, info.sample_rate) if start_s is not None else None
    se = to_samples(end_s, info.sample_rate) if end_s is not None else None
    if ss is not None and se is not None and se <= ss:
        raise ValueError(f"end_s ({end_s}) must be after start_s ({start_s})")
    _STEM[ext](input_path, output_path, keep_norm, ss, se)
    _maybe_title(output_path, title)
    return output_path


# Single dispatch entry point, convenient for a server endpoint.
_OPS = {
    "cut_between": cut_between,
    "cut_to_end": cut_to_end,
    "insert_silence": insert_silence,
    "compress": compress,
    "copy_between": copy_between,
    "stem": stem,
}


def apply_edit(op: str, input_path: str, output_path: str, **kwargs) -> str:
    """Dispatch to a named operation. *op* is one of ``_OPS``."""
    try:
        fn = _OPS[op]
    except KeyError:
        raise ValueError(f"unknown op {op!r}; valid: {', '.join(sorted(_OPS))}") from None
    return fn(input_path, output_path, **kwargs)
