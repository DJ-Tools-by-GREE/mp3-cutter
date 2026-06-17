#!/usr/bin/env python3
"""
Vocal removal via `audio-separator` (UVR / MDX-Net models).

Single entry point: `remove_vocals(input_path, ...)` → returns the path to the
instrumental file (and optionally the vocals stem). Models are downloaded on
first use and cached under `~/.cache/audio-separator/` (or wherever
`audio-separator` puts them) — subsequent runs are offline.

Run directly to process a file:
    python vocal_remover.py "Starships.mp3"
    python vocal_remover.py "song.flac" --output-dir ./stems --keep-vocals
    python vocal_remover.py "song.mp3" --model "UVR-MDX-NET-Inst_HQ_3.onnx"

Install:
    pip install "audio-separator[cpu]"        # CPU-only
    pip install "audio-separator[gpu]"        # CUDA
    # On Apple Silicon, the default install picks up CoreML / MPS automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Optional, Tuple


# A small curated list of high-quality vocal/instrumental models known to ship
# via audio-separator. Picked for "best quality on a typical pop/EDM mix."
# See: https://github.com/nomadkaraoke/python-audio-separator#supported-models
DEFAULT_MODEL = "UVR-MDX-NET-Inst_HQ_4.onnx"

KNOWN_GOOD_MODELS = (
    # MDX-Net — generally the best vocal/instrumental separation today.
    "UVR-MDX-NET-Inst_HQ_4.onnx",     # newer, slightly cleaner instrumentals
    "UVR-MDX-NET-Inst_HQ_3.onnx",     # well-tested fallback
    "UVR-MDX-NET-Voc_FT.onnx",        # tuned to keep the vocal cleaner
    # Demucs — second opinion / different artifact profile.
    "htdemucs.yaml",
    "htdemucs_ft.yaml",
)


def remove_vocals(
    input_path: str,
    output_dir: Optional[str] = None,
    model_filename: str = DEFAULT_MODEL,
    keep_vocals: bool = False,
    output_format: str = "FLAC",
) -> dict:
    """
    Separate `input_path` into vocals + instrumental and write the instrumental
    (and optionally the vocals) next to it.

    Parameters
    ----------
    input_path : str
        Path to an audio file (mp3, flac, wav, m4a, …).
    output_dir : str, optional
        Where to write stems. Defaults to the input file's directory.
    model_filename : str
        Which audio-separator model to load. Must be one that splits into
        vocals + instrumental (two-stem). Demucs YAMLs work too — they just
        produce 4 stems and we pick out vocals/no_vocals.
    keep_vocals : bool
        If False (default), only the instrumental file is kept; the vocals
        stem is deleted after separation.
    output_format : str
        "FLAC" (default, lossless), "WAV", or "MP3". Lossless is recommended
        — re-encoding to lossy after separation compounds artifacts.

    Returns
    -------
    dict
        {"instrumental": <path>, "vocals": <path or None>, "model": <name>}
    """
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise ImportError(
            "audio-separator is not installed. Install with:\n"
            "    pip install \"audio-separator[cpu]\"\n"
            "or [gpu] for CUDA. See https://github.com/nomadkaraoke/python-audio-separator"
        ) from e

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    sep = Separator(
        output_dir=output_dir,
        output_format=output_format,
        # log_level=logging.INFO  # uncomment if you want progress chatter
    )
    sep.load_model(model_filename=model_filename)

    # Returns a list of file paths the separator wrote.
    output_files = sep.separate(input_path)
    output_files = [
        f if os.path.isabs(f) else os.path.join(output_dir, f)
        for f in output_files
    ]

    instrumental = _pick_stem(output_files, ("instrumental", "no_vocals", "other"))
    vocals = _pick_stem(output_files, ("vocals",))

    if instrumental is None:
        raise RuntimeError(
            f"Could not find an instrumental stem in separator output: {output_files}"
        )

    if not keep_vocals and vocals and os.path.isfile(vocals):
        try:
            os.remove(vocals)
            vocals = None
        except OSError:
            pass  # not fatal if cleanup fails

    return {
        "instrumental": instrumental,
        "vocals": vocals,
        "model": model_filename,
    }


def _pick_stem(paths, keywords):
    """Return the first path whose filename contains any of `keywords` (case-insensitive)."""
    for p in paths:
        name = os.path.basename(p).lower()
        if any(k in name for k in keywords):
            return p
    return None


# ---------------------------------------------------------------------------
# Cached helper for use from other modules (e.g. the copy-beats mixer)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.expanduser("~/.cache/mp3-cutter/instrumentals")


def _cache_key(input_path: str, model_filename: str) -> str:
    """Stable cache key from absolute path + mtime + model name. Mtime invalidates
    the cache automatically if the user re-saves the source file."""
    abs_path = os.path.abspath(input_path)
    mtime = os.path.getmtime(abs_path)
    h = hashlib.sha1(f"{abs_path}|{mtime}|{model_filename}".encode("utf-8")).hexdigest()
    return h[:16]


def get_instrumental_pcm(
    input_path: str,
    sample_rate: int,
    n_channels: int = 2,
    model_filename: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> "np.ndarray":  # type: ignore[name-defined]
    """
    Return the instrumental (vocal-removed) version of `input_path` as a
    `(n_samples, n_channels)` float64 numpy array, resampled / channel-matched
    to the host track's sample rate and channel count.

    Result is cached as a FLAC under `~/.cache/mp3-cutter/instrumentals/`,
    keyed by (path, mtime, model). Re-running on the same file is instant.

    Designed to be called from the copy-beats mixer: separation runs once on
    the *full track* (so the model sees full musical context — no chunk-edge
    artifacts at the cue boundaries), and the caller slices the section it
    wants from the returned array.
    """
    import numpy as np
    import soundfile as sf

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(input_path, model_filename)
    cached_flac = os.path.join(CACHE_DIR, f"{key}.flac")

    if not (use_cache and os.path.isfile(cached_flac)):
        # Run the separator into a temp dir, then move the instrumental into the cache.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = remove_vocals(
                input_path=input_path,
                output_dir=tmp,
                model_filename=model_filename,
                keep_vocals=False,
                output_format="FLAC",
            )
            os.replace(result["instrumental"], cached_flac)

    pcm, sr = sf.read(cached_flac, dtype="float64", always_2d=True)

    # Match channel count to host track.
    if pcm.shape[1] != n_channels:
        if pcm.shape[1] == 1 and n_channels == 2:
            pcm = np.repeat(pcm, 2, axis=1)
        elif pcm.shape[1] == 2 and n_channels == 1:
            pcm = pcm.mean(axis=1, keepdims=True)
        else:
            raise ValueError(
                f"Cannot reconcile instrumental channels ({pcm.shape[1]}) "
                f"with host track channels ({n_channels})."
            )

    # Match sample rate to host track. The separator works at its native rate
    # (often 44.1 kHz); if the host is at 48 kHz we resample with libsamplerate
    # via soxr (preferred) or scipy as a fallback.
    if sr != sample_rate:
        pcm = _resample(pcm, sr, sample_rate)

    return pcm


def _resample(pcm, src_sr, dst_sr):
    """Resample (n_samples, n_channels) float64 PCM. Tries soxr → librosa → scipy."""
    import numpy as np
    if src_sr == dst_sr:
        return pcm
    try:
        import soxr
        return soxr.resample(pcm, src_sr, dst_sr).astype(np.float64)
    except ImportError:
        pass
    try:
        import librosa
        # librosa expects (n_channels, n_samples)
        return librosa.resample(pcm.T, orig_sr=src_sr, target_sr=dst_sr).T.astype(np.float64)
    except ImportError:
        pass
    # Last resort: scipy polyphase. Lower quality than soxr but available everywhere.
    from math import gcd
    from scipy.signal import resample_poly  # type: ignore
    g = gcd(int(src_sr), int(dst_sr))
    up, down = int(dst_sr) // g, int(src_sr) // g
    return resample_poly(pcm, up, down, axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description="Remove vocals from an audio file using audio-separator (UVR / MDX-Net).",
    )
    parser.add_argument("input", help="Path to input audio file (mp3/flac/wav/…).")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: alongside the input file).",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"Model filename for audio-separator (default: {DEFAULT_MODEL}). "
             f"Other good picks: {', '.join(KNOWN_GOOD_MODELS[1:])}",
    )
    parser.add_argument(
        "--keep-vocals", action="store_true",
        help="Also keep the vocals stem (default: delete it, keep only the instrumental).",
    )
    parser.add_argument(
        "--format", default="FLAC", choices=("FLAC", "WAV", "MP3"),
        help="Output format (default: FLAC, lossless).",
    )
    args = parser.parse_args()

    try:
        result = remove_vocals(
            input_path=args.input,
            output_dir=args.output_dir,
            model_filename=args.model,
            keep_vocals=args.keep_vocals,
            output_format=args.format,
        )
    except (FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"model:        {result['model']}")
    print(f"instrumental: {result['instrumental']}")
    if result["vocals"]:
        print(f"vocals:       {result['vocals']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
