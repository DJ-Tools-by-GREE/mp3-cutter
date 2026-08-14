"""Round-trip tests for the DB-agnostic audio_edits core.

Run in an isolated env (no project deps needed beyond the engine's mandatory
numpy + soundfile + mutagen):

    uv run --no-project --with numpy --with soundfile --with mutagen \
        --with pytest python -m pytest tests/test_audio_edits.py -q

The lossless FLAC/WAV path uses zero-crossing snap (_ZC_WINDOW=500 samples each
side), so cut lengths are asserted within that tolerance, not exactly.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_edits as ae
from engineDJ_cutByHotCues import _ZC_WINDOW

SR = 44100
DUR_S = 4.0
FRAMES = int(SR * DUR_S)
# A cut's snapped edges can each drift up to _ZC_WINDOW samples looking for a
# zero crossing; two edges → 2x on each side.
TOL = 2 * _ZC_WINDOW + 2


def _write_sine(path: str, freq: float = 220.0) -> None:
    t = np.arange(FRAMES) / SR
    mono = 0.2 * np.sin(2 * np.pi * freq * t)
    stereo = np.column_stack([mono, mono]).astype(np.float32)
    sf.write(path, stereo, SR)


@pytest.fixture()
def src(tmp_path):
    p = str(tmp_path / "sine.wav")
    _write_sine(p)
    return p


def test_probe(src):
    info = ae.probe(src)
    assert info.sample_rate == SR
    assert info.channels == 2
    assert info.frames == FRAMES
    assert info.duration_s == pytest.approx(DUR_S, abs=1e-6)


def test_to_samples():
    assert ae.to_samples(1.0, 44100) == 44100
    assert ae.to_samples(0.5, 48000) == 24000
    with pytest.raises(ValueError):
        ae.to_samples(-0.1, 44100)


def test_build_output_path():
    out = ae.build_output_path("/m/Song.flac", "/m/edits", appendix="(Edit)")
    assert out == os.path.join("/m/edits", "Song (Edit).flac")
    # ext override (COMPRESS → mp3)
    out2 = ae.build_output_path("/m/Song.flac", "/m/edits", appendix="(320)", ext=".mp3")
    assert out2 == os.path.join("/m/edits", "Song (320).mp3")
    # no appendix keeps the stem
    out3 = ae.build_output_path("/m/Song.wav", "/out")
    assert out3 == os.path.join("/out", "Song.wav")


def test_insert_silence_exact(src, tmp_path):
    out = str(tmp_path / "out.wav")
    ae.insert_silence(src, out, at_s=2.0, duration_s=1.0)
    info = ae.probe(out)
    # Silence insertion is not zero-crossing-snapped: length is exact.
    assert info.frames == FRAMES + round(1.0 * SR)
    assert info.sample_rate == SR
    assert info.channels == 2


def test_cut_between_reduces_length(src, tmp_path):
    out = str(tmp_path / "out.wav")
    ae.cut_between(src, out, start_s=1.0, end_s=3.0)
    info = ae.probe(out)
    expected = FRAMES - round(2.0 * SR)
    assert abs(info.frames - expected) <= TOL


def test_cut_to_end_trims(src, tmp_path):
    out = str(tmp_path / "out.wav")
    ae.cut_to_end(src, out, start_s=2.0)
    info = ae.probe(out)
    expected = round(2.0 * SR)
    assert abs(info.frames - expected) <= TOL


def test_apply_edit_dispatch(src, tmp_path):
    out = str(tmp_path / "out.wav")
    ae.apply_edit("cut_between", src, out, start_s=1.0, end_s=2.0)
    assert os.path.isfile(out)
    with pytest.raises(ValueError):
        ae.apply_edit("nope", src, out, start_s=0, end_s=1)


def test_never_overwrites_source(src):
    with pytest.raises(ValueError):
        ae.cut_between(src, src, start_s=1.0, end_s=2.0)
    with pytest.raises(ValueError):
        ae.insert_silence(src, src, at_s=1.0, duration_s=1.0)


def test_bad_ranges(src, tmp_path):
    out = str(tmp_path / "out.wav")
    with pytest.raises(ValueError):
        ae.cut_between(src, out, start_s=3.0, end_s=1.0)  # end before start
    with pytest.raises(ValueError):
        ae.cut_to_end(src, out, start_s=DUR_S + 1.0)  # past end
    with pytest.raises(ValueError):
        ae.insert_silence(src, out, at_s=1.0, duration_s=0.0)  # non-positive


def test_unsupported_ext(tmp_path):
    fake = str(tmp_path / "x.ogg")
    with open(fake, "wb") as fh:
        fh.write(b"\0")
    with pytest.raises(ValueError):
        ae.probe(fake)
    with pytest.raises(ValueError):
        ae.cut_between(fake, str(tmp_path / "o.ogg"), 0.0, 1.0)
