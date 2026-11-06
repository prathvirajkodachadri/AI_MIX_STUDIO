from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_mix_studio.session import analyze_session


def _write(path: Path, freq: float, seconds: float = 1.0, amp: float = 0.4, phase: float = 0.0) -> None:
    sr = 48_000
    t = np.arange(int(sr * seconds)) / sr
    tone = amp * np.sin(2 * np.pi * freq * t + phase)
    sf.write(path, np.stack([tone, tone], axis=1).astype(np.float32), sr)


def test_session_bus_and_conflicts(tmp_path: Path) -> None:
    # two stems, same band (440 Hz) -> masking conflict; summed bus louder than each
    a = tmp_path / "bass.wav"
    b = tmp_path / "keys.wav"
    _write(a, 440.0, amp=0.4)
    _write(b, 440.0, amp=0.3, phase=0.5)

    report = analyze_session([("bass", a), ("keys", b)])

    assert report.track_count == 2
    assert report.duration_seconds == 1.0
    assert set(report.tracks) == {"bass", "keys"}
    assert report.bus is not None
    assert report.bus.rms_dbfs > report.tracks["bass"].rms_dbfs
    assert any(c.track_a == "bass" and c.track_b == "keys" for c in report.conflicts)
    assert report.summary_lines()


def test_session_distinct_bands_no_conflict(tmp_path: Path) -> None:
    a = tmp_path / "low.wav"
    b = tmp_path / "high.wav"
    _write(a, 100.0, amp=0.4)
    _write(b, 8000.0, amp=0.4)

    report = analyze_session([("low", a), ("high", b)])

    assert report.conflicts == ()


def _write_mono(path: Path, freq: float, seconds: float = 1.0, amp: float = 0.4) -> None:
    sr = 48_000
    t = np.arange(int(sr * seconds)) / sr
    tone = amp * np.sin(2 * np.pi * freq * t)
    sf.write(path, tone.astype(np.float32), sr)


def test_session_mono_stems(tmp_path: Path) -> None:
    # Regression: mono stems used to crash bus summing with
    # "non-broadcastable output operand ... (48000,48000)".
    a = tmp_path / "vocal.wav"
    b = tmp_path / "bass.wav"
    _write_mono(a, 440.0, amp=0.4)
    _write_mono(b, 110.0, amp=0.3)

    report = analyze_session([("vocal", a), ("bass", b)])

    assert report.track_count == 2
    assert report.bus is not None
    assert report.bus.channels == 1
    assert report.bus.rms_dbfs > report.tracks["vocal"].rms_dbfs


def test_session_mixed_mono_and_stereo(tmp_path: Path) -> None:
    a = tmp_path / "vocal.wav"  # mono
    b = tmp_path / "keys.wav"  # stereo
    _write_mono(a, 440.0, amp=0.4)
    _write(b, 440.0, amp=0.3)

    report = analyze_session([("vocal", a), ("keys", b)])

    assert report.track_count == 2
    assert report.bus is not None
    assert report.bus.channels == 2  # mono stem is fed to both bus sides
    assert report.bus.stereo_correlation > 0.9


def test_session_rejects_empty() -> None:
    with pytest.raises(ValueError):
        analyze_session([])
