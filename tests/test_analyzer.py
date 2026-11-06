from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_mix_studio.analyzer import analyze_audio
from ai_mix_studio.audio import load_audio


def _write_tone(path: Path, freq: float = 440.0, seconds: float = 1.0, stereo: bool = False) -> None:
    sr = 48_000
    t = np.arange(int(sr * seconds)) / sr
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    data = np.stack([tone, tone], axis=1) if stereo else tone
    sf.write(path, data.astype(np.float32), sr)


def test_mono_levels(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    samples = np.zeros(4800, dtype=np.float32)
    samples[100:200] = 0.5
    sf.write(path, samples, 48_000)

    result = analyze_audio(load_audio(path))

    assert result.channels == 1
    assert result.duration_seconds == 0.1
    assert result.peak_dbfs < 0
    assert result.rms_dbfs < result.peak_dbfs
    assert result.true_peak_dbtp >= result.peak_dbfs - 0.1


def test_full_tone_loudness_and_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    _write_tone(path, stereo=True)

    result = analyze_audio(load_audio(path))

    assert result.channels == 2
    # Identical L/R: perfect correlation, no side energy.
    assert result.stereo_correlation == pytest.approx(1.0, abs=1e-6)
    assert result.side_rms_dbfs <= -100.0
    # 0.5-amplitude sine => RMS -9.03 dBFS, LUFS near that (K-weighting ~flat at 1 kHz).
    assert abs(result.rms_dbfs - (-9.03)) < 0.2
    assert -14.0 < result.lufs_integrated < -6.0
    assert result.lra < 1.0  # steady tone has no loudness range
    assert len(result.masking_bands) > 0
    assert result.spectral_centroid_hz > 0


def test_anti_phase_stereo(tmp_path: Path) -> None:
    sr = 48_000
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    path = tmp_path / "anti.wav"
    sf.write(path, np.stack([tone, -tone], axis=1).astype(np.float32), sr)

    result = analyze_audio(load_audio(path))

    assert result.stereo_correlation == pytest.approx(-1.0, abs=1e-6)
    assert result.side_rms_dbfs > result.mid_rms_dbfs
