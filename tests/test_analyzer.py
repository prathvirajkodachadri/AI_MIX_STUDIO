from pathlib import Path

import numpy as np
import soundfile as sf

from ai_mix_studio.analyzer import analyze_audio
from ai_mix_studio.audio import load_audio


def test_analyze_audio(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    samples = np.zeros(4800, dtype=np.float32)
    samples[100:200] = 0.5
    sf.write(path, samples, 48_000)

    result = analyze_audio(load_audio(path))

    assert result.channels == 1
    assert result.duration_seconds == 0.1
    assert result.peak_dbfs < 0
    assert result.rms_dbfs < result.peak_dbfs
