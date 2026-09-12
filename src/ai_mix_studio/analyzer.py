from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import AudioBuffer


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    duration_seconds: float
    channels: int


def _dbfs(value: float, floor: float = -120.0) -> float:
    if value <= 0.0:
        return floor
    return max(floor, float(20.0 * np.log10(value)))


def analyze_audio(buffer: AudioBuffer) -> AnalysisResult:
    samples = np.asarray(buffer.samples, dtype=np.float64)
    if samples.ndim == 2:
        mono = np.mean(samples, axis=1)
    else:
        mono = samples

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    crest = 20.0 * np.log10(peak / rms) if rms > 0 and peak > 0 else 0.0

    return AnalysisResult(
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        crest_factor_db=float(crest),
        duration_seconds=buffer.duration_seconds,
        channels=buffer.channels,
    )
