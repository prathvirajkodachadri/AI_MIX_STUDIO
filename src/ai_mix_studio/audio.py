from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def load_audio(path: Path) -> AudioBuffer:
    if not path.exists():
        raise FileNotFoundError(path)
    samples, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    return AudioBuffer(samples=samples, sample_rate=sample_rate)
