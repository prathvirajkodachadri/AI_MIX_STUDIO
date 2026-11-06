"""Session-level mixing analysis for multi-stem projects.

Designed for 25-50 full-length stems aligned at 0:00. Streams one stem at a
time (only one full buffer + the bus sum live in memory) and produces:
  - per-stem measurements (reuses analyze_audio)
  - bus (sum of all stems) measurements: clipping/loudness risk on the mix
  - cross-track octave-band masking conflicts, sorted by severity
Deterministic DSP only — no AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .analyzer import AnalysisResult, analyze_audio, _dbfs
from .audio import AudioBuffer, load_audio

BAND_EDGES = [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]
CONFLICT_WINDOW_DB = 30.0  # band counts as "present" if within this of the track's loudest band


@dataclass(frozen=True, slots=True)
class MaskingConflict:
    track_a: str
    track_b: str
    center_hz: float
    level_a_db: float
    level_b_db: float
    severity_db: float  # min of the two band levels — louder overlap = more masking


@dataclass(frozen=True, slots=True)
class SessionReport:
    sample_rate: int
    duration_seconds: float
    track_count: int
    tracks: dict[str, AnalysisResult] = field(default_factory=dict)
    bus: AnalysisResult | None = None
    conflicts: tuple[MaskingConflict, ...] = ()

    def summary_lines(self) -> list[str]:
        lines = [
            f"Session: {self.track_count} stems, {self.duration_seconds:.1f} s @ {self.sample_rate} Hz",
        ]
        if self.bus is not None:
            b = self.bus
            lines += [
                "",
                "BUS (sum of all stems)",
                f"Peak {b.peak_dbfs:.2f} dBFS | True peak {b.true_peak_dbtp:.2f} dBTP | RMS {b.rms_dbfs:.2f} dBFS",
                f"LUFS int {b.lufs_integrated:.2f} | LRA {b.lra:.2f} | Correlation {b.stereo_correlation:+.3f}",
            ]
            if b.true_peak_dbtp > -1.0:
                lines.append("FLAG: bus true peak above -1 dBTP — gain staging needed before summing/export.")
            if b.stereo_correlation < 0.0:
                lines.append("FLAG: bus L/R correlation below 0 — check for out-of-phase stems.")
        if self.conflicts:
            lines += ["", f"MASKING CONFLICTS (top {min(10, len(self.conflicts))} of {len(self.conflicts)})"]
            for c in self.conflicts[:10]:
                lines.append(
                    f"  {c.track_a} vs {c.track_b} @ {c.center_hz:.0f} Hz "
                    f"(L {c.level_a_db:.1f} / {c.level_b_db:.1f} dB)"
                )
        else:
            lines += ["", "MASKING: no significant cross-track band conflicts found."]
        return lines


def _band_energies(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Octave-band energy per track (mono mix), linear scale."""
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)
    n = mono.size
    if n < 256:
        return np.zeros(len(BAND_EDGES) - 1)
    # Welch-style: average periodograms of 64k-sample windows (memory-safe for 15 min stems)
    seg = 65536
    window = np.hanning(seg)
    acc = None
    used = seg
    for start in range(0, n - seg + 1, seg // 2):
        spec = np.abs(np.fft.rfft(mono[start : start + seg] * window)) ** 2
        acc = spec if acc is None else acc + spec
    if acc is None:
        used = n
        acc = np.abs(np.fft.rfft(mono * np.hanning(n))) ** 2
    acc /= max(1, (n - seg) // (seg // 2) + 1) if used == seg else 1
    freqs = np.fft.rfftfreq(used, 1.0 / sample_rate)
    edges = [f for f in BAND_EDGES if f < sample_rate / 2]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        out.append(float(np.sum(acc[m])) if np.any(m) else 0.0)
    return np.array(out)


def analyze_session(stems: list[tuple[str, Path]]) -> SessionReport:
    """Analyze a full session. `stems` = [(name, path), ...], all starting at 0:00."""
    if not stems:
        raise ValueError("no stems provided")

    sample_rate = load_audio(stems[0][1]).sample_rate  # peek for bus allocation
    bus: np.ndarray | None = None
    duration = 0.0
    tracks: dict[str, AnalysisResult] = {}
    band_levels: dict[str, np.ndarray] = {}

    for name, path in stems:
        buf = load_audio(path)
        if buf.sample_rate != sample_rate:
            gcd = np.gcd(buf.sample_rate, sample_rate)
            buf = AudioBuffer(
                resample_poly(buf.samples, sample_rate // gcd, buf.sample_rate // gcd, axis=0),
                sample_rate,
            )
        result = analyze_audio(buf)
        tracks[name] = result
        band_levels[name] = _band_energies(np.asarray(buf.samples, dtype=np.float64), sample_rate)
        duration = max(duration, buf.duration_seconds)

        mono_sum = buf.samples if buf.samples.ndim == 1 else buf.samples
        if bus is None:
            bus = np.zeros((buf.frames, buf.channels), dtype=np.float64)
        if buf.frames > bus.shape[0]:
            pad = np.zeros((buf.frames - bus.shape[0], bus.channels), dtype=np.float64)
            bus = np.vstack([bus, pad])
        bus[: buf.frames] += mono_sum

    bus_result = analyze_audio(AudioBuffer(bus, sample_rate)) if bus is not None else None

    # Pairwise masking conflicts from stored band energies
    names = list(band_levels)
    conflicts: list[MaskingConflict] = []
    centers = [float(np.sqrt(lo * hi)) for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ea, eb = band_levels[a], band_levels[b]
            if ea.size != eb.size:
                continue
            ref_a = ea.max() if ea.size else 0.0
            ref_b = eb.max() if eb.size else 0.0
            for k in range(ea.size):
                if ea[k] <= 0 or eb[k] <= 0:
                    continue
                la, lb = _dbfs(np.sqrt(ea[k])), _dbfs(np.sqrt(eb[k]))
                # both tracks present in this band?
                if la < _dbfs(np.sqrt(ref_a)) - CONFLICT_WINDOW_DB:
                    continue
                if lb < _dbfs(np.sqrt(ref_b)) - CONFLICT_WINDOW_DB:
                    continue
                conflicts.append(
                    MaskingConflict(
                        track_a=a,
                        track_b=b,
                        center_hz=centers[k] if k < len(centers) else 0.0,
                        level_a_db=la,
                        level_b_db=lb,
                        severity_db=min(la, lb),
                    )
                )
    conflicts.sort(key=lambda c: c.severity_db, reverse=True)

    return SessionReport(
        sample_rate=sample_rate,
        duration_seconds=duration,
        track_count=len(stems),
        tracks=tracks,
        bus=bus_result,
        conflicts=tuple(conflicts),
    )
