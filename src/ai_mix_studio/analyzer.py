"""Local audio measurement engine for mixing engineers.

Computes level, loudness, dynamics, stereo and masking metrics.
No AI/ML components — deterministic DSP only (numpy/scipy).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

from .audio import AudioBuffer

# ITU-R BS.1770-4 K-weighting filter coefficients (48 kHz reference design).
# For other sample rates the filter is applied after resampling to 48 kHz.
_K_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_RLB_B = np.array([1.0, -2.0, 1.0])
_K_RLB_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

_OVERSAMPLE = 4  # for true-peak (dBTP) estimation


@dataclass(frozen=True, slots=True)
class BandMasking:
    """Per-band spectral overlap between left and right channels."""

    center_hz: float
    left_db: float
    right_db: float
    overlap: float  # 1.0 = identical energy (no masking), 0.0 = one fully masks the other


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    duration_seconds: float
    sample_rate: int
    channels: int
    # Level / dynamics
    peak_dbfs: float
    true_peak_dbtp: float
    rms_dbfs: float
    crest_factor_db: float
    # Loudness (ITU-R BS.1770)
    lufs_integrated: float
    lufs_momentary_max: float  # max 400 ms block loudness
    lra: float  # loudness range (10th–95th percentile of gated short-term blocks)
    # Stereo
    stereo_correlation: float  # Pearson r between L and R, -1..1
    mid_rms_dbfs: float
    side_rms_dbfs: float
    stereo_width_db: float  # 20*log10(side_rms / mid_rms)
    # Masking
    spectral_centroid_hz: float
    masking_bands: tuple[BandMasking, ...] = field(default_factory=tuple)


def _dbfs(value: float, floor: float = -120.0) -> float:
    if value <= 0.0:
        return floor
    return max(floor, float(20.0 * np.log10(value)))


def _to_stereo(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim == 2 and samples.shape[1] >= 2:
        return samples[:, 0], samples[:, 1]
    return samples, samples


def _k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply BS.1770 K-weighting, resampling to 48 kHz when needed."""
    if sample_rate != 48_000:
        from scipy.signal import resample_poly as _rp

        gcd = np.gcd(sample_rate, 48_000)
        samples = _rp(samples, 48_000 // gcd, sample_rate // gcd, axis=0)
    from scipy.signal import lfilter

    y = lfilter(_K_SHELF_B, _K_SHELF_A, samples, axis=0)
    y = lfilter(_K_RLB_B, _K_RLB_A, y, axis=0)
    return y


def _block_loudness_lUFS(block: np.ndarray, channels: int) -> float:
    """Loudness of one block in LUFS (BS.1770-4)."""
    k = _k_weight(block, 48_000)
    if k.ndim == 1:
        mean_sq = float(np.mean(k**2))
        total = mean_sq
    else:
        weights = np.ones(k.shape[1])
        if channels >= 3:
            weights = np.array([1.0, 1.0, 1.41] + [1.41] * max(0, k.shape[1] - 3))
        total = float(np.sum(np.mean(k**2, axis=0) * weights[: k.shape[1]]))
    if total <= 0.0:
        return -120.0
    return -0.691 + 10.0 * float(np.log10(total))


def _integrated_loudness(samples: np.ndarray, sample_rate: int, channels: int) -> tuple[float, float, float]:
    """Return (integrated LUFS, max momentary LUFS, LRA)."""
    if sample_rate != 48_000:
        from scipy.signal import resample_poly as _rp

        gcd = np.gcd(sample_rate, 48_000)
        samples = _rp(samples, 48_000 // gcd, sample_rate // gcd, axis=0)
        sample_rate = 48_000

    block = int(0.400 * sample_rate)  # 400 ms
    hop = int(0.100 * sample_rate)  # 75% overlap
    if samples.shape[0] < block:
        l = _block_loudness_lUFS(samples, channels)
        return l, l, 0.0

    momentary: list[float] = []
    short_term: list[float] = []
    for start in range(0, samples.shape[0] - block + 1, hop):
        bl = samples[start : start + block]
        l = _block_loudness_lUFS(bl, channels)
        momentary.append(l)
        if l > -70.0:  # absolute gate
            short_term.append(l)

    if not momentary:
        return -120.0, -120.0, 0.0

    if not short_term:
        integ = -120.0
        lra = 0.0
    else:
        rel_threshold = -0.691 + 10.0 * np.log10(np.mean(10 ** (np.array(short_term) / 10.0))) - 10.0
        gated = [l for l in short_term if l >= rel_threshold]
        if gated:
            integ = -0.691 + 10.0 * float(np.log10(np.mean(10 ** (np.array(gated) / 10.0))))
            ordered = np.sort(np.array(gated))
            lo = ordered[int(0.10 * (len(ordered) - 1))]
            hi = ordered[int(0.95 * (len(ordered) - 1))]
            lra = float(hi - lo)
        else:
            integ, lra = -120.0, 0.0

    return integ, float(max(momentary)), lra


def _true_peak_db(samples: np.ndarray) -> float:
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)
    if mono.size == 0:
        return -120.0
    over = resample_poly(mono, _OVERSAMPLE, 1)
    return _dbfs(float(np.max(np.abs(over))))


def _masking_bands(left: np.ndarray, right: np.ndarray, sample_rate: int) -> tuple[tuple[BandMasking, ...], float]:
    """Octave-band spectral overlap between L and R + spectral centroid of the mono mix."""
    n = min(left.size, right.size)
    if n < 256:
        return (), 0.0
    window = np.hanning(n)
    l_spec = np.abs(np.fft.rfft(left[:n] * window))
    r_spec = np.abs(np.fft.rfft(right[:n] * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    mono = (left[:n] + right[:n]) / 2.0
    centroid = float(np.sum(freqs * np.abs(np.fft.rfft(mono * window))) / max(np.sum(np.abs(np.fft.rfft(mono * window))), 1e-12))

    edges = [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]
    edges = [f for f in edges if f < sample_rate / 2]
    bands: list[BandMasking] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        le = float(np.sum(l_spec[mask] ** 2))
        re_ = float(np.sum(r_spec[mask] ** 2))
        overlap = 2.0 * min(le, re_) / (le + re_) if (le + re_) > 0 else 1.0
        bands.append(
            BandMasking(
                center_hz=float(np.sqrt(lo * hi)),
                left_db=_dbfs(np.sqrt(le)),
                right_db=_dbfs(np.sqrt(re_)),
                overlap=float(overlap),
            )
        )
    return tuple(bands), centroid


def analyze_audio(buffer: AudioBuffer) -> AnalysisResult:
    samples = np.asarray(buffer.samples, dtype=np.float64)
    left, right = _to_stereo(samples)
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    crest = 20.0 * np.log10(peak / rms) if rms > 0 and peak > 0 else 0.0

    lufs_i, lufs_m, lra = _integrated_loudness(samples, buffer.sample_rate, buffer.channels)

    if samples.ndim == 2 and samples.shape[1] >= 2 and left.size:
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_rms = float(np.sqrt(np.mean(mid**2)))
        side_rms = float(np.sqrt(np.mean(side**2)))
        if np.std(left) > 0 and np.std(right) > 0:
            corr = float(np.corrcoef(left, right)[0, 1])
        else:
            corr = 1.0
        width = 20.0 * np.log10(side_rms / mid_rms) if mid_rms > 0 and side_rms > 0 else -120.0
        bands, centroid = _masking_bands(left, right, buffer.sample_rate)
    else:
        mid_rms = rms
        side_rms = 0.0
        corr = 1.0
        width = -120.0
        bands, centroid = (), 0.0

    return AnalysisResult(
        duration_seconds=buffer.duration_seconds,
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        peak_dbfs=_dbfs(peak),
        true_peak_dbtp=_true_peak_db(samples),
        rms_dbfs=_dbfs(rms),
        crest_factor_db=float(crest),
        lufs_integrated=lufs_i,
        lufs_momentary_max=lufs_m,
        lra=lra,
        stereo_correlation=corr,
        mid_rms_dbfs=_dbfs(mid_rms),
        side_rms_dbfs=_dbfs(side_rms),
        stereo_width_db=float(width),
        spectral_centroid_hz=centroid,
        masking_bands=bands,
    )
