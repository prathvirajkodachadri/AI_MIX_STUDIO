"""Local audio measurement engine for mixing engineers.

Computes level, loudness, dynamics, stereo and masking metrics.
No AI/ML components — deterministic DSP only (numpy/scipy).

Performance notes: every stage is streaming/chunked so full-length stems
(minutes long, tens of MB) analyze in about a second instead of building
whole-file oversampled/FFT intermediates that needed >1 GB of RAM:

* loudness K-weights the signal once and derives every 400 ms block from a
  cumulative sum of squares (previously each overlapping block was
  re-filtered from scratch, with the filter state reset every block);
* true peak oversamples in chunks with context overlap instead of
  resampling the whole file to 4x in one array;
* masking bands / spectral centroid use Welch-averaged spectra over a
  bounded number of windowed segments instead of one FFT of the whole
  file (previously the mono spectrum was even computed twice).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter, resample_poly

from .audio import AudioBuffer

# ITU-R BS.1770-4 K-weighting filter coefficients (48 kHz reference design).
# For other sample rates the filter is applied after resampling to 48 kHz.
_K_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_RLB_B = np.array([1.0, -2.0, 1.0])
_K_RLB_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

_OVERSAMPLE = 4  # for true-peak (dBTP) estimation
_TRUE_PEAK_CHUNK = 1 << 16  # samples per chunk (context overlap on both sides)
_TRUE_PEAK_CONTEXT = 256
_SPECTRUM_SEGMENT = 65536  # Welch segment length for masking/centroid
_SPECTRUM_MAX_SEGMENTS = 64  # evenly spaced segments are enough for band stats
_BAND_EDGES = [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]


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
    mono = samples[:, 0] if samples.ndim == 2 else samples
    return mono, mono


def _as_2d(samples: np.ndarray) -> np.ndarray:
    """Return samples as a (frames, channels) float64 array."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _to_48k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 48_000:
        return samples
    gcd = int(np.gcd(sample_rate, 48_000))
    return resample_poly(samples, 48_000 // gcd, sample_rate // gcd, axis=0)


def _k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply BS.1770 K-weighting, resampling to 48 kHz when needed."""
    y = _to_48k(samples, sample_rate)
    y = lfilter(_K_SHELF_B, _K_SHELF_A, y, axis=0)
    y = lfilter(_K_RLB_B, _K_RLB_A, y, axis=0)
    return y


def _channel_weights(channels: int) -> np.ndarray:
    if channels >= 3:
        return np.array([1.0, 1.0, 1.41] + [1.41] * (channels - 3))
    return np.ones(channels)


def _integrated_loudness(samples: np.ndarray, sample_rate: int, channels: int) -> tuple[float, float, float]:
    """Return (integrated LUFS, max momentary LUFS, LRA).

    The signal is K-weighted once; each 400 ms block's mean square is then
    read off a cumulative sum, so cost is O(n) regardless of file length.
    """
    data = _as_2d(samples)
    if data.shape[0] == 0:
        return -120.0, -120.0, 0.0
    weighted = _k_weight(data, sample_rate)
    weights = _channel_weights(weighted.shape[1])
    # Weighted sum of squares per sample, then a cumulative sum so any
    # block's mean square is a single subtraction.
    square_sum = np.sum(weighted**2 * weights, axis=1)
    prefix = np.concatenate(([0.0], np.cumsum(square_sum)))
    n = weighted.shape[0]

    block = int(0.400 * 48_000)  # 400 ms at the (resampled) 48 kHz rate
    hop = int(0.100 * 48_000)  # 75% overlap
    if n < block:
        mean_sq = float(prefix[n] / n)
        loud = -0.691 + 10.0 * float(np.log10(mean_sq)) if mean_sq > 0 else -120.0
        return loud, loud, 0.0

    starts = np.arange(0, n - block + 1, hop)
    block_ms = (prefix[starts + block] - prefix[starts]) / block
    with np.errstate(divide="ignore"):
        momentary = np.where(block_ms > 0, -0.691 + 10.0 * np.log10(block_ms), -120.0)
    if momentary.size == 0:
        return -120.0, -120.0, 0.0

    short_term = momentary[momentary > -70.0]  # absolute gate
    if short_term.size == 0:
        return -120.0, float(np.max(momentary)), 0.0

    rel_threshold = -0.691 + 10.0 * float(np.log10(np.mean(10 ** (short_term / 10.0)))) - 10.0
    gated = short_term[short_term >= rel_threshold]
    if gated.size == 0:
        return -120.0, float(np.max(momentary)), 0.0
    integ = -0.691 + 10.0 * float(np.log10(np.mean(10 ** (gated / 10.0))))
    ordered = np.sort(gated)
    lo = ordered[int(0.10 * (len(ordered) - 1))]
    hi = ordered[int(0.95 * (len(ordered) - 1))]
    return integ, float(np.max(momentary)), float(hi - lo)


def _true_peak_db(samples: np.ndarray) -> float:
    """4x-oversampled true peak (dBTP), max over channels, computed in chunks."""
    data = _as_2d(samples)
    if data.shape[0] == 0:
        return -120.0
    peak = float(np.max(np.abs(data)))
    n = data.shape[0]
    for ch in range(data.shape[1]):
        chan = data[:, ch]
        if n <= _TRUE_PEAK_CHUNK:
            over = resample_poly(chan, _OVERSAMPLE, 1)
            peak = max(peak, float(np.max(np.abs(over))))
            continue
        for start in range(0, n, _TRUE_PEAK_CHUNK):
            end = min(n, start + _TRUE_PEAK_CHUNK)
            lo = max(0, start - _TRUE_PEAK_CONTEXT)
            hi = min(n, end + _TRUE_PEAK_CONTEXT)
            over = resample_poly(chan[lo:hi], _OVERSAMPLE, 1)
            # Keep only the oversampled region belonging to [start, end).
            a = (start - lo) * _OVERSAMPLE
            b = a + (end - start) * _OVERSAMPLE
            peak = max(peak, float(np.max(np.abs(over[a:b]))))
    return _dbfs(peak)


def _segment_starts(n: int, seg: int) -> list[int]:
    if n <= seg:
        return [0]
    starts = list(range(0, n - seg + 1, seg // 2))
    if len(starts) > _SPECTRUM_MAX_SEGMENTS:
        picks = np.linspace(0, len(starts) - 1, _SPECTRUM_MAX_SEGMENTS).astype(int)
        starts = [starts[i] for i in sorted(set(picks.tolist()))]
    return starts


def _averaged_power(mono_or_chan: np.ndarray, seg: int, starts: list[int], window: np.ndarray) -> np.ndarray:
    acc: np.ndarray | None = None
    for s in starts:
        frame = mono_or_chan[s : s + seg]
        if frame.size < seg:  # short file: zero-pad the single segment
            frame = np.pad(frame, (0, seg - frame.size))
        spec = np.abs(np.fft.rfft(frame * window)) ** 2
        acc = spec if acc is None else acc + spec
    assert acc is not None
    return acc / len(starts)


def _band_mean_square(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float, norm: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    # Positive-frequency bins (except DC/Nyquist, excluded by our band edges)
    # carry half the total energy, hence the factor of 2 in `norm`.
    return float(norm * np.sum(power[mask]))


def _masking_bands(left: np.ndarray, right: np.ndarray, sample_rate: int) -> tuple[tuple[BandMasking, ...], float]:
    """Octave-band spectral overlap between L and R + spectral centroid of the mono mix.

    Uses Welch-averaged spectra over a bounded number of segments, so cost
    stays flat for multi-minute files. Band levels are true band RMS in
    dBFS (a full-band sine reads about its normal RMS level, not +77 dB).
    """
    n = min(left.size, right.size)
    if n < 256:
        return (), 0.0
    seg = min(_SPECTRUM_SEGMENT, 1 << int(np.floor(np.log2(n))))
    seg = max(seg, 256)
    window = np.hanning(seg)
    starts = _segment_starts(n, seg)
    freqs = np.fft.rfftfreq(seg, 1.0 / sample_rate)
    norm = 2.0 / (seg * float(np.sum(window**2)))

    l_power = _averaged_power(left[:n], seg, starts, window)
    r_power = l_power if right is left else _averaged_power(right[:n], seg, starts, window)
    mono = (left[:n] + right[:n]) / 2.0
    m_power = l_power if right is left else _averaged_power(mono, seg, starts, window)

    magnitude = np.sqrt(m_power)
    total_mag = float(np.sum(magnitude))
    centroid = float(np.sum(freqs * magnitude) / total_mag) if total_mag > 0 else 0.0

    edges = [f for f in _BAND_EDGES if f < sample_rate / 2]
    bands: list[BandMasking] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        l_ms = _band_mean_square(l_power, freqs, lo, hi, norm)
        r_ms = _band_mean_square(r_power, freqs, lo, hi, norm)
        overlap = 2.0 * min(l_ms, r_ms) / (l_ms + r_ms) if (l_ms + r_ms) > 0 else 1.0
        bands.append(
            BandMasking(
                center_hz=float(np.sqrt(lo * hi)),
                left_db=_dbfs(np.sqrt(l_ms)),
                right_db=_dbfs(np.sqrt(r_ms)),
                overlap=float(overlap),
            )
        )
    return tuple(bands), centroid


def analyze_audio(buffer: AudioBuffer) -> AnalysisResult:
    samples = np.asarray(buffer.samples, dtype=np.float64)
    left, right = _to_stereo(samples)
    data = _as_2d(samples)

    # Peak/RMS over every channel. (Previously both were taken from the
    # mono sum, so an anti-phase stereo file reported silence.)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    crest = 20.0 * np.log10(peak / rms) if rms > 0 and peak > 0 else 0.0

    lufs_i, lufs_m, lra = _integrated_loudness(samples, buffer.sample_rate, buffer.channels)

    if left.size:
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_rms = float(np.sqrt(np.mean(mid**2)))
        side_rms = float(np.sqrt(np.mean(side**2)))
        if left is right:
            corr = 1.0  # mono (or single-channel) material
        elif np.std(left) > 0 and np.std(right) > 0:
            corr = float(np.corrcoef(left, right)[0, 1])
        else:
            corr = 1.0
        width = 20.0 * np.log10(side_rms / mid_rms) if mid_rms > 0 and side_rms > 0 else -120.0
        bands, centroid = _masking_bands(left, right, buffer.sample_rate)
    else:
        mid_rms, side_rms, corr, width = rms, 0.0, 1.0, -120.0
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
