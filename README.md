# AI Mix Studio

A local-first desktop music production and mixing application with a deterministic measurement engine for mixing engineers. **No AI/ML components** — all analysis is classic DSP (numpy/scipy).

## Goals

- Import and organize audio stems
- Play and arrange tracks on a timeline
- Analyze audio locally and show measurements that help a mixing engineer:
  - Sample peak (dBFS) and true peak (dBTP, 4x oversampled)
  - RMS level and crest factor (dynamics)
  - LUFS integrated / momentary and LRA loudness (ITU-R BS.1770-4 K-weighted, gated)
  - Stereo: mid/side RMS, stereo width, L/R phase correlation
  - Masking: octave-band L/R spectral overlap, spectral centroid
- Apply non-destructive mix decisions through a mixer/control surface
- Compare source and processed audio
- Export WAV/MP3 and analysis reports
- Provide a foundation for VST3 integration later

## Architecture

```text
Mix Studio Desktop UI
        |
        +-- Project / Session Layer
        +-- Audio Engine
        +-- Analysis / Metering Engine   (peak, dBTP, RMS, LUFS, LRA, stereo, correlation, masking)
        +-- DSP / Processing Graph
        +-- Export / Reporting
        +-- Plugin Host (future VST3)
```

## Development status

Phase 1 establishes the application shell, project model, audio-engine abstraction, and the measurement/analyzer engine. Timeline arrangement, DSP processing and export follow next.

This repository is intentionally independent of `AI_MIX_ANALYZER`.
