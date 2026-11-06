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

## Running an analyzer

All three front-ends share the same measurements (peak/dBTP, RMS, LUFS/LRA, stereo, masking) and accept WAV/FLAC/AIFF/MP3/OGG, mono or stereo, full-length stems:

```bash
pip install -r requirements.txt   # includes Flask for the web analyzer
```

- **Browser (no server):** open `index.html` — import stems, select a track, *Analyze Selected*, or *Analyze Session* for bus + masking. Everything runs client-side.
- **Web server:** `python web_analyzer.py` → http://localhost:8000 (one file = track analysis; many files = session analysis).
- **Desktop:** `python main.py`.

## Tests

```bash
python -m pytest tests/ -q          # engine + session tests
node --test tests/test_browser_dsp.mjs   # browser (index.html) DSP tests
```
