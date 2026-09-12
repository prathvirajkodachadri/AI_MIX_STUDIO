# AI Mix Studio

A local-first desktop music production and AI-assisted mixing application.

## Goals

- Import and organize audio stems
- Play and arrange tracks on a timeline
- Analyze levels, dynamics, frequency balance, loudness, and stereo image locally
- Provide explainable AI mixing recommendations
- Apply non-destructive mix decisions through a mixer/control surface
- Compare source and processed audio
- Export WAV/MP3 and analysis reports
- Provide a foundation for VST3 integration later

## Architecture

```text
AI Mix Studio Desktop UI
        |
        +-- Project / Session Layer
        +-- Audio Engine
        +-- Analysis Engine
        +-- Mix Recommendation Engine
        +-- DSP / Processing Graph
        +-- Export / Reporting
        +-- Plugin Host (future VST3)
```

## Development status

Phase 1 establishes the application shell, project model, audio-engine abstraction, analyzer interfaces, and a clean path for future DSP and AI capabilities.

This repository is intentionally independent of `AI_MIX_ANALYZER`.
