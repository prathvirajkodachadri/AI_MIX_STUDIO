from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


TrackRole = Literal["vocals", "drums", "bass", "guitar", "keys", "fx", "other"]


@dataclass(slots=True)
class Track:
    name: str
    source: Path
    role: TrackRole = "other"
    gain_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False


@dataclass(slots=True)
class Project:
    name: str = "Untitled Session"
    sample_rate: int = 48_000
    bpm: float = 120.0
    tracks: list[Track] = field(default_factory=list)

    def add_track(self, track: Track) -> None:
        self.tracks.append(track)
