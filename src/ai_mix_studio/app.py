from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from .analyzer import analyze_audio
from .audio import load_audio
from .mix_advisor import recommend
from .models import Project, Track


class MixStudioApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Mix Studio")
        self.geometry("1200x760")
        self.minsize(960, 640)
        self.project = Project()
        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self, padx=12, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="AI MIX STUDIO", font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Button(top, text="Import Audio", command=self.import_audio).pack(side="right", padx=(8, 0))
        tk.Button(top, text="Analyze Selected", command=self.analyze_selected).pack(side="right")

        body = tk.PanedWindow(self, sashrelief="raised", orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = tk.Frame(body, padx=8, pady=8)
        center = tk.Frame(body, padx=8, pady=8)
        right = tk.Frame(body, padx=8, pady=8)
        body.add(left, minsize=250)
        body.add(center, minsize=430)
        body.add(right, minsize=300)

        tk.Label(left, text="TRACKS", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.track_list = tk.Listbox(left, exportselection=False)
        self.track_list.pack(fill="both", expand=True, pady=(8, 0))

        tk.Label(center, text="TIMELINE / WORKSPACE", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.workspace = tk.Text(center, wrap="word", state="disabled")
        self.workspace.pack(fill="both", expand=True, pady=(8, 0))
        self._workspace("Create a session by importing audio stems.\n\nThis first build focuses on a stable local-first foundation.\n")

        tk.Label(right, text="AI MIX ASSISTANT", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.advice = tk.Text(right, wrap="word", state="disabled")
        self.advice.pack(fill="both", expand=True, pady=(8, 0))

        self.status = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status, anchor="w", padx=12, pady=6).pack(fill="x")

    def _workspace(self, text: str) -> None:
        self.workspace.configure(state="normal")
        self.workspace.delete("1.0", "end")
        self.workspace.insert("1.0", text)
        self.workspace.configure(state="disabled")

    def _advice(self, text: str) -> None:
        self.advice.configure(state="normal")
        self.advice.delete("1.0", "end")
        self.advice.insert("1.0", text)
        self.advice.configure(state="disabled")

    def import_audio(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Import audio",
            filetypes=[
                ("Audio", "*.wav *.flac *.aiff *.aif"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        for raw in paths:
            path = Path(raw)
            track = Track(name=path.stem, source=path)
            self.project.add_track(track)
            self.track_list.insert("end", track.name)

        self.status.set(f"Imported {len(paths)} track(s)")
        self._workspace("\n".join(f"• {track.name} — {track.source}" for track in self.project.tracks))

    def analyze_selected(self) -> None:
        selection = self.track_list.curselection()
        if not selection:
            messagebox.showinfo("AI Mix Studio", "Select a track first.")
            return

        track = self.project.tracks[selection[0]]
        try:
            result = analyze_audio(load_audio(track.source))
        except Exception as exc:  # pragma: no cover - GUI error path
            messagebox.showerror("Analysis failed", str(exc))
            return

        lines = [
            f"Track: {track.name}",
            f"Duration: {result.duration_seconds:.2f} s",
            f"Channels: {result.channels}",
            f"Peak: {result.peak_dbfs:.2f} dBFS",
            f"RMS: {result.rms_dbfs:.2f} dBFS",
            f"Crest factor: {result.crest_factor_db:.2f} dB",
            "",
            "RECOMMENDATIONS",
        ]
        lines.extend(
            f"[{item.priority.upper()}] {item.title}\n{item.detail}\n" for item in recommend(result)
        )
        self._advice("\n".join(lines))
        self.status.set(f"Analyzed: {track.name}")


def main() -> None:
    app = MixStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
