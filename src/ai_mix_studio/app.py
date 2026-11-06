from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .analyzer import analyze_audio
from .audio import load_audio
from .models import Project, Track


class MixStudioWindow(QMainWindow):
    """Local-first mixing workstation UI; analysis engine provides measurements only (no AI)."""

    def __init__(self) -> None:
        super().__init__()
        self.project = Project()
        self.setWindowTitle("Mix Studio — Analysis & Metering")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self._build_menu()
        self._build_toolbar()
        self._build_workspace()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — import stems to begin")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        import_action = QAction("Import Audio…", self)
        import_action.triggered.connect(self.import_audio)
        file_menu.addAction(import_action)
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(QAction("Undo", self))
        edit_menu.addAction(QAction("Redo", self))

        self.menuBar().addMenu("Track")
        self.menuBar().addMenu("Mix")
        self.menuBar().addMenu("View")
        self.menuBar().addMenu("Help")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Transport", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        for label in ("◀◀", "▶", "■", "●", "Loop"):
            button = QPushButton(label)
            button.setMinimumWidth(62)
            toolbar.addWidget(button)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  00:00:00.000  "))
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("BPM"))
        bpm = QPushButton("120.0")
        bpm.setMinimumWidth(70)
        toolbar.addWidget(bpm)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  4/4  |  48 kHz  |  24-bit  "))
        self.addToolBar(toolbar)

    def _panel(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        heading = QLabel(title)
        heading.setObjectName("panelTitle")
        layout.addWidget(heading)
        return frame, layout

    def _build_workspace(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)

        main_split = QSplitter(Qt.Horizontal)
        root_layout.addWidget(main_split, 1)

        tracks_panel, tracks_layout = self._panel("TRACKS")
        self.track_list = QListWidget()
        self.track_list.setAlternatingRowColors(True)
        self.track_list.currentRowChanged.connect(self._track_selected)
        tracks_layout.addWidget(self.track_list, 1)
        import_btn = QPushButton("＋ Import Stems")
        import_btn.clicked.connect(self.import_audio)
        tracks_layout.addWidget(import_btn)
        main_split.addWidget(tracks_panel)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        timeline_panel, timeline_layout = self._panel("ARRANGEMENT / TIMELINE")
        timeline_info = QLabel(
            "TIME  00:00 ─────────────────────────────────────────────────────────────\n\n"
            "Track lanes will host waveforms, clips, automation, markers and snap/grid editing."
        )
        timeline_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        timeline_info.setMinimumHeight(390)
        timeline_layout.addWidget(timeline_info, 1)
        mixer_panel, mixer_layout = self._panel("MIXER")
        mixer_layout.addWidget(QLabel("Master   ┃  Peak ─────────  LUFS ─────────  Stereo Correlation ─────"))
        mixer_layout.addWidget(QLabel("[EQ]   [Comp]   [Saturation]   [FX]   [Fader]   [Pan]"))
        center_layout.addWidget(timeline_panel, 3)
        center_layout.addWidget(mixer_panel, 1)
        main_split.addWidget(center)

        meter_panel, meter_layout = self._panel("ANALYSIS / METERS")
        self.meter_label = QLabel(
            "Import stems, then select a track.\n\n"
            "Deterministic local measurements for the mixing engineer:\n"
            "• peak (dBFS) and true peak (dBTP)\n"
            "• RMS, crest factor, dynamics\n"
            "• LUFS integrated / momentary, LRA (ITU-R BS.1770)\n"
            "• stereo width, mid/side levels, phase correlation\n"
            "• octave-band masking / spectral overlap, spectral centroid"
        )
        self.meter_label.setWordWrap(True)
        self.meter_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        meter_layout.addWidget(self.meter_label, 1)
        analyze_btn = QPushButton("Analyze Selected Track")
        analyze_btn.clicked.connect(self.analyze_selected)
        meter_layout.addWidget(analyze_btn)
        main_split.addWidget(meter_panel)

        main_split.setSizes([250, 850, 340])
        self.setCentralWidget(root)

    def import_audio(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import audio stems",
            "",
            "Audio (*.wav *.flac *.aiff *.aif *.mp3 *.ogg);;All files (*.*)",
        )
        if not paths:
            return
        first_new = len(self.project.tracks)
        for raw in paths:
            path = Path(raw)
            track = Track(name=path.stem, source=path)
            self.project.add_track(track)
            self.track_list.addItem(track.name)
        # Select the first imported stem so "Analyze Selected Track"
        # works immediately (previously nothing was selected and the
        # button just popped a "select a track" dialog).
        if self.track_list.currentRow() < 0:
            self.track_list.setCurrentRow(first_new)
        self.statusBar().showMessage(f"Imported {len(paths)} stem(s)")

    def _track_selected(self, row: int) -> None:
        if row >= 0 and row < len(self.project.tracks):
            self.statusBar().showMessage(f"Selected: {self.project.tracks[row].name}")

    def analyze_selected(self) -> None:
        row = self.track_list.currentRow()
        if row < 0 or row >= len(self.project.tracks):
            QMessageBox.information(self, "Mix Studio", "Select a track first.")
            return
        track = self.project.tracks[row]
        # Analysis of a full-length stem takes a moment; show progress and
        # a busy cursor instead of appearing frozen with no feedback.
        self.statusBar().showMessage(f"Analyzing {track.name}…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            result = analyze_audio(load_audio(track.source))
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", str(exc))
            self.statusBar().showMessage(f"Analysis failed: {track.name}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        lines = [
            f"{track.name}",
            f"Duration: {result.duration_seconds:.2f} s   {result.sample_rate} Hz   {result.channels} ch",
            "",
            "LEVEL / DYNAMICS",
            f"Peak:        {result.peak_dbfs:.2f} dBFS",
            f"True peak:   {result.true_peak_dbtp:.2f} dBTP",
            f"RMS:         {result.rms_dbfs:.2f} dBFS",
            f"Crest:       {result.crest_factor_db:.2f} dB",
            "",
            "LOUDNESS (BS.1770)",
            f"LUFS int:    {result.lufs_integrated:.2f} LUFS",
            f"LUFS max:    {result.lufs_momentary_max:.2f} LUFS",
            f"LRA:         {result.lra:.2f} LU",
            "",
            "STEREO",
            f"Correlation: {result.stereo_correlation:+.3f}",
            f"Mid RMS:     {result.mid_rms_dbfs:.2f} dBFS",
            f"Side RMS:    {result.side_rms_dbfs:.2f} dBFS",
            f"Width:       {result.stereo_width_db:.2f} dB",
            "",
            "MASKING / SPECTRUM",
            f"Spectral centroid: {result.spectral_centroid_hz:.0f} Hz",
        ]
        for band in result.masking_bands:
            lines.append(
                f"  {band.center_hz:7.0f} Hz  L {band.left_db:7.1f}  R {band.right_db:7.1f}  overlap {band.overlap:.2f}"
            )
        self.meter_label.setText("\n".join(lines))
        self.statusBar().showMessage(f"Analysis complete: {track.name}")


def _apply_theme(app: QApplication) -> None:
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        """
        QMainWindow { background: #15171a; color: #e7e9ec; }
        QWidget { color: #e7e9ec; }
        QFrame#panel { background: #1d2024; border: 1px solid #30343a; border-radius: 5px; }
        QLabel#panelTitle { font-size: 11px; font-weight: 700; letter-spacing: 1px; }
        QPushButton { background: #292d33; border: 1px solid #3a3f46; padding: 7px 12px; border-radius: 4px; }
        QPushButton:hover { background: #353a42; }
        QListWidget { background: #17191d; border: 1px solid #30343a; }
        QListWidget::item { padding: 8px; }
        QListWidget::item:selected { background: #343a43; }
        QToolBar, QMenuBar, QStatusBar { background: #111316; border: 0; }
        QMenuBar::item:selected, QMenu::item:selected { background: #30343a; }
        QSplitter::handle { background: #0e1012; }
        """
    )


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    _apply_theme(app)
    window = MixStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
