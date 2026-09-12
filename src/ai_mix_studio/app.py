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
from .mix_advisor import recommend
from .models import Project, Track


class MixStudioWindow(QMainWindow):
    """Professional UI foundation; audio/DSP engines are kept modular for later expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.project = Project()
        self.setWindowTitle("AI Mix Studio — Professional")
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
        self.menuBar().addMenu("AI")
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
        mixer_layout.addWidget(QLabel("[AI Mix Engine]   [EQ]   [Comp]   [Saturation]   [FX]   [Fader]   [Pan]"))
        center_layout.addWidget(timeline_panel, 3)
        center_layout.addWidget(mixer_panel, 1)
        main_split.addWidget(center)

        ai_panel, ai_layout = self._panel("AI MIX ASSISTANT")
        self.ai_label = QLabel(
            "Import stems, then select a track.\n\n"
            "The professional engine will provide:\n"
            "• masking / frequency conflict detection\n"
            "• level and dynamics suggestions\n"
            "• stereo / phase analysis\n"
            "• bus and routing recommendations\n"
            "• reference-track matching\n\n"
            "Every AI action will be previewable, reversible and explainable."
        )
        self.ai_label.setWordWrap(True)
        self.ai_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        ai_layout.addWidget(self.ai_label, 1)
        analyze_btn = QPushButton("Analyze Selected Track")
        analyze_btn.clicked.connect(self.analyze_selected)
        ai_layout.addWidget(analyze_btn)
        main_split.addWidget(ai_panel)

        main_split.setSizes([250, 850, 340])
        self.setCentralWidget(root)

    def import_audio(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import audio stems",
            "",
            "Audio (*.wav *.flac *.aiff *.aif);;All files (*.*)",
        )
        if not paths:
            return
        for raw in paths:
            path = Path(raw)
            track = Track(name=path.stem, source=path)
            self.project.add_track(track)
            self.track_list.addItem(track.name)
        self.statusBar().showMessage(f"Imported {len(paths)} stem(s)")

    def _track_selected(self, row: int) -> None:
        if row >= 0 and row < len(self.project.tracks):
            self.statusBar().showMessage(f"Selected: {self.project.tracks[row].name}")

    def analyze_selected(self) -> None:
        row = self.track_list.currentRow()
        if row < 0 or row >= len(self.project.tracks):
            QMessageBox.information(self, "AI Mix Studio", "Select a track first.")
            return
        track = self.project.tracks[row]
        try:
            result = analyze_audio(load_audio(track.source))
            advice = recommend(result)
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", str(exc))
            return
        lines = [
            f"{track.name}",
            f"Duration: {result.duration_seconds:.2f} s",
            f"Peak: {result.peak_dbfs:.2f} dBFS",
            f"RMS: {result.rms_dbfs:.2f} dBFS",
            f"Crest: {result.crest_factor_db:.2f} dB",
            "",
            "RECOMMENDATIONS",
        ]
        lines.extend(f"[{item.priority.upper()}] {item.title}\n{item.detail}\n" for item in advice)
        self.ai_label.setText("\n".join(lines))
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
