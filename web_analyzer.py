"""Online mixing analyzer — web UI around the local measurement engine.

Run:  python web_analyzer.py   (serves on 0.0.0.0:8000)
Upload a WAV/FLAC/AIFF file and get peak, dBTP, RMS, LUFS, LRA, stereo
correlation, width, and octave-band masking measurements. No AI — pure DSP.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_mix_studio.analyzer import analyze_audio
from ai_mix_studio.audio import load_audio
from ai_mix_studio.session import analyze_session

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mix Studio — Online Analyzer</title>
<style>
  body { font-family: Segoe UI, sans-serif; background:#15171a; color:#e7e9ec; max-width:860px; margin:40px auto; padding:0 16px; }
  h1 { font-size:22px; } .sub { color:#9aa0a8; margin-top:-8px; }
  form { background:#1d2024; border:1px solid #30343a; border-radius:8px; padding:20px; margin:20px 0; }
  button { background:#2f6fed; color:#fff; border:0; padding:10px 18px; border-radius:5px; font-size:15px; cursor:pointer; }
  table { width:100%; border-collapse:collapse; margin-top:8px; }
  td { padding:6px 8px; border-bottom:1px solid #262a30; font-variant-numeric:tabular-nums; }
  td:first-child { color:#9aa0a8; width:45%; }
  h2 { font-size:15px; letter-spacing:1px; color:#7ea4f5; margin:22px 0 4px; }
  #out { white-space:pre-wrap; }
</style>
</head>
<body>
<h1>🎛️ Mix Studio — Online Mixing Analyzer</h1>
<p class="sub">Local DSP measurements only (no AI): dBTP · RMS · LUFS/LRA · stereo · correlation · masking</p>
<p style="color:#9aa0a8">Select one stem for track analysis, or many stems (25–50, all starting at 0:00) for full session analysis: bus mix + masking conflicts.</p>
<form id="f" enctype="multipart/form-data">
  <input type="file" name="file" accept=".wav,.flac,.aiff,.aif" multiple required>
  <button type="submit">Analyze</button>
</form>
<div id="out">Upload a stem to see its measurements…</div>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const out = document.getElementById('out');
  out.textContent = 'Analyzing…';
  const fd = new FormData(e.target);
  const multi = e.target.file.files.length > 1;
  if (multi) { fd.delete('file'); [...e.target.file.files].forEach(f => fd.append('files', f)); }
  const r = await fetch(multi ? '/analyze-session' : '/analyze', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { out.textContent = 'Error: ' + (j.error || r.statusText); return; }
  if (j.track_count) {
    let h = `<h2>SESSION — ${j.track_count} stems, ${j.duration_seconds.toFixed(1)} s</h2><table>`;
    if (j.bus) h += `<tr><td>BUS true peak</td><td>${j.bus.true_peak_dbtp.toFixed(2)} dBTP</td></tr><tr><td>BUS LUFS int</td><td>${j.bus.lufs_integrated.toFixed(2)}</td></tr><tr><td>BUS correlation</td><td>${j.bus.stereo_correlation.toFixed(3)}</td></tr>`;
    h += '</table><h2>PER-STEM</h2><table>';
    for (const [n, t] of Object.entries(j.tracks))
      h += `<tr><td>${n}</td><td>peak ${t.peak_dbfs.toFixed(1)} · dBTP ${t.true_peak_dbtp.toFixed(1)} · LUFS ${t.lufs_integrated.toFixed(1)} · corr ${t.stereo_correlation.toFixed(2)}</td></tr>`;
    h += '</table><h2>MASKING CONFLICTS (top 50)</h2><table>';
    for (const c of j.conflicts)
      h += `<tr><td>${c.track_a} vs ${c.track_b}</td><td>${c.center_hz.toFixed(0)} Hz · ${c.level_a_db.toFixed(1)} / ${c.level_b_db.toFixed(1)} dB</td></tr>`;
    out.innerHTML = h + '</table>'; return;
  }
  let h = `<h2>LEVEL / DYNAMICS</h2><table>
    <tr><td>Duration</td><td>${j.duration_seconds.toFixed(2)} s · ${j.sample_rate} Hz · ${j.channels} ch</td></tr>
    <tr><td>Peak</td><td>${j.peak_dbfs.toFixed(2)} dBFS</td></tr>
    <tr><td>True peak</td><td>${j.true_peak_dbtp.toFixed(2)} dBTP</td></tr>
    <tr><td>RMS</td><td>${j.rms_dbfs.toFixed(2)} dBFS</td></tr>
    <tr><td>Crest factor</td><td>${j.crest_factor_db.toFixed(2)} dB</td></tr></table>
  <h2>LOUDNESS (ITU-R BS.1770)</h2><table>
    <tr><td>LUFS integrated</td><td>${j.lufs_integrated.toFixed(2)} LUFS</td></tr>
    <tr><td>LUFS momentary max</td><td>${j.lufs_momentary_max.toFixed(2)} LUFS</td></tr>
    <tr><td>LRA</td><td>${j.lra.toFixed(2)} LU</td></tr></table>
  <h2>STEREO</h2><table>
    <tr><td>Correlation</td><td>${j.stereo_correlation.toFixed(3)}</td></tr>
    <tr><td>Mid RMS</td><td>${j.mid_rms_dbfs.toFixed(2)} dBFS</td></tr>
    <tr><td>Side RMS</td><td>${j.side_rms_dbfs.toFixed(2)} dBFS</td></tr>
    <tr><td>Width</td><td>${j.stereo_width_db.toFixed(2)} dB</td></tr></table>
  <h2>MASKING / SPECTRUM</h2><table>
    <tr><td>Spectral centroid</td><td>${j.spectral_centroid_hz.toFixed(0)} Hz</td></tr></table><table>`;
  for (const b of j.masking_bands)
    h += `<tr><td>${b.center_hz.toFixed(0)} Hz band</td><td>L ${b.left_db.toFixed(1)} · R ${b.right_db.toFixed(1)} · overlap ${b.overlap.toFixed(2)}</td></tr>`;
  out.innerHTML = h + '</table>';
};
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/analyze")
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="no file uploaded"), 400
    with tempfile.NamedTemporaryFile(suffix=Path(f.filename).suffix or ".wav", delete=False) as tmp:
        f.save(tmp.name)
        try:
            result = analyze_audio(load_audio(Path(tmp.name)))
        except Exception as exc:
            return jsonify(error=str(exc)), 400
    return jsonify(
        duration_seconds=result.duration_seconds,
        sample_rate=result.sample_rate,
        channels=result.channels,
        peak_dbfs=result.peak_dbfs,
        true_peak_dbtp=result.true_peak_dbtp,
        rms_dbfs=result.rms_dbfs,
        crest_factor_db=result.crest_factor_db,
        lufs_integrated=result.lufs_integrated,
        lufs_momentary_max=result.lufs_momentary_max,
        lra=result.lra,
        stereo_correlation=result.stereo_correlation,
        mid_rms_dbfs=result.mid_rms_dbfs,
        side_rms_dbfs=result.side_rms_dbfs,
        stereo_width_db=result.stereo_width_db,
        spectral_centroid_hz=result.spectral_centroid_hz,
        masking_bands=[asdict(b) for b in result.masking_bands],
    )


@app.post("/analyze-session")
def analyze_session_route():
    files = request.files.getlist("files")
    if not files or not any(f.filename for f in files):
        return jsonify(error="no files uploaded"), 400
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        stems = []
        for f in files:
            if not f.filename:
                continue
            dest = Path(td) / Path(f.filename).name
            f.save(dest)
            stems.append((dest.stem, dest))
        try:
            report = analyze_session(stems)
        except Exception as exc:
            return jsonify(error=str(exc)), 400
    return jsonify(
        sample_rate=report.sample_rate,
        duration_seconds=report.duration_seconds,
        track_count=report.track_count,
        summary=report.summary_lines(),
        bus=None if report.bus is None else {
            "peak_dbfs": report.bus.peak_dbfs,
            "true_peak_dbtp": report.bus.true_peak_dbtp,
            "rms_dbfs": report.bus.rms_dbfs,
            "lufs_integrated": report.bus.lufs_integrated,
            "lra": report.bus.lra,
            "stereo_correlation": report.bus.stereo_correlation,
        },
        tracks={name: {
            "peak_dbfs": r.peak_dbfs,
            "true_peak_dbtp": r.true_peak_dbtp,
            "rms_dbfs": r.rms_dbfs,
            "lufs_integrated": r.lufs_integrated,
            "stereo_correlation": r.stereo_correlation,
        } for name, r in report.tracks.items()},
        conflicts=[asdict(c) for c in report.conflicts[:50]],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
