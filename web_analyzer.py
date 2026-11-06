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
<form id="f" enctype="multipart/form-data">
  <input type="file" name="file" accept=".wav,.flac,.aiff,.aif" required>
  <button type="submit">Analyze</button>
</form>
<div id="out">Upload a stem to see its measurements…</div>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const out = document.getElementById('out');
  out.textContent = 'Analyzing…';
  const fd = new FormData(e.target);
  const r = await fetch('/analyze', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { out.textContent = 'Error: ' + (j.error || r.statusText); return; }
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
