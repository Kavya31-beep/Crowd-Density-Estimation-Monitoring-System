"""Step 4: alarm sound and event logging (timestamp + camera ID)."""
import csv
import datetime
import io
import math
import struct
import wave
from functools import lru_cache

import pandas as pd

from . import config

COLUMNS = ["timestamp", "camera_id", "source", "mode", "total_count", "overloaded_zones", "max_density_per_m2"]
ALARM_SECONDS = 60  # how long the siren and the full-screen warning last

@lru_cache(maxsize=1)
def make_alarm_wav(duration=ALARM_SECONDS,rate=22050):
    """Generates a two-tone siren as WAV bytes (no audio file needed)."""
    frames = bytearray()
    for i in range(int(duration * rate)):
        t = i / rate
        f = 880 if int(t * 4) % 2 == 0 else 660
        frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * f * t)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def zone_names(flags):
    rows, cols = flags.shape
    return [f"R{r + 1}C{c + 1}" for r in range(rows) for c in range(cols) if flags[r, c]]


def log_event(camera_id, source, mode, total, flags, dens):
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not config.LOG_FILE.exists()
    with open(config.LOG_FILE, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(COLUMNS)
        wr.writerow([datetime.datetime.now().isoformat(timespec="seconds"), camera_id, source, mode,
                     int(round(total)), ";".join(zone_names(flags)), round(float(dens.max()), 2)])


def read_log():
    if not config.LOG_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(config.LOG_FILE)

import base64


def alarm_component_html(zones, count, seconds=ALARM_SECONDS):
    """Large flashing alert banner at the top of the dashboard + looping siren.
    Stays until the operator presses STOP (or `seconds` pass)."""
    wav_b64 = base64.b64encode(make_alarm_wav(2)).decode()  # 2 s clip, looped by the browser
    html = """
<script>
(function () {
  const win = window.parent, doc = win.document;
  const old = doc.getElementById('crowd-alarm-root');
  if (old) old.remove();
  if (win.__crowdAudio) { try { win.__crowdAudio.pause(); } catch (e) {} }

  const root = doc.createElement('div');
  root.id = 'crowd-alarm-root';
  root.innerHTML = `
    <style>
      @keyframes caflash { 0%,100% {background: rgb(150,0,0);} 50% {background: rgb(255,0,0);} }
      #crowd-alarm-root { position: sticky; top: 3.5rem; z-index: 999; margin-bottom: 1rem; }
      #crowd-alarm-root .ca-wrap { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
        gap: 1rem; padding: 1rem 1.5rem; border-radius: 0.8rem; border: 4px solid #000; color: #fff;
        font-family: Arial, Helvetica, sans-serif; animation: caflash 0.8s ease-in-out infinite; }
      #crowd-alarm-root .big { font-size: clamp(1.8rem, 4.5vw, 4rem); font-weight: 900; line-height: 1.1; }
      #crowd-alarm-root .info { font-size: clamp(1.1rem, 2.5vw, 2.2rem); font-weight: 800; margin-top: 0.4rem; }
      #crowd-alarm-root button { padding: 0.8rem 1.8rem; font-size: clamp(1.1rem, 2vw, 1.8rem); font-weight: 900;
        color: #b00000; background: #fff; border: 4px solid #000; border-radius: 0.8rem; cursor: pointer; }
    </style>
    <div class="ca-wrap">
      <div>
        <div class="big">&#9888; DANGER: OVERCROWDING &#9888;</div>
        <div class="info">ZONES: __ZONES__ &nbsp;|&nbsp; ESTIMATED PEOPLE: __COUNT__</div>
      </div>
      <button id="ca-stop">STOP ALARM</button>
    </div>`;

  const host = doc.querySelector('[data-testid="stMainBlockContainer"]')
            || doc.querySelector('.block-container') || doc.body;
  host.insertBefore(root, host.firstChild);

  const audio = new win.Audio('data:audio/wav;base64,__WAV__');
  audio.loop = true;
  win.__crowdAudio = audio;
  audio.play().catch(function () {});

  let timer = null;
  function stopAlarm() { audio.pause(); root.remove(); if (timer) clearTimeout(timer); }
  root.querySelector('#ca-stop').onclick = stopAlarm;
  timer = setTimeout(stopAlarm, __MS__);
})();
</script>
"""
    return (html.replace("__ZONES__", ", ".join(zones))
                .replace("__COUNT__", str(int(round(count))))
                .replace("__WAV__", wav_b64)
                .replace("__MS__", str(int(seconds * 1000))))
