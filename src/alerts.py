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


@lru_cache(maxsize=1)
def make_alarm_wav(duration=2.0, rate=22050):
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
