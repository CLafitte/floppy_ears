#!/usr/bin/env python3
"""floppy_ears.py v0.3.0-beta - human-vs-dog relative audibility filter.

Sources: ISO 389-7:2019 Table 1 (human); Heffner (1983) Behav Neurosci 97(2):310-318 (dog).
Run --sources for details. See README for model, assumptions and limitations.
"""

import argparse
import sys

import numpy as np
from scipy.signal import firwin2, freqz, lfilter, oaconvolve

# ISO 389-7:2019 Table 1, free-field: (Hz, dB SPL)
HUMAN_FREE_FIELD = np.array([
    (20, 78.1), (25, 68.7), (31.5, 59.5), (40, 51.1), (50, 44.0),
    (63, 37.5), (80, 31.5), (100, 26.5), (125, 22.1), (160, 17.9),
    (200, 14.4), (250, 11.4), (315, 8.6), (400, 6.2), (500, 4.4),
    (630, 3.0), (750, 2.4), (800, 2.2), (1000, 2.4), (1250, 3.5),
    (1500, 2.4), (1600, 1.7), (2000, -1.3), (2500, -4.2), (3000, -5.8),
    (3150, -6.0), (4000, -5.4), (5000, -1.5), (6000, 4.3), (6300, 6.0),
    (8000, 12.6), (9000, 13.9), (10000, 13.9), (11200, 13.0),
    (12500, 12.3), (14000, 18.4), (16000, 40.2), (18000, 70.4),
])

# Dog anchors: (Hz, dB SPL, provenance). Points between anchors are interpolated.
DOG_ANCHORS = [
    (67.0, 60.0, "low edge of 60 dB range (Heffner 1983, as commonly cited)"),
    (500.0, 20.0, "measured, mean of 4 dogs (Heffner 1983)"),
    (4000.0, 4.0, "measured, mean of 4 dogs (Heffner 1983)"),
    (16000.0, 6.0, "measured, mean of 4 dogs (Heffner 1983)"),
    (45000.0, 60.0, "high edge of 60 dB range (Heffner 1983, as commonly cited)"),
]

DOG_LOW_LIMIT = DOG_ANCHORS[0][0]
DOG_HIGH_LIMIT = DOG_ANCHORS[-1][0]
HUMAN_HIGH_LIMIT = float(HUMAN_FREE_FIELD[-1, 0])


def _interp_log(f, table_f, table_db):
    return np.interp(np.log2(f), np.log2(table_f), table_db)


def human_threshold_db(f):
    return _interp_log(np.asarray(f, float), HUMAN_FREE_FIELD[:, 0], HUMAN_FREE_FIELD[:, 1])


def dog_threshold_db(f):
    tf = np.array([a[0] for a in DOG_ANCHORS])
    tdb = np.array([a[1] for a in DOG_ANCHORS])
    return _interp_log(np.asarray(f, float), tf, tdb)


def relative_gain_db(f, max_boost_db=30.0, max_cut_db=30.0):
    """Human threshold minus dog threshold (dB), clipped; edge behavior is assumed."""
    f = np.maximum(np.asarray(f, float), 1.0)
    f_core = np.clip(f, DOG_LOW_LIMIT, HUMAN_HIGH_LIMIT)
    g = human_threshold_db(f_core) - dog_threshold_db(f_core)
    g = np.clip(g, -max_cut_db, max_boost_db)
    g = np.where(f > HUMAN_HIGH_LIMIT, max_boost_db, g)
    g = np.where(f > DOG_HIGH_LIMIT, -max_cut_db, g)
    return g


def design_filter(sr, numtaps=2049, max_boost_db=30.0, max_cut_db=30.0):
    if numtaps % 2 == 0:
        numtaps += 1
    nyq = sr / 2.0
    if nyq <= 20.0:
        raise ValueError(f"sample rate {sr} Hz is too low")
    grid = np.concatenate(([0.0], np.geomspace(10.0, nyq, 800)))
    grid[-1] = nyq
    gdb = relative_gain_db(grid, max_boost_db, max_cut_db)
    gdb[0] = gdb[1]
    return firwin2(numtaps, grid / nyq, 10.0 ** (gdb / 20.0))


def curve_report(h, sr, max_boost_db=30.0, max_cut_db=30.0):
    freqs = [f for f in (63, 125, 250, 500, 1000, 2000, 4000, 6300, 8000,
                         10000, 12500, 14000, 16000, 18000) if f < sr / 2]
    _, H = freqz(h, worN=np.array(freqs, float), fs=sr)
    real = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    tgt = relative_gain_db(freqs, max_boost_db, max_cut_db)
    lines = [" Hz    human  dog    target  realised   (dB)"]
    for f, r, t in zip(freqs, real, tgt):
        lines.append(f"{f:>6} {float(human_threshold_db(f)):>6.1f} "
                     f"{float(dog_threshold_db(f)):>6.1f} {t:>7.1f} {r:>8.1f}")
    return "\n".join(lines)


def apply_filter(audio, h):
    if audio.ndim == 1:
        return oaconvolve(audio, h, mode="same")
    return oaconvolve(audio, h[:, None], mode="same", axes=0)


def finalize(processed, original, match_rms=False, ceiling=0.98):
    """Optional RMS match, then linear scale-down only if needed to avoid clipping."""
    notes = []
    out = processed
    if match_rms:
        rms_o = np.sqrt(np.mean(original ** 2))
        rms_p = np.sqrt(np.mean(out ** 2))
        if rms_p > 1e-12 and rms_o > 1e-12:
            k = rms_o / rms_p
            out = out * k
            notes.append(f"RMS matched to input (gain {20 * np.log10(k):+.1f} dB)")
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > ceiling:
        k = ceiling / peak
        out = out * k
        notes.append(f"scaled by {20 * np.log10(k):.1f} dB to avoid clipping"
                     + (" (RMS no longer matched)" if match_rms else ""))
    return out, notes


def load_audio(filename):
    import soundfile as sf
    return sf.read(filename, dtype="float64")


def save_audio(filename, audio, sr, subtype="PCM_24"):
    import soundfile as sf
    sf.write(filename, audio, sr, subtype=subtype)


def process_audio(input_file, output_file, taps, max_boost, max_cut,
                  match_rms, subtype):
    audio, sr = load_audio(input_file)
    if audio.size == 0:
        raise SystemExit("input file contains no audio")
    print(f"Loaded '{input_file}': {sr} Hz, {audio.shape[0] / sr:.2f} s")
    if sr < 90000:
        print(f"[note] Nyquist is {sr / 2000:.1f} kHz; content above that "
              f"(dog hearing extends to ~45 kHz) cannot be present in this file.")
    h = design_filter(sr, taps, max_boost, max_cut)
    processed, notes = finalize(apply_filter(audio, h), audio, match_rms)
    for n in notes:
        print(f"[note] {n}")
    save_audio(output_file, processed, sr, subtype)
    print(f"Saved '{output_file}' ({subtype}).")


def preview_realtime(sr, h, blocksize=1024, device=None):
    import sounddevice as sd

    zi = np.zeros(len(h) - 1)

    def callback(indata, outdata, frames, time_info, status):
        nonlocal zi
        if status:
            print(status, file=sys.stderr, flush=True)
        y, zi = lfilter(h, [1.0], indata[:, 0].astype(np.float64), zi=zi)
        outdata[:, 0] = np.clip(y, -1.0, 1.0)

    latency_ms = 1000.0 * ((len(h) - 1) / 2 + blocksize) / sr
    print(f"[INFO] Preview at {sr} Hz, ~{latency_ms:.0f} ms latency. "
          f"Use headphones and start with the volume low. Press Enter to stop.")
    with sd.Stream(samplerate=sr, blocksize=blocksize, channels=1,
                   dtype="float32", callback=callback, device=device):
        input()


def print_sources():
    print("Human: ISO 389-7:2019 Table 1 (free-field, frontal incidence).")
    print("Dog:   Heffner (1983) Behav Neurosci 97(2):310-318; anchors:")
    for f, db, why in DOG_ANCHORS:
        print(f"   {f:>8.0f} Hz  {db:>5.1f} dB SPL  {why}")
    print("All other dog points are interpolated (linear dB vs log-frequency).")


def main():
    p = argparse.ArgumentParser(description="Floppy Ears - human-vs-dog audibility filter")
    p.add_argument("input_file", nargs="?")
    p.add_argument("output_file", nargs="?")
    p.add_argument("--preview", action="store_true", help="real-time mic preview")
    p.add_argument("--sr", type=int, default=None, help="preview sample rate")
    p.add_argument("--blocksize", type=int, default=1024)
    p.add_argument("--taps", type=int, default=2049, help="FIR length (odd)")
    p.add_argument("--max-boost-db", type=float, default=30.0)
    p.add_argument("--max-cut-db", type=float, default=30.0)
    p.add_argument("--match-rms", action="store_true", help="match output RMS to input")
    p.add_argument("--subtype", choices=["PCM_16", "PCM_24", "FLOAT"], default="PCM_24")
    p.add_argument("--show-curve", action="store_true", help="print filter gain and exit")
    p.add_argument("--sources", action="store_true", help="print data sources and exit")
    args = p.parse_args()

    if args.sources:
        print_sources()
        return

    if args.show_curve or args.preview:
        sr = args.sr
        if sr is None:
            if args.preview:
                import sounddevice as sd
                sr = int(sd.query_devices(kind="input")["default_samplerate"])
            else:
                sr = 44100
        h = design_filter(sr, args.taps, args.max_boost_db, args.max_cut_db)
        if args.show_curve:
            print(f"sample rate {sr} Hz, {len(h)} taps")
            print(curve_report(h, sr, args.max_boost_db, args.max_cut_db))
        else:
            preview_realtime(sr, h, args.blocksize)
        return

    if not args.input_file or not args.output_file:
        p.error("input_file and output_file are required unless using --preview")
    process_audio(args.input_file, args.output_file, args.taps,
                  args.max_boost_db, args.max_cut_db, args.match_rms, args.subtype)


if __name__ == "__main__":
    main()
