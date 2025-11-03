#!/usr/bin/env python3
"""
floppy_ears.py v0.2.1-beta
MVP: Transform a WAV file to approximate canine hearing perception.
This version adds:
- Real-time preview mode (--preview) using the same DSP chain
Precision: float64 throughout
Dependencies: numpy, scipy, soundfile, librosa, sounddevice
"""

import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter, butter
from scipy.fft import rfft, irfft
import argparse
import librosa
import sounddevice as sd  # 🔹 added for preview mode

# -------------------- Core Utilities -------------------- #

def load_audio(filename):
    audio, sr = sf.read(filename, dtype='float64')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

def save_audio(filename, audio, sr):
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    sf.write(filename, audio.astype(np.float64), sr)

# -------------------- DSP Functions -------------------- #

def frequency_shaping(audio, sr, low=4000, high=16000, numtaps=101):
    """Band-pass FIR filter emphasizing canine-sensitive frequencies."""
    b = firwin(numtaps, [low/(sr/2), high/(sr/2)], pass_zero=False)
    return lfilter(b, [1.0], audio).astype(np.float64)

def amplitude_compensation(audio, sr):
    """Simple spectral gain for high frequencies."""
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    spectrum = rfft(audio)
    gain = np.ones_like(spectrum, dtype=np.float64)
    gain[(freqs >= 4000) & (freqs < 8000)] *= 1.2
    gain[(freqs >= 8000) & (freqs < 12000)] *= 1.5
    gain[(freqs >= 12000) & (freqs <= 16000)] *= 1.3
    return irfft(spectrum * gain).astype(np.float64)

def transient_enhancement(audio, sr, boost=0.5, time_constant_ms=10):
    """Enhance transients using an exponential envelope follower."""
    coeff = np.exp(-1.0 / (time_constant_ms * 0.001 * sr))
    env = 0.0
    out = np.zeros_like(audio)
    max_env = 1e-9
    for i, x in enumerate(audio):
        env = coeff * env + (1 - coeff) * abs(x)
        max_env = max(max_env, env)
        out[i] = x * (1 + boost * env / max_env)
    return out.astype(np.float64)

# -------------------- Soft Upward Expansion -------------------- #

def soft_expand(audio, sr, threshold_db=-40, ratio=0.5, attack_ms=10, release_ms=100):
    """Exponential envelope follower for efficient soft upward expansion."""
    eps = 1e-12
    attack_coeff = np.exp(-1.0 / (attack_ms * 0.001 * sr))
    release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr))
    env = 0.0
    out = np.zeros_like(audio, dtype=np.float64)

    for i, x in enumerate(audio):
        x_abs = abs(x)
        if x_abs > env:
            env = attack_coeff * env + (1 - attack_coeff) * x_abs
        else:
            env = release_coeff * env + (1 - release_coeff) * x_abs

        env_db = 20 * np.log10(env + eps)
        gain_db = max(0.0, (threshold_db - env_db) * ratio)
        gain = 10 ** (gain_db / 20.0)
        out[i] = x * gain

    return out

def two_band_expand(audio, sr, threshold_db=-40, ratio=0.5):
    """Apply upward expansion only to high-frequency band (>4 kHz)."""
    b, a = butter(2, 4000/(sr/2), btype='high')
    high = lfilter(b, a, audio)
    low = audio - high
    high_expanded = soft_expand(high, sr, threshold_db=threshold_db, ratio=ratio)
    return low + high_expanded

def optional_pitch_shift(audio, sr, n_steps=2):
    """Optional: Shift pitch to emphasize high frequencies."""
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps).astype(np.float64)

def soft_limit(audio, threshold=0.9):
    """Prevent clipping gracefully."""
    return np.tanh(audio / threshold) * threshold

# -------------------- DSP Pipeline -------------------- #

def dsp_chain(audio, sr, apply_pitch=False, apply_expand=False):
    """Reusable DSP chain used by both file and real-time modes."""
    audio = frequency_shaping(audio, sr)
    audio = amplitude_compensation(audio)
    if apply_expand:
        audio = two_band_expand(audio, sr)
    audio = transient_enhancement(audio, sr)
    if apply_pitch:
        audio = optional_pitch_shift(audio, sr)
    audio = soft_limit(audio)
    return audio

def process_audio(input_file, output_file, apply_pitch=False, apply_expand=False):
    audio, sr = load_audio(input_file)
    print(f"Loaded '{input_file}' with {sr} Hz sample rate.")
    processed = dsp_chain(audio, sr, apply_pitch, apply_expand)
    save_audio(output_file, processed, sr)
    print(f"Saved transformed audio to '{output_file}'.")

# -------------------- Real-Time Preview -------------------- #

def preview_realtime(sr, apply_pitch=False, apply_expand=False):
    """Low-latency real-time preview using sounddevice."""
    print("[INFO] Starting real-time preview (Ctrl+C or Enter to stop)...")

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status, flush=True)
        processed = dsp_chain(indata[:, 0], sr, apply_pitch, apply_expand)
        outdata[:] = np.expand_dims(processed, axis=1)

    with sd.Stream(channels=1, samplerate=sr, blocksize=1024,
                   callback=callback, dtype='float64'):
        input()  # waits until user presses Enter

    print("[INFO] Preview stopped.")

# -------------------- CLI -------------------- #

def main():
    parser = argparse.ArgumentParser(description="Floppy Ears - Transform WAV files or preview in real-time")
    parser.add_argument("input_file", nargs="?", help="Path to input WAV file (omit for --preview)")
    parser.add_argument("output_file", nargs="?", help="Path to output WAV file")
    parser.add_argument("--pitch", action="store_true", help="Apply optional high-frequency pitch shift")
    parser.add_argument("--expand", action="store_true", help="Apply soft upward expansion (exponential envelope smoothing)")
    parser.add_argument("--preview", action="store_true", help="Enable real-time playback preview of processed signal")
    args = parser.parse_args()

    if args.preview:
        # Default to 44.1 kHz if no file provided
        sr = 44100
        preview_realtime(sr, apply_pitch=args.pitch, apply_expand=args.expand)
    else:
        if not args.input_file or not args.output_file:
            parser.error("input_file and output_file are required unless using --preview")
        process_audio(args.input_file, args.output_file, apply_pitch=args.pitch, apply_expand=args.expand)

if __name__ == "__main__":
    main()
