#!/usr/bin/env python3
"""
floppy_ears.py v0.2
MVP: Transform a WAV file to approximate canine hearing perception.
This version adds:
- Exponential envelope soft upward expansion (optional)
- Modular pipeline for clean integration
Precision: float64 throughout
Dependencies: numpy, scipy, soundfile, librosa (optional)
"""

import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter, butter
from scipy.fft import rfft, irfft
import argparse
import librosa

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

def transient_enhancement(audio, window_size=1024, boost=0.5):
    """Enhance transients to mimic dogs' acute temporal sensitivity."""
    envelope = np.convolve(np.abs(audio), np.ones(window_size)/window_size, mode='same')
    audio = audio * (1 + boost * (envelope / np.max(envelope)))
    return audio.astype(np.float64)

# -------------------- New Soft Upward Expansion -------------------- #

def soft_expand(audio, sr, threshold_db=-40, ratio=0.5, attack_ms=10, release_ms=100):
    """Exponential envelope follower for efficient soft upward expansion."""
    eps = 1e-12
    attack_coeff = np.exp(-1.0 / (attack_ms * 0.001 * sr))
    release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr))

    env = 0.0
    out = np.zeros_like(audio, dtype=np.float64)

    for i, x in enumerate(audio):
        x_abs = abs(x)
        # Exponential envelope
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

# -------------------- Pipeline -------------------- #

def process_audio(input_file, output_file, apply_pitch=False, apply_expand=False):
    audio, sr = load_audio(input_file)
    print(f"Loaded '{input_file}' with {sr} Hz sample rate.")

    audio = audio.astype(np.float64)
    audio = frequency_shaping(audio, sr)
    audio = amplitude_compensation(audio, sr)
    
    if apply_expand:
        audio = two_band_expand(audio, sr)  # new exponential envelope smoothing expansion

    audio = transient_enhancement(audio)
    
    if apply_pitch:
        audio = optional_pitch_shift(audio, sr)

    audio = soft_limit(audio)
    save_audio(output_file, audio, sr)
    print(f"Saved transformed audio to '{output_file}'.")

# -------------------- CLI -------------------- #

def main():
    parser = argparse.ArgumentParser(description="Floppy Ears - Transform WAV files to approximate dog hearing")
    parser.add_argument("input_file", help="Path to input WAV file")
    parser.add_argument("output_file", help="Path to output WAV file")
    parser.add_argument("--pitch", action="store_true", help="Apply optional high-frequency pitch shift")
    parser.add_argument("--expand", action="store_true", help="Apply soft upward expansion (exponential envelope smoothing)")
    args = parser.parse_args()

    process_audio(args.input_file, args.output_file, apply_pitch=args.pitch, apply_expand=args.expand)

if __name__ == "__main__":
    main()
