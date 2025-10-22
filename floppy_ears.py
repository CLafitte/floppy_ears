#!/usr/bin/env python3
"""
floppy_ears.py
MVP: Transform a WAV file to approximate canine hearing perception. 
Note that most consumer microphones will not capture the entire canine hearing range. 
This script performs basic filtering and DSP to approximate canine hearing perception for the human range.
Precision: Uses float64 throughout for accurate spectral and dynamic transformations.
Dependencies: numpy, scipy, soundfile, librosa (optional)
"""

import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
from scipy.fft import rfft, irfft
import argparse
import librosa

def load_audio(filename):
    audio, sr = sf.read(filename, dtype='float64')  # ensure float64 precision
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # convert stereo to mono
    return audio, sr

def save_audio(filename, audio, sr):
    # Convert to float64-safe range before saving
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    sf.write(filename, audio.astype(np.float64), sr)

def frequency_shaping(audio, sr, low=4000, high=16000, numtaps=101):
    """Band-pass FIR filter emphasizing canine-sensitive frequencies."""
    b = firwin(numtaps, [low/(sr/2), high/(sr/2)], pass_zero=False)
    return lfilter(b, [1.0], audio).astype(np.float64)

def amplitude_compensation(audio, sr):
    """Apply simple frequency-dependent gain based on canine audiogram approximation."""
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    spectrum = rfft(audio)
    gain = np.ones_like(spectrum, dtype=np.float64)
    # Adjustable gain bands
    gain[(freqs >= 4000) & (freqs < 8000)] *= 1.2
    gain[(freqs >= 8000) & (freqs < 12000)] *= 1.5
    gain[(freqs >= 12000) & (freqs <= 16000)] *= 1.3
    return irfft(spectrum * gain).astype(np.float64)

def transient_enhancement(audio, window_size=1024, boost=0.5):
    """Enhance transients to mimic dogs' acute temporal sensitivity."""
    envelope = np.convolve(np.abs(audio), np.ones(window_size)/window_size, mode='same')
    audio = audio * (1 + boost * (envelope / np.max(envelope)))
    return audio.astype(np.float64)

def optional_pitch_shift(audio, sr, n_steps=2):
    """Optional: Shift pitch to emphasize high frequencies."""
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps).astype(np.float64)

def process_audio(input_file, output_file, apply_pitch=False):
    audio, sr = load_audio(input_file)
    print(f"Loaded '{input_file}' with {sr} Hz sample rate.")

    # Enforce float64 precision for all transformations
    audio = audio.astype(np.float64)

    audio = frequency_shaping(audio, sr)
    audio = amplitude_compensation(audio, sr)
    audio = transient_enhancement(audio)

    if apply_pitch:
        audio = optional_pitch_shift(audio, sr)

    save_audio(output_file, audio, sr)
    print(f"Saved transformed audio to '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(description="Floppy Ears - Transform WAV files to approximate dog hearing")
    parser.add_argument("input_file", help="Path to input WAV file")
    parser.add_argument("output_file", help="Path to output WAV file")
    parser.add_argument("--pitch", action="store_true", help="Apply optional high-frequency pitch shift")
    args = parser.parse_args()

    process_audio(args.input_file, args.output_file, apply_pitch=args.pitch)

if __name__ == "__main__":
    main()
