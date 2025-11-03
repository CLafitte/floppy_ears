#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <fftw3.h> // Using FFTW for FFT-based amplitude compensation

class FloppyEarsDSP {
public:
    FloppyEarsDSP(double sampleRate)
        : sr(sampleRate),
          transientBoost(0.5),
          expandEnabled(false),
          pitchShiftSemitones(0.0),
          envelope(0.0),
          maxEnv(1e-9)
    {
        initFrequencyShaping();
        initTwoBandExpansion();
    }

    void processBlock(float* buffer, int numSamples) {
        // For FFT-based amplitude compensation
        fftInput.resize(numSamples);
        fftOutput.resize(numSamples);

        for (int i = 0; i < numSamples; ++i) {
            float x = buffer[i];

            // -------------------- Transient Enhancement --------------------
            envelope = transientCoeff * envelope + (1.0f - transientCoeff) * std::abs(x);
            maxEnv = std::max(maxEnv, envelope);
            x = x * (1.0f + transientBoost * envelope / maxEnv);

            // -------------------- Optional Soft Upward Expansion --------------------
            if (expandEnabled) {
                x = applyTwoBandExpansion(x);
            }

            // -------------------- Frequency Shaping --------------------
            x = applyFIRFilter(x);

            buffer[i] = x;
            fftInput[i] = x; // Save for FFT amplitude compensation
        }

        // -------------------- Amplitude Compensation --------------------
        applyAmplitudeCompensation(buffer, numSamples);

        // -------------------- Soft Limiter --------------------
        for (int i = 0; i < numSamples; ++i) {
            buffer[i] = softLimit(buffer[i]);
        }

        // Optional pitch shift (not implemented here)
        if (pitchShiftSemitones != 0.0f) {
            // Implement real-time pitch shift
        }
    }

    // -------------------- Parameter Setters --------------------
    void setTransientBoost(float boost) { transientBoost = boost; }
    void enableExpansion(bool enabled) { expandEnabled = enabled; }
    void setPitchShift(float semitones) { pitchShiftSemitones = semitones; }

private:
    double sr;
    float transientBoost;
    bool expandEnabled;
    float pitchShiftSemitones;

    float envelope;
    float maxEnv;
    float transientCoeff = 0.9f;

    // FIR filter
    std::vector<float> firCoeffs;
    std::vector<float> firBuffer;
    int firIndex = 0;

    // Two-band expansion
    float hpPrevSample = 0.0f;
    float expandEnvelope = 0.0f;
    float expandMaxEnv = 1e-9;
    float expandCoeffAttack = 0.99f;
    float expandCoeffRelease = 0.999f;
    float expandRatio = 0.5f;
    float expandThresholdDB = -40.0f;

    // FFT for amplitude compensation
    std::vector<float> fftInput;
    std::vector<std::complex<float>> fftOutput;

    // -------------------- DSP Stage Implementations --------------------
    void initFrequencyShaping() {
        // Example: 101-tap FIR band-pass 4–16 kHz
        int numTaps = 101;
        firCoeffs.resize(numTaps);
        firBuffer.resize(numTaps, 0.0f);
        double low = 4000.0 / (sr/2);
        double high = 16000.0 / (sr/2);
        for (int i = 0; i < numTaps; ++i) {
            double n = i - (numTaps-1)/2.0;
            if (n == 0) firCoeffs[i] = high - low;
            else firCoeffs[i] = (sin(M_PI*high*n) - sin(M_PI*low*n)) / (M_PI*n);
            // Apply window (Hamming)
            firCoeffs[i] *= 0.54 - 0.46 * cos(2*M_PI*i/(numTaps-1));
        }
    }

    float applyFIRFilter(float x) {
        firBuffer[firIndex] = x;
        float y = 0.0f;
        int bufSize = firBuffer.size();
        for (int i = 0; i < bufSize; ++i) {
            int idx = (firIndex - i + bufSize) % bufSize;
            y += firBuffer[idx] * firCoeffs[i];
        }
        firIndex = (firIndex + 1) % bufSize;
        return y;
    }

    void initTwoBandExpansion() {
        // High-pass filter using simple difference equation (2nd-order)
        // placeholder coefficients
    }

    float applyTwoBandExpansion(float x) {
        // Simple high-pass placeholder
        float hp = x - hpPrevSample;
        hpPrevSample = x;

        // Exponential envelope
        float absHp = std::abs(hp);
        if (absHp > expandEnvelope)
            expandEnvelope = expandCoeffAttack * expandEnvelope + (1.0f - expandCoeffAttack) * absHp;
        else
            expandEnvelope = expandCoeffRelease * expandEnvelope + (1.0f - expandCoeffRelease) * absHp;
        expandMaxEnv = std::max(expandMaxEnv, expandEnvelope);

        // Gain in linear
        float envDB = 20.0f * log10(expandEnvelope + 1e-12f);
        float gainDB = std::max(0.0f, (expandThresholdDB - envDB) * expandRatio);
        float gain = pow(10.0f, gainDB/20.0f);

        return x + hp * gain;
    }

    void applyAmplitudeCompensation(float* buffer, int numSamples) {
        // FFT placeholder: boost 4-8kHz, 8-12kHz, 12-16kHz
        // Could implement with FFTW real-to-complex
    }

    float softLimit(float x, float threshold = 0.9f) {
        return std::tanh(x / threshold) * threshold;
    }
};
