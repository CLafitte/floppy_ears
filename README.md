# floppy_ears

Re-renders a recording to approximate how its audibility would differ for a dog compared with a human listener. The filter's gain at each frequency is the human hearing threshold minus the dog threshold: boosted where dogs are more sensitive, cut where they are less.

**Status:** beta (v0.3.0). This is a threshold-difference model, not a simulation of canine perception.

### Install

```
pip install numpy scipy soundfile
pip install sounddevice      # only for --preview
```

### Usage

```
python floppy_ears.py input.wav output.wav [--match-rms]
python floppy_ears.py --preview        # mic -> headphones; use headphones, start quiet
python floppy_ears.py --show-curve     # print the filter gain and exit
python floppy_ears.py --sources        # print data sources and exit
```

Options:

- `--max-boost-db`, `--max-cut-db` (default 30 each): gain limits in dB.
- `--taps` (default 2049): FIR length. More taps give finer low-frequency detail and more latency.
- `--match-rms` (default off): match output RMS to the input so A/B comparison isn't decided by loudness.
- `--subtype` (default `PCM_24`): output format, one of `PCM_16`, `PCM_24`, `FLOAT`.
- `--sr` (default: device rate), `--blocksize` (default 1024): preview settings.

Channels are preserved. Output is scaled down linearly only if needed to avoid clipping.

### How it works

`gain_dB(f) = T_human(f) - T_dog(f)`, clipped to the gain limits, realised as one linear-phase FIR. At 44.1 kHz the gain is roughly -24 dB at 63 Hz, -9 dB at 4 kHz, +8 dB at 8 kHz, and reaches the +30 dB cap at 16 kHz.

### Sources

- **Human:** ISO 389-7:2019, Table 1 (free-field, frontal incidence, otologically normal 18-25 year-olds, 20 Hz-18 kHz).
- **Dog:** Heffner, H. E. (1983), *Behavioral Neuroscience* 97(2), 310-318, doi:10.1037/0735-7044.97.2.310 (four dogs). The values used were taken from the table of Heffner's thresholds in *Vet. Sci.* 2024, 11(2), 67, doi:10.3390/vetsci11020067.

**Dog data caveat:** only these points are numeric; everything between is interpolated (linear dB vs log frequency).

- 67 Hz, 60 dB SPL: low edge of the 60 dB range, as commonly cited.
- 500 Hz, 20 dB SPL: measured, mean of four dogs.
- 4 kHz, 4 dB SPL: measured, mean of four dogs.
- 16 kHz, 6 dB SPL: measured, mean of four dogs.
- 45 kHz, 60 dB SPL: high edge of the 60 dB range, as commonly cited.

The best-sensitivity region near 8 kHz is therefore probably understated. To improve the curve, digitise Heffner's Figure 2 (or Fay 1988) and replace `DOG_ANCHORS` in the script.

### Limitations

- Models threshold sensitivity only: not playback level, loudness growth, masking, or localisation.
- A file can't contain frequencies above half its sample rate; a 44.1 kHz file misses most of the dog's range up to ~45 kHz.
- Edge assumptions, not measurements: gain is held below 67 Hz; from 18 to 45 kHz the human is treated as deaf (gain at the boost cap); above 45 kHz the dog is treated as deaf (gain at the cut cap).
- Both curves are small-sample averages from different studies; individual thresholds vary.
