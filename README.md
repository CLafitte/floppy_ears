**PRE-RELEASE/EXPERIMENTAL** 

This is a pre-release repository for floppy_ears, a Python tool that applies DSP to a WAV file
to approximate canine perception within the human-audible range. 

At present, this tool consists of a DSP core, written in Python and ported to C++. 
The pipeline assumes input from a consumer microphone capturing the ~20Hz - 20kHz range; 
these mics don't capture the full canine hearing spectrum (~40 Hz to 65,000 Hz). To 
accommodate, this pipeline aims to mimic dogs' sensitivity to high frequencies and 
low-level signals within human hearing range. The pipeline emphasizes high-frequency 
content (~4–16 kHz) and boosts quiet sounds using soft upward expansion to mimic 
dogs’ greater sensitivity to high frequencies and low-level signals. It also enhances 
transients and optionally shifts pitch slightly upward, reflecting dogs’ faster temporal 
resolution and extended hearing range within the limits of typical microphones.

For more specific notes on the decisions behind each DSP stage, see: 
`dsp_pipeline_instructions.txt`

## Minimal Python Usage

```bash
# Process a WAV file and save the output
python floppy_ears.py input.wav output.wav

# Optional: apply high-frequency pitch shift
python floppy_ears.py input.wav output.wav --pitch

# Optional: apply soft upward expansion
python floppy_ears.py input.wav output.wav --expand

# Optional: combine both
python floppy_ears.py input.wav output.wav --pitch --expand
