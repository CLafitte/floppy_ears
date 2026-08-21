"""
Regression tests for floppy_ears.py's DSP chain.

Covers two crash bugs found in v0.2.1-beta:
  1. dsp_chain() called amplitude_compensation() without `sr`, so every
     invocation of dsp_chain (and therefore process_audio / preview_realtime)
     raised TypeError.
  2. optional_pitch_shift() called librosa.effects.pitch_shift() with `sr`
     as a positional argument. librosa has required `sr` as keyword-only
     since 0.9+ (re-enforced in 0.10+), so --pitch crashed on any current
     librosa install.

Run with: pytest tests/test_dsp_chain.py -v
"""
import sys
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, "..")  # adjust if your test runner's rootdir differs
import floppy_ears as fe


SR = 44100


@pytest.fixture
def audio():
    """1 second of deterministic noise, mono, float64."""
    rng = np.random.default_rng(seed=0)
    return rng.uniform(-0.5, 0.5, size=SR).astype(np.float64)


# -------------------- Bug 1: amplitude_compensation missing sr -------------------- #

def test_dsp_chain_default_does_not_raise(audio):
    """Regression test for TypeError: amplitude_compensation() missing 1
    required positional argument: 'sr'."""
    out = fe.dsp_chain(audio, SR)
    assert out.shape == audio.shape
    assert out.dtype == np.float64


def test_amplitude_compensation_requires_sr_and_returns_same_length(audio):
    out = fe.amplitude_compensation(audio, SR)
    assert out.shape == audio.shape


# -------------------- Bug 2: librosa.effects.pitch_shift positional sr -------------------- #

def test_optional_pitch_shift_calls_librosa_with_sr_as_keyword(audio):
    """Regression test for TypeError: pitch_shift() takes 1 positional
    argument but 2 positional arguments were given (librosa >=0.9)."""
    with patch("floppy_ears.librosa.effects.pitch_shift") as mock_shift:
        mock_shift.return_value = audio
        fe.optional_pitch_shift(audio, SR, n_steps=2)

        assert mock_shift.called
        args, kwargs = mock_shift.call_args
        assert len(args) == 1, (
            f"pitch_shift called with {len(args)} positional args; "
            "sr must be passed as a keyword argument"
        )
        assert kwargs.get("sr") == SR
        assert kwargs.get("n_steps") == 2


def test_dsp_chain_with_pitch_does_not_raise(audio):
    with patch("floppy_ears.librosa.effects.pitch_shift") as mock_shift:
        mock_shift.return_value = audio
        out = fe.dsp_chain(audio, SR, apply_pitch=True)
        assert out.shape == audio.shape


# -------------------- General sanity / flag combinations -------------------- #

@pytest.mark.parametrize("apply_pitch,apply_expand", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
def test_dsp_chain_all_flag_combinations(audio, apply_pitch, apply_expand):
    with patch("floppy_ears.librosa.effects.pitch_shift") as mock_shift:
        mock_shift.return_value = audio
        out = fe.dsp_chain(audio, SR, apply_pitch=apply_pitch, apply_expand=apply_expand)
        assert out.shape == audio.shape
        assert not np.isnan(out).any(), "dsp_chain produced NaNs"
        assert not np.isinf(out).any(), "dsp_chain produced Infs"


def test_dsp_chain_output_is_bounded_by_soft_limit(audio):
    """soft_limit uses tanh, so output should never exceed the threshold."""
    with patch("floppy_ears.librosa.effects.pitch_shift") as mock_shift:
        mock_shift.return_value = audio
        out = fe.dsp_chain(audio, SR)
        assert np.max(np.abs(out)) <= 0.9 + 1e-9
