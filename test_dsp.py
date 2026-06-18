"""
test_dsp.py
===========

Proof that the DSP core is correct, with no microphone required. We synthesize
signals whose answer we already know (pure sine waves at chosen frequencies) and
check that the analyzer finds the right frequency and the right amplitude.

Run it with:  python test_dsp.py
"""

import numpy as np
from dsp_core import compute_spectrum, make_window, magnitude_to_db


def _peak_freq(freqs, magnitude):
    """Return the frequency with the largest magnitude."""
    return freqs[int(np.argmax(magnitude))]


def test_peak_on_bin():
    """A 1000 Hz sine sampled so it lands exactly on an FFT bin.

    With no spectral leakage, the analyzer should report the peak at exactly
    1000 Hz and recover the sine's amplitude (0.5) almost perfectly.
    """
    sample_rate = 8000
    n = 8000               # 1 second -> bin spacing is exactly 1 Hz
    freq = 1000.0
    amp = 0.5

    t = np.arange(n) / sample_rate
    signal = amp * np.sin(2 * np.pi * freq * t)

    freqs, magnitude = compute_spectrum(signal, sample_rate, make_window(n))
    peak = _peak_freq(freqs, magnitude)

    assert abs(peak - freq) < 1.0, f"peak at {peak} Hz, expected {freq} Hz"
    assert abs(magnitude.max() - amp) < 0.05, f"amp {magnitude.max():.3f}, expected {amp}"
    print(f"test_peak_on_bin       OK  peak={peak:.1f} Hz  amplitude={magnitude.max():.3f}")


def test_two_tones():
    """A loud 440 Hz tone plus a quieter 5000 Hz tone.

    The dominant peak should be the 440 Hz tone, and a clear second peak should
    appear near 5000 Hz.
    """
    sample_rate = 44100
    n = 4096
    t = np.arange(n) / sample_rate
    signal = 0.30 * np.sin(2 * np.pi * 440 * t) + 0.20 * np.sin(2 * np.pi * 5000 * t)

    freqs, magnitude = compute_spectrum(signal, sample_rate, make_window(n))
    dominant = _peak_freq(freqs, magnitude)

    # Find the second tone by looking only above 2 kHz.
    high = freqs > 2000
    second = freqs[high][int(np.argmax(magnitude[high]))]

    assert abs(dominant - 440) < 15, f"dominant peak {dominant} Hz, expected ~440 Hz"
    assert abs(second - 5000) < 30, f"second peak {second} Hz, expected ~5000 Hz"
    print(f"test_two_tones         OK  peaks at {dominant:.0f} Hz and {second:.0f} Hz")


def test_db_conversion():
    """Silence should sit at the dB floor; full-scale should sit near 0 dBFS."""
    silence = magnitude_to_db(np.array([0.0]))
    full = magnitude_to_db(np.array([1.0]))
    assert silence[0] <= -119.0, f"silence at {silence[0]} dB"
    assert abs(full[0]) < 0.001, f"full scale at {full[0]} dB"
    print(f"test_db_conversion     OK  silence={silence[0]:.0f} dBFS  fullscale={full[0]:.1f} dBFS")


if __name__ == "__main__":
    test_peak_on_bin()
    test_two_tones()
    test_db_conversion()
    print("\nAll tests passed. The DSP core is correct.")
