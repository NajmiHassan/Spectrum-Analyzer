"""
dsp_core.py
===========

The heart of the spectrum analyzer, with ZERO audio or plotting dependencies.

Everything here is pure NumPy. That separation is on purpose: the actual signal
processing (windowing -> FFT -> magnitude -> dB) lives here and is fully
testable on synthetic signals, while the messy real-world parts (grabbing audio
from a microphone, drawing a live plot) live in other files.

If you understand the four functions below, you understand the project. The
real-time analyzer is "just" these functions called 30 times a second on fresh
microphone data.
"""

import numpy as np


def make_window(n: int) -> np.ndarray:
    """Return a Hann (a.k.a. Hanning) window of length ``n``.

    WHY WINDOWING?
    --------------
    The FFT assumes the chunk of audio you give it repeats forever, end-to-end.
    A raw chunk almost never starts and ends at the same value, so that imagined
    repetition has a sharp "step" at the seam. The FFT sees that step as extra
    high-frequency energy and smears every real tone across nearby bins. This is
    called *spectral leakage*.

    A window fixes this by tapering the chunk smoothly down to zero at both
    edges, so the seam is seamless. The Hann window is the classic, friendly
    default: good leakage suppression, easy to reason about.
    """
    return np.hanning(n)


def compute_spectrum(samples, sample_rate: int, window=None):
    """Turn a block of audio samples into a frequency spectrum.

    Parameters
    ----------
    samples : 1-D array of audio samples (the time domain signal)
    sample_rate : how many samples per second were recorded (e.g. 44100)
    window : optional precomputed window; if None a Hann window is made for you

    Returns
    -------
    freqs : the frequency (in Hz) that each magnitude value corresponds to
    magnitude : the strength of each frequency, normalized so a pure sine wave
                of amplitude A shows up as a peak of height ~A

    THE PIPELINE
    ------------
    1. Multiply the samples by the window (taper the edges -> less leakage).
    2. rfft: the real-input Fast Fourier Transform. Because audio is real-valued
       (not complex), we only need the "positive" half of the spectrum, so rfft
       is twice as fast as a full fft and returns n/2 + 1 values.
    3. abs(): the FFT output is complex (it encodes amplitude AND phase). We take
       the magnitude because we only care about "how much of this frequency is
       present", not its phase.
    4. Normalize so the numbers mean something physical (amplitude recovery).
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if window is None:
        window = make_window(n)

    windowed = samples * window

    # Real FFT: input is n real samples, output is n/2 + 1 complex numbers.
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)

    # Amplitude-correct normalization.
    # A windowed sine of amplitude A produces a peak of height A * sum(window)/2,
    # so dividing by sum(window)/2 makes the peak read back as A. This is what
    # lets you trust the y-axis instead of seeing arbitrary big numbers.
    magnitude = magnitude / (np.sum(window) / 2.0)

    # rfftfreq builds the frequency axis: bin k corresponds to k * sample_rate / n.
    # The highest representable frequency is sample_rate / 2 (the Nyquist limit).
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return freqs, magnitude


def magnitude_to_db(magnitude, floor_db: float = -120.0):
    """Convert a linear magnitude spectrum to decibels (dBFS).

    WHY dB?
    -------
    Hearing is roughly logarithmic, and microphone signals span a huge dynamic
    range (a whisper vs. a shout differ by thousands of times in amplitude). On
    a linear axis quiet detail collapses to a flat line near zero. The decibel
    scale (20 * log10) compresses that range so you can actually see structure.

    0 dBFS = full scale (the loudest a normalized sample can be). Everything
    quieter is negative. We clamp to ``floor_db`` so silence doesn't plunge to
    negative infinity and wreck the plot.
    """
    db = 20.0 * np.log10(magnitude + 1e-12)
    return np.maximum(db, floor_db)
