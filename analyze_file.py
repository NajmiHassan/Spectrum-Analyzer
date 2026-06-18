"""
analyze_file.py
===============

A bonus, microphone-free way to use the same DSP core. It loads an audio FILE
with Librosa (the library named on the project card) and shows three views:

  1. Waveform        - the raw signal in the time domain.
  2. Spectrum        - one frame run through our compute_spectrum (the FFT).
  3. Spectrogram     - frequency content over time (FFT applied to many frames).

This is handy when microphone permissions are a pain, or when you want a stable
picture you can stare at instead of a moving one.

Run it with:  python analyze_file.py path/to/audio.wav
Works with WAV, MP3, FLAC, OGG, and more (whatever your Librosa build supports).
"""

import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from dsp_core import compute_spectrum, magnitude_to_db, make_window


def main(path: str):
    # sr=None keeps the file's native sample rate; mono=True averages to one channel.
    samples, sample_rate = librosa.load(path, sr=None, mono=True)
    print(f"Loaded {path}: {len(samples)} samples at {sample_rate} Hz "
          f"({len(samples) / sample_rate:.2f} s)")

    # Take one analysis frame from the middle of the file for the FFT plot.
    n = 1 << 14   # 16384 samples -> fine frequency resolution.
    if len(samples) < n:
        frame = np.pad(samples, (0, n - len(samples)))
    else:
        start = (len(samples) - n) // 2
        frame = samples[start:start + n]

    freqs, magnitude = compute_spectrum(frame, sample_rate, make_window(n))
    spectrum_db = magnitude_to_db(magnitude)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # 1. Waveform
    t = np.arange(len(samples)) / sample_rate
    axes[0].plot(t, samples, lw=0.5)
    axes[0].set_title("Waveform (time domain)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # 2. Magnitude spectrum from our own DSP core
    axes[1].semilogx(freqs, spectrum_db, lw=1.0)
    axes[1].set_xlim(20, sample_rate / 2)
    axes[1].set_ylim(-120, 0)
    axes[1].set_title("Magnitude spectrum of one frame (our FFT)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("dBFS")
    axes[1].grid(True, which="both", ls=":", alpha=0.4)

    # 3. Spectrogram (Librosa's STFT = the FFT slid across the whole file)
    stft_db = librosa.amplitude_to_db(np.abs(librosa.stft(samples)), ref=np.max)
    img = librosa.display.specshow(
        stft_db, sr=sample_rate, x_axis="time", y_axis="log", ax=axes[2]
    )
    axes[2].set_title("Spectrogram (frequency content over time)")
    fig.colorbar(img, ax=axes[2], format="%+2.0f dB")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_file.py path/to/audio.wav")
        sys.exit(1)
    main(sys.argv[1])
