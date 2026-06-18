"""
spectrum_analyzer.py
====================

The main event: a live audio spectrum analyzer.

It opens your microphone, grabs small blocks of audio many times per second,
runs each block through the DSP core (window -> FFT -> magnitude -> dB), and
animates the result so you watch the frequency content of whatever you say,
sing, or play in real time.

Run it with:  python spectrum_analyzer.py
Close the plot window to stop.
"""

import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dsp_core import make_window, compute_spectrum, magnitude_to_db

# ----------------------------------------------------------------------------
# Configuration. Tweak these and re-run to see what changes.
# ----------------------------------------------------------------------------
SAMPLE_RATE = 44100   # samples per second. 44100 is CD quality and a safe default.
BLOCK_SIZE = 4096     # samples per FFT frame. Bigger = finer frequency detail but
                      # slower updates. Frequency resolution = SAMPLE_RATE / BLOCK_SIZE
                      # which here is about 10.8 Hz per bin.
CHANNELS = 1          # mono input. We only need one microphone channel.

# Precompute the things that never change so the hot loop stays fast.
WINDOW = make_window(BLOCK_SIZE)
FREQS = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)

# A buffer the audio thread writes into and the plot reads from. Audio arrives on
# a separate thread (via the callback below), so we keep the latest block here.
latest_block = np.zeros(BLOCK_SIZE, dtype=np.float64)


def audio_callback(indata, frames, time_info, status):
    """Called automatically by sounddevice every time a new block of audio is ready.

    ``indata`` has shape (frames, channels). We grab channel 0 and stash it.
    This function must be fast and must not block, so it does nothing but copy.
    """
    global latest_block
    if status:
        # e.g. input overflow if the machine is busy. Informational, not fatal.
        print(status, file=sys.stderr)
    latest_block = indata[:, 0].astype(np.float64)


def main():
    # ---- Set up the plot once ----
    fig, ax = plt.subplots(figsize=(10, 5))

    # Start the line flat at the noise floor. semilogx gives a log frequency axis,
    # which matches how we hear: each octave (doubling of frequency) gets equal width.
    line, = ax.semilogx(FREQS, np.full_like(FREQS, -120.0), lw=1.0)

    ax.set_xlim(20, SAMPLE_RATE / 2)   # 20 Hz (low end of hearing) up to Nyquist.
    ax.set_ylim(-120, 0)               # dBFS: 0 is loudest, more negative is quieter.
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_title("Real-time audio spectrum  -  close window to stop")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    def update(_frame):
        """Called ~30x/sec by the animation. Analyze the newest audio and redraw."""
        _, magnitude = compute_spectrum(latest_block, SAMPLE_RATE, WINDOW)
        line.set_ydata(magnitude_to_db(magnitude))
        return (line,)

    # blit=True only redraws what changed (the line), which keeps it smooth.
    # cache_frame_data=False because this is an open-ended live stream, not a clip.
    anim = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)

    # ---- Open the microphone and show the plot ----
    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )

    print("Listening to your microphone. Make some noise!")
    print("Try whistling a rising note and watch the peak slide to the right.")

    with stream:
        plt.show()   # blocks here until you close the window; audio runs in background.

    # Keep a reference to anim so it is not garbage-collected before plt.show returns.
    del anim


if __name__ == "__main__":
    main()
