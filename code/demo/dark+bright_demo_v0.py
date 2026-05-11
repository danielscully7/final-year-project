import soundfile as sf
import numpy as np

# Load the sweep file
audio, fs = sf.read("audio/2400Hz.wav")

# Invert the waveform
audio_inverted = -1 * audio

# Save the inverted sweep
sf.write("audio/2400Hz_inverted.wav", audio_inverted, fs)

print("Inverted sweep saved as 2400Hz_inverted.wav")