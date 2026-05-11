import numpy as np
import librosa
import soundfile as sf

speed_of_sound = 343.0

# -----------------------------
# 1. Load audio
# -----------------------------
audio, sr = librosa.load("audio/2400Hz.wav", sr=None)
audio = audio / np.sqrt(np.mean(audio**2))  # normalize
freq = 2400
omega = 2 * np.pi * freq

# -----------------------------
# 2. Speaker positions
# -----------------------------
speakers = np.array([
    [0.0, 1.0],
    [0.1, 1.14],
    [0.3, 1.14],
    [0.4, 1.0],
    [0.81, 1.0],
    [0.91, 1.14],
    [1.11, 1.14],
    [1.21, 1.0]
])

# -----------------------------
# 3. Bright and dark zones
# -----------------------------
bright = np.array([0.2, 0.0])
dark = np.array([1.01, 0.0])

# -----------------------------
# 4. Steering vector
# -----------------------------
def steering_vector(point):
    d = np.linalg.norm(speakers - point, axis=1) # distance from each speaker to the target point
    phase = np.exp(-1j * omega * d / speed_of_sound) # phase shift due to propagation delay
    return phase / d # distance attenuation

g_bright = steering_vector(bright) # acoustic transfer to bright zone
g_dark = steering_vector(dark) # acoustic transfer to dark zone

# -----------------------------
# 5. Acoustic Contrast Control
# -----------------------------
R_b = np.outer(g_bright, np.conj(g_bright)) # covariance matrix for bright zone
R_d = np.outer(g_dark, np.conj(g_dark)) # covariance matrix for dark zone

eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(R_d) @ R_b) # generalized eigenvalue problem to maximize contrast
w = eigvecs[:, np.argmax(eigvals)] # optimal weights for maximum contrast
w = w / np.max(np.abs(w))  # normalize

# -----------------------------
# 6. Print speaker weights info
# -----------------------------
print("\n=== Speaker Weights ===") 
for i, weight in enumerate(w, start=1):
    print(f"Speaker {i}: magnitude={np.abs(weight):.3f}, phase={np.angle(weight, deg=True):.1f}°") # print magnitude and phase in degrees

# -----------------------------
# 7. Generate delayed signals
# -----------------------------
def apply_phase(signal, phase):
    # Convert phase to sample delay
    shift = int((np.angle(phase) / (2 * np.pi * freq)) * sr) # convert phase to time delay, then to samples
    return np.roll(signal, shift) # circular shift to simulate delay

speaker_signals = []
for i in range(len(speakers)):
    sig = apply_phase(audio, w[i]) # apply phase delay for this speaker
    sig = sig * np.abs(w[i]) # apply amplitude weighting
    speaker_signals.append(sig)

# -----------------------------
# 8. Simulate listener signals 
# -----------------------------
def simulate_listener(point):
    combined = np.zeros_like(speaker_signals[0])
    for i, spk in enumerate(speakers):
        distance = np.linalg.norm(point - spk) # distance from speaker to listener
        delay = int((distance / speed_of_sound) * sr) # convert distance to time delay, then to samples
        sig_delayed = np.roll(speaker_signals[i], delay) # apply delay to speaker signal
        sig_delayed /= distance  # apply distance attenuation
        combined += sig_delayed # sum contributions from all speakers
    return combined

bright_out = simulate_listener(bright) # simulate signal at bright zone
dark_out = simulate_listener(dark) # simulate signal at dark zone

bright_rms = np.sqrt(np.mean(bright_out**2)) # calculate RMS of bright zone output
dark_rms = np.sqrt(np.mean(dark_out**2)) # calculate RMS of dark zone output
contrast_db = 20 * np.log10(bright_rms / dark_rms + 1e-12) # calculate contrast in dB, add small value to avoid log(0)

print("\n=== Listener Simulation ===")
print(f"Bright zone RMS: {bright_rms:.4f}")
print(f"Dark zone RMS:   {dark_rms:.4f}")
print(f"Contrast (dB):  {contrast_db:.2f} dB")


# -----------------------------
# 9. Save WAV files
# -----------------------------
for i, sig in enumerate(speaker_signals):
    sf.write(f"audio/silence_{i+1}.wav", sig, sr)

print("\nSilent beam signals generated and saved as WAV files.")