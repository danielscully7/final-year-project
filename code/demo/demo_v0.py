import numpy as np
import librosa
import soundfile as sf

# -----------------------------
# 1. Load and normalize audio tracks
# -----------------------------
def normalize_audio(signal):
    return signal / np.sqrt(np.mean(signal**2))

audio_a, sr_a = librosa.load("audio/2000to3000Hz.wav", sr=None)
audio_b, sr_b = librosa.load("audio/3500to5000Hz.wav", sr=None)

min_len = min(len(audio_a), len(audio_b))
audio_a = normalize_audio(audio_a[:min_len])
audio_b = normalize_audio(audio_b[:min_len])

# -----------------------------
# 2. Define speaker arrays and listener positions
# -----------------------------

speakers_a = np.array([
    [0.0, 0.16],
    [0.1, 0.3],
    [0.24, 0.3],
    [0.34, 0.16]
])

speakers_b = np.array([
    [0.54, 0.16],
    [0.64, 0.3],
    [0.78, 0.3],
    [0.88, 0.16]
])

listener_a_pos = np.array([0.17, 0.0])
listener_b_pos = np.array([0.71, 0.0])

# Combine arrays for processing
speakers = np.vstack((speakers_a, speakers_b))

# -----------------------------
# 3. Calculate per-speaker delays
# -----------------------------
speed_of_sound = 343.0  # m/s

def calculate_delays(listener_pos, speaker_array, sr):
    delays = []
    for i, spk in enumerate(speaker_array):
        distance = np.linalg.norm(listener_pos - spk)
        delay_samples = int(distance / speed_of_sound * sr)
        delay_ms = delay_samples / sr * 1000
        delays.append(delay_samples)
        print(f"Speaker {i+1}: Distance = {distance:.2f} m, Delay = {delay_samples} samples ({delay_ms:.2f} ms)")
    return delays

delays_a = calculate_delays(listener_a_pos, speakers_a, sr_a)
delays_b = calculate_delays(listener_b_pos, speakers_b, sr_a)

# -----------------------------
# 4. Apply delays to each track
# -----------------------------
def delayed_signal(signal, delay_samples):
    out = np.zeros(len(signal) + delay_samples)
    out[delay_samples:delay_samples+len(signal)] = signal
    return out[:len(signal)]  # trim back to original length

# Generate per-speaker signals
speaker_signals_a = np.zeros((speakers_a.shape[0], min_len))
speaker_signals_b = np.zeros((speakers_b.shape[0], min_len))

for i, d in enumerate(delays_a):
    speaker_signals_a[i, :] = delayed_signal(audio_a, d)

for i, d in enumerate(delays_b):
    speaker_signals_b[i, :] = delayed_signal(audio_b, d)

# -----------------------------
# 5. Normalize and apply gain
# -----------------------------
gain = 0.25  # scale down to avoid clipping

speaker_signals_a *= gain
speaker_signals_b *= gain

# -----------------------------
# 6. Save per-speaker audio files
# -----------------------------
for i, sig in enumerate(speaker_signals_a):
    sf.write(f"audio/sweepHZspeakerA_{i+1}.wav", sig, sr_a)

for i, sig in enumerate(speaker_signals_b):
    sf.write(f"audio/sweepHZspeakerB_{i+1}.wav", sig, sr_a)

print("Per-speaker audio files generated for Behringer UMC1820 setup.")
