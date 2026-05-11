import numpy as np
import librosa
import soundfile as sf

# -----------------------------
# 1. Load and normalize audio tracks
# -----------------------------
def normalize_audio(signal):
    return signal / np.sqrt(np.mean(signal**2))

audio_a, sr_a = librosa.load("audio/normalized_beethoven2.wav", sr=None)
audio_b, sr_b = librosa.load("audio/normalized_weather.wav", sr=None)

min_len = min(len(audio_a), len(audio_b))
audio_a = normalize_audio(audio_a[:min_len])
audio_b = normalize_audio(audio_b[:min_len])

# -----------------------------
# 2. Define speaker arrays and listener positions
# -----------------------------
speakers_a = np.array([
    [0.0, 1.0],
    [0.1, 1.14],
    [0.3, 1.14],
    [0.4, 1.0]
])

speakers_b = np.array([
    [0.81, 1.0],
    [0.91, 1.14],
    [1.11, 1.14],
    [1.21, 1.0]
])

listener_a_pos = np.array([0.2, 0.0])  
listener_b_pos = np.array([1.01, 0.0])

speed_of_sound = 343.0  # m/s

# -----------------------------
# 3. Calculate RELATIVE delays
# -----------------------------
def calculate_delays(listener_pos, speaker_array, sr):
    raw_delays = []

    for spk in speaker_array:
        distance = np.linalg.norm(listener_pos - spk)
        delay_in_samples = (distance / speed_of_sound) * sr
        raw_delays.append(delay_in_samples)

    raw_delays = np.array(raw_delays)
    relative_delays = raw_delays - np.min(raw_delays)

    return relative_delays

delays_a = calculate_delays(listener_a_pos, speakers_a, sr_a)
delays_b = calculate_delays(listener_b_pos, speakers_b, sr_a)

# -----------------------------
# PRINT THE DELAYS
# -----------------------------
print("\n=== Delay Values ===")

print("\nArray A delays (samples):")
print(delays_a)
print("Array A delays (ms):")
print(delays_a / sr_a * 1000)

print("\nArray B delays (samples):")
print(delays_b)
print("Array B delays (ms):")
print(delays_b / sr_b * 1000)

# -----------------------------
# 4. Fractional-sample delay (sinc interpolation)
# -----------------------------
def fractional_delay(signal, delay_samples):
    N = 64  # Sinc kernel half-width
    n = np.arange(-N, N)

    frac = delay_samples - int(delay_samples)
    kernel = np.sinc(n - frac)
    kernel *= np.hanning(len(kernel))  # Hann window to reduce ringing

    padded = np.pad(signal, (int(delay_samples), 0))
    out = np.convolve(padded, kernel, mode='full')

    return out

# -----------------------------
# 5. Gentle taper
# -----------------------------
weights_a = np.array([0.85, 1.0, 1.0, 0.85])
weights_b = np.array([0.85, 1.0, 1.0, 0.85])

# -----------------------------
# 6. Create per-speaker signals
# -----------------------------
speaker_signals_a = []
speaker_signals_b = []

# Array A
for i, d in enumerate(delays_a):
    sig = fractional_delay(audio_a, d)
    sig *= weights_a[i]
    speaker_signals_a.append(sig)

# Array B
for i, d in enumerate(delays_b):
    sig = fractional_delay(audio_b, d)
    sig *= weights_b[i]
    speaker_signals_b.append(sig)

# -----------------------------
# 7. Normalize gain
# -----------------------------
gain = 0.25

speaker_signals_a = [sig * gain for sig in speaker_signals_a]
speaker_signals_b = [sig * gain for sig in speaker_signals_b]

# -----------------------------
# 8. Save WAV files
# -----------------------------
for i, sig in enumerate(speaker_signals_a):
    sf.write(f"audio/test1_beethoven2_speakerA_{i+1}.wav", sig, sr_a)

for i, sig in enumerate(speaker_signals_b):
    sf.write(f"audio/test1_weather_speakerB_{i+1}.wav", sig, sr_a)

print("\nBeamforming WAVs generated with gentle taper applied.")
