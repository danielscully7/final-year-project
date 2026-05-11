import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal

# -----------------------------
# 1. Load and normalize audio tracks
# -----------------------------
def normalize_audio(signal_in): 
    return signal_in / np.sqrt(np.mean(signal_in**2) + 1e-8) 

# load normalized audio files here
audio_a, sr_a = librosa.load("audio/normalized_radio.wav", sr=None) 
audio_b, sr_b = librosa.load("audio/normalized_weather.wav", sr=None)

# Ensure both tracks are the same length
min_len = min(len(audio_a), len(audio_b)) 
audio_a = normalize_audio(audio_a[:min_len])
audio_b = normalize_audio(audio_b[:min_len])

# -----------------------------
# 2. HIGH-PASS FILTER (reduce low-frequency bleed)
# -----------------------------
def highpass_filter(sig, sr, cutoff=250): # 250 Hz cutoff
    b, a = signal.butter(4, cutoff / (sr / 2), btype='high')
    return signal.filtfilt(b, a, sig)

audio_a = highpass_filter(audio_a, sr_a) # 
audio_b = highpass_filter(audio_b, sr_b)

# -----------------------------
# 3. Define speaker arrays and listener positions
# -----------------------------
speakers_a = np.array([ # Array A speakers
    [0.0, 1.0],
    [0.1, 1.14],
    [0.3, 1.14],
    [0.4, 1.0]
])

speakers_b = np.array([ # Array B speakers
    [0.81, 1.0],
    [0.91, 1.14],
    [1.11, 1.14],
    [1.21, 1.0]
])

listener_a_pos = np.array([0.2, 0.0]) # Listener A position
listener_b_pos = np.array([1.0, 0.0]) # Listener B position

speed_of_sound = 343.0  # m/s

# -----------------------------
# 4. Calculate RELATIVE delays
# -----------------------------
def calculate_delays(listener_pos, speaker_array, sr): # Calculate relative delays from listener to each speaker
    raw_delays = []

    for spk in speaker_array:
        distance = np.linalg.norm(listener_pos - spk) # Calculate distance from listener to speaker
        delay_in_samples = (distance / speed_of_sound) * sr # Convert distance to delay in samples
        raw_delays.append(delay_in_samples)

    raw_delays = np.array(raw_delays)
    relative_delays = raw_delays - np.min(raw_delays) # Make the shortest delay zero (relative to closest speaker)

    return relative_delays

# Slight over-steering improves separation
steer_factor = 1.2

delays_a = steer_factor * calculate_delays(listener_a_pos, speakers_a, sr_a)
delays_b = steer_factor * calculate_delays(listener_b_pos, speakers_b, sr_b)

# -----------------------------
# 5. Fractional-sample delay
# -----------------------------
def fractional_delay(signal_in, delay_samples): # Apply fractional delay using sinc interpolation
    int_delay = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay

    N = 64
    n = np.arange(-N, N)

    kernel = np.sinc(n - frac_delay) # Sinc interpolation kernel with Hanning window
    kernel *= np.hanning(len(kernel))

    padded = np.pad(signal_in, (int_delay, 0)) # Pad signal to allow for integer delay
    out = np.convolve(padded, kernel, mode='full') # Apply fractional delay

    return out

# -----------------------------
# 6. Custom taper (optimised for 4 speakers)
# -----------------------------
weights_a = np.array([0.75, 1.0, 1.0, 0.75]) # Tapered weights for Array A (reduce outer speakers slightly)
weights_b = np.array([0.75, 1.0, 1.0, 0.75]) # Tapered weights for Array B (reduce outer speakers slightly)

# Normalize weights to sum to 1 for each array
weights_a /= np.sum(weights_a) 
weights_b /= np.sum(weights_b) 

# -----------------------------
# 7. Create per-speaker signals
# -----------------------------
speaker_signals_a = []
speaker_signals_b = []

# Array A
for i, d in enumerate(delays_a):
    sig = fractional_delay(audio_a, d) # Apply fractional delay

    # Distance-based attenuation 
    distance = np.linalg.norm(listener_a_pos - speakers_a[i])
    attenuation = 1.0 / (distance + 1e-6)

    sig *= weights_a[i] * attenuation # Apply custom taper and distance attenuation
    speaker_signals_a.append(sig)

# Array B
for i, d in enumerate(delays_b):
    sig = fractional_delay(audio_b, d) # Apply fractional delay

    distance = np.linalg.norm(listener_b_pos - speakers_b[i])
    attenuation = 1.0 / (distance + 1e-6)

    sig *= weights_b[i] * attenuation # Apply custom taper and distance attenuation
    speaker_signals_b.append(sig)

# -----------------------------
# 8. Normalize gain
# -----------------------------
gain = 0.25 # Overall gain to prevent clipping when summing speakers

speaker_signals_a = [sig * gain for sig in speaker_signals_a]
speaker_signals_b = [sig * gain for sig in speaker_signals_b]

# -----------------------------
# 9. Save WAV files
# -----------------------------
for i, sig in enumerate(speaker_signals_a):
    sf.write(f"audio/v3_radio_speakerA_{i+1}.wav", sig, sr_a)

for i, sig in enumerate(speaker_signals_b):
    sf.write(f"audio/v3_weather_speakerB_{i+1}.wav", sig, sr_b)

print("\nBeamforming WAVs generated with custom taper.")