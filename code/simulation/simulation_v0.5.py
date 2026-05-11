# directional_stereo_beamforming_beam_v1.py
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import librosa
import soundfile as sf

# -----------------------------
# 1. Load and normalise click tracks
# -----------------------------
def normalize_audio(signal):
    return signal / np.sqrt(np.mean(signal**2))

click_a, sr_a = librosa.load("audio/quietriot.wav", sr=None)
click_b, sr_b = librosa.load("audio/beethoven.wav", sr=None)

min_len = min(len(click_a), len(click_b))
click_a = normalize_audio(click_a[:min_len])
click_b = normalize_audio(click_b[:min_len])

# -----------------------------
# 2. Define room
# -----------------------------
room_dim = [5.0, 4.0, 3.0]  # X, Y, Z
absorption = 0.3
room = pra.ShoeBox(room_dim, fs=sr_a, materials=pra.Material(absorption), max_order=0)  # reflections ignored

# -----------------------------
# 3. Define speaker array
# -----------------------------
num_speakers = 60
x_positions = np.linspace(0, room_dim[0], num_speakers)
y_wall = room_dim[1]  # top wall
z_height = 1.0

speakers = np.zeros((3, num_speakers))
for i in range(num_speakers):
    speakers[:, i] = [x_positions[i], y_wall, z_height]

# -----------------------------
# 4. Define listener positions
# -----------------------------
listener_a_pos = np.array([1.0, 1.0, 1.0])
listener_b_pos = np.array([4.0, 1.0, 1.0])
room.add_microphone(listener_a_pos)
room.add_microphone(listener_b_pos)

# -----------------------------
# 5. Beamforming helper
# -----------------------------
def delayed_signal(signal, distance, sr):
    delay_samples = int(distance / 343 * sr)
    out = np.zeros(len(signal) + delay_samples)
    out[delay_samples:delay_samples+len(signal)] = signal
    return out[:len(signal)]

def apply_beamforming(listener_pos, speakers, signals, sr, offaxis_scale=0.3):
    """
    Sum contributions from all speakers with beamforming:
    - Each speaker delayed for distance
    - Off-axis speakers attenuated
    """
    left = np.zeros(min_len)
    right = np.zeros(min_len)
    mid_x = np.mean(speakers[0, :])
    
    for i, spk in enumerate(speakers.T):
        distance = np.linalg.norm(listener_pos - spk)
        sig = delayed_signal(signals[i], distance, sr)
        
        # Off-axis attenuation
        if listener_pos[0] < mid_x and spk[0] > mid_x:
            sig *= offaxis_scale
        elif listener_pos[0] > mid_x and spk[0] <= mid_x:
            sig *= offaxis_scale
        
        # Assign to left/right
        if spk[0] <= mid_x:
            left += sig
        else:
            right += sig
    
    # Normalize
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_val > 0:
        left /= max_val
        right /= max_val
        
    return np.vstack((left, right)).T

# -----------------------------
# 6. Generate speaker signals
# -----------------------------
speaker_signals = np.zeros((num_speakers, min_len))
for i in range(num_speakers):
    if i < num_speakers // 2:
        speaker_signals[i, :] = click_a
    else:
        speaker_signals[i, :] = click_b
    room.add_source(speakers[:, i], signal=speaker_signals[i, :])

# -----------------------------
# 7. Generate stereo audio for each listener
# -----------------------------
stereo_a = apply_beamforming(listener_a_pos, speakers, speaker_signals, sr_a)
stereo_b = apply_beamforming(listener_b_pos, speakers, speaker_signals, sr_a)

# Reduce amplitude to avoid distortion
stereo_a *= 0.25
stereo_b *= 0.25

# Save audio files
sf.write("audio/a_no_angle60.wav", stereo_a, sr_a)
sf.write("audio/b_no_angle60.wav", stereo_b, sr_a)
print("Stereo audio files saved for listeners A and B.")

# -----------------------------
# 8. Compute energy maps
# -----------------------------
nx, ny = 50, 50
x = np.linspace(0, room_dim[0], nx)
y = np.linspace(0, room_dim[1], ny)

def compute_energy_map(speakers_subset, signal_subset):
    energy_map = np.zeros((nx, ny))
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            pos = np.array([xx, yy, 1.0])
            sig = np.zeros(min_len)
            for i, spk in enumerate(speakers_subset.T):
                d = np.linalg.norm(pos - spk)
                sig += delayed_signal(signal_subset[i], d, sr_a)
            energy_map[ix, iy] = np.mean(sig**2)
    return energy_map

energy_map_a = compute_energy_map(speakers[:, :num_speakers//2], speaker_signals[:num_speakers//2])
energy_map_b = compute_energy_map(speakers[:, num_speakers//2:], speaker_signals[num_speakers//2:])
energy_map_combined = energy_map_a + energy_map_b

# -----------------------------
# 9. Plot beam energy
# -----------------------------
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.imshow(energy_map_a.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Blues')
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.title("Beam Energy: Click Track A")
plt.colorbar(label="Energy")
plt.legend()

plt.subplot(1,3,2)
plt.imshow(energy_map_b.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Greens')
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.title("Beam Energy: Click Track B")
plt.colorbar(label="Energy")
plt.legend()

plt.subplot(1,3,3)
plt.imshow(energy_map_combined.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Purples')
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.title("Combined Beam Energy")
plt.colorbar(label="Energy")
plt.legend()

plt.tight_layout()
plt.show()
