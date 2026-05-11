# directional_stereo_beamforming_topwall_v4.py
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import librosa
import soundfile as sf

# -----------------------------
# 1. Load and normalise songs
# -----------------------------
def normalize_audio(signal):
    return signal / np.sqrt(np.mean(signal**2))

song_a, sr_a = librosa.load("audio/aceofbase.wav", sr=None)
song_b, sr_b = librosa.load("audio/metallica.wav", sr=None)

# Truncate to shortest
min_len = min(len(song_a), len(song_b))
song_a = normalize_audio(song_a[:min_len])
song_b = normalize_audio(song_b[:min_len])

# -----------------------------
# 2. Define room
# -----------------------------
room_dim = [5.0, 4.0, 3.0]  # X = width, Y = length, Z = height
absorption = 0.3
room = pra.ShoeBox(room_dim, fs=sr_a, materials=pra.Material(absorption), max_order=3)

# -----------------------------
# 3. Define speaker array (spans full top wall)
# -----------------------------
num_speakers = 12
x_positions = np.linspace(0, room_dim[0], num_speakers)
y_wall = room_dim[1]  # top wall
z_height = 1.0        # mid-level height

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
# 5. Generate speaker signals with delays
# -----------------------------
def delayed_signal(signal, distance, sr):
    """Add propagation delay based on distance (speed of sound 343 m/s)."""
    delay_samples = int(distance / 343 * sr)
    out = np.zeros(len(signal) + delay_samples)
    out[delay_samples:delay_samples+len(signal)] = signal
    return out[:len(signal)]

speaker_signals = np.zeros((num_speakers, min_len))
for i in range(num_speakers):
    if i < num_speakers // 2:
        speaker_signals[i, :] = delayed_signal(song_a, np.linalg.norm(listener_a_pos - speakers[:, i]), sr_a)
    else:
        speaker_signals[i, :] = delayed_signal(song_b, np.linalg.norm(listener_b_pos - speakers[:, i]), sr_a)
    room.add_source(speakers[:, i], signal=speaker_signals[i, :])

# -----------------------------
# 6. Run room simulation
# -----------------------------
room.compute_rir()
room.simulate()

# -----------------------------
# 7. Stereo processing for each listener
# -----------------------------
def stereo_listener(listener_pos, speakers, signals, sr, offaxis_scale=0.3):
    """Generate stereo signal for listener with left/right assignment and off-axis scaling."""
    left = np.zeros(min_len)
    right = np.zeros(min_len)
    mid_x = np.mean(speakers[0, :])
    for i, spk in enumerate(speakers.T):
        distance = np.linalg.norm(listener_pos - spk)
        sig = delayed_signal(signals[i], distance, sr)
        # Reduce off-axis amplitude
        if listener_pos[0] < mid_x and spk[0] > mid_x:
            sig *= offaxis_scale
        elif listener_pos[0] > mid_x and spk[0] <= mid_x:
            sig *= offaxis_scale
        # Assign left/right channel
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

# Generate six audio files
stereo_a_listener_a = stereo_listener(listener_a_pos, speakers[:, :num_speakers//2], speaker_signals[:num_speakers//2], sr_a)
stereo_a_listener_b = stereo_listener(listener_b_pos, speakers[:, :num_speakers//2], speaker_signals[:num_speakers//2], sr_a)
stereo_b_listener_a = stereo_listener(listener_a_pos, speakers[:, num_speakers//2:], speaker_signals[num_speakers//2:], sr_a)
stereo_b_listener_b = stereo_listener(listener_b_pos, speakers[:, num_speakers//2:], speaker_signals[num_speakers//2:], sr_a)
stereo_combined_listener_a = stereo_listener(listener_a_pos, speakers, speaker_signals, sr_a)
stereo_combined_listener_b = stereo_listener(listener_b_pos, speakers, speaker_signals, sr_a)

# Reduce amplitude to avoid distortion
stereo_a_listener_a *= 0.25
stereo_a_listener_b *= 0.25
stereo_b_listener_a *= 0.25
stereo_b_listener_b *= 0.25
stereo_combined_listener_a *= 0.25
stereo_combined_listener_b *= 0.25

# Save files
sf.write("audio/listener_a_songA.wav", stereo_a_listener_a, sr_a)
sf.write("audio/listener_b_songA.wav", stereo_a_listener_b, sr_a)
sf.write("audio/listener_a_songB.wav", stereo_b_listener_a, sr_a)
sf.write("audio/listener_b_songB.wav", stereo_b_listener_b, sr_a)
sf.write("audio/listener_a_combined.wav", stereo_combined_listener_a, sr_a)
sf.write("audio/listener_b_combined.wav", stereo_combined_listener_b, sr_a)
print("All 6 stereo audio files saved.")

# -----------------------------
# 8. Compute energy maps for plotting
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
energy_map_combined = energy_map_a + energy_map_b  # true combined energy

# -----------------------------
# 9. Plot energy maps with overlap highlighting
# -----------------------------
plt.figure(figsize=(18,5))

# Song A energy
plt.subplot(1,3,1)
plt.imshow(energy_map_a.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]],
           cmap='Blues', vmin=0, vmax=np.max(energy_map_combined))
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Song A Energy (First-half Array)")
plt.colorbar(label="Energy")
plt.legend()

# Song B energy
plt.subplot(1,3,2)
plt.imshow(energy_map_b.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]],
           cmap='Greens', vmin=0, vmax=np.max(energy_map_combined))
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Song B Energy (Second-half Array)")
plt.colorbar(label="Energy")
plt.legend()

# Combined energy with clearer overlap
plt.subplot(1,3,3)
plt.imshow(energy_map_a.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Blues', alpha=0.5, vmin=0, vmax=np.max(energy_map_combined))
plt.imshow(energy_map_b.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Greens', alpha=0.5, vmin=0, vmax=np.max(energy_map_combined))
# Overlap mask
threshold_a = 0.2 * np.max(energy_map_a)
threshold_b = 0.2 * np.max(energy_map_b)
bleed_mask = (energy_map_a.T > threshold_a) & (energy_map_b.T > threshold_b)
plt.imshow(np.where(bleed_mask, 1, np.nan), origin='lower', extent=[0, room_dim[0], 0, room_dim[1]],
           cmap='Reds', alpha=0.7)

plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Combined Energy with Overlap Highlighted")
plt.colorbar(label="Energy")
plt.legend()

plt.tight_layout()
plt.show()
