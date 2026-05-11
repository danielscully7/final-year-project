# directional_stereo_beamforming_N_arrays_null.py
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

click_a, sr_a = librosa.load("audio/beethoven.wav", sr=None)
click_b, sr_b = librosa.load("audio/metallica.wav", sr=None)

min_len = min(len(click_a), len(click_b))
click_a = normalize_audio(click_a[:min_len])
click_b = normalize_audio(click_b[:min_len])

# -----------------------------
# 2. Define room
# -----------------------------
room_dim = [5.0, 4.0, 3.0]  # X, Y, Z
absorption = 0.3
room = pra.ShoeBox(room_dim, fs=sr_a, materials=pra.Material(absorption), max_order=0)

# -----------------------------
# 3. Define N-shaped speaker arrays
# -----------------------------
num_speakers = 8
curve_depth = 0.4  # how far toward the listener the curve comes
y_wall = room_dim[1]  # top wall
z_height = 1.0

# Left array (listener A)
x_a = np.linspace(0, room_dim[0]/2, num_speakers)
y_a = y_wall - curve_depth * (np.sin(np.linspace(-np.pi/2, np.pi/2, num_speakers))**2)
speakers_a = np.vstack((x_a, y_a, np.ones_like(x_a)*z_height))

# Right array (listener B)
x_b = np.linspace(room_dim[0]/2, room_dim[0], num_speakers)
y_b = y_wall - curve_depth * (np.sin(np.linspace(-np.pi/2, np.pi/2, num_speakers))**2)
speakers_b = np.vstack((x_b, y_b, np.ones_like(x_b)*z_height))

# Combine arrays for processing
speakers = np.hstack((speakers_a, speakers_b))

# -----------------------------
# 4. Define listener positions
# -----------------------------
listener_a_pos = np.array([1.25, 1.0, 1.0])
listener_b_pos = np.array([3.75, 1.0, 1.0])
room.add_microphone(listener_a_pos)
room.add_microphone(listener_b_pos)

# -----------------------------
# 5. Beamforming helpers with null-steering
# -----------------------------
def delayed_signal(signal, distance, sr):
    delay_samples = int(distance / 343 * sr)
    out = np.zeros(len(signal) + delay_samples)
    out[delay_samples:delay_samples+len(signal)] = signal
    return out[:len(signal)]

def apply_beamforming(listener_pos, speakers, signals, sr, offaxis_scale=0.3, null_array=None):
    """
    listener_pos: position of listener
    speakers: all speaker positions
    signals: all signals
    null_array: indices of speakers to attenuate (for null-steering)
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
        
        # Null-steering: attenuate speakers in the null array
        if null_array is not None and i in null_array:
            sig *= 0.1  # perfect null; could be a smaller factor for partial null
        
        # Assign to left/right channels
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
speaker_signals = np.zeros((speakers.shape[1], min_len))
for i in range(speakers.shape[1]):
    if i < speakers_a.shape[1]:
        speaker_signals[i, :] = click_a
    else:
        speaker_signals[i, :] = click_b
    room.add_source(speakers[:, i], signal=speaker_signals[i, :])

# -----------------------------
# 7. Generate stereo audio for each listener with null-steering
# -----------------------------
# Null arrays: each listener nulls the other array
null_a = np.arange(num_speakers, 2*num_speakers)  # attenuate right array for listener A
null_b = np.arange(0, num_speakers)  # attenuate left array for listener B

stereo_a = apply_beamforming(listener_a_pos, speakers, speaker_signals, sr_a, null_array=null_a)
stereo_b = apply_beamforming(listener_b_pos, speakers, speaker_signals, sr_a, null_array=null_b)

# Reduce amplitude to avoid clipping
stereo_a *= 0.25
stereo_b *= 0.25

# Save audio
sf.write("audio/listener_a_8_B+M_array_null.wav", stereo_a, sr_a)
sf.write("audio/listener_b_8_B+M_array_null.wav", stereo_b, sr_a)
print("Stereo N-array audio files with null-steering saved.")

# -----------------------------
# 8. Compute energy maps
# -----------------------------
nx, ny = 50, 50
x_grid = np.linspace(0, room_dim[0], nx)
y_grid = np.linspace(0, room_dim[1], ny)

def compute_energy_map(speakers_subset, signal_subset):
    energy_map = np.zeros((nx, ny))
    for ix, xx in enumerate(x_grid):
        for iy, yy in enumerate(y_grid):
            pos = np.array([xx, yy, 1.0])
            sig = np.zeros(min_len)
            for i, spk in enumerate(speakers_subset.T):
                d = np.linalg.norm(pos - spk)
                sig += delayed_signal(signal_subset[i], d, sr_a)
            energy_map[ix, iy] = np.mean(sig**2)
    return energy_map

energy_map_a = compute_energy_map(speakers_a, speaker_signals[:num_speakers])
energy_map_b = compute_energy_map(speakers_b, speaker_signals[num_speakers:])
energy_map_combined = energy_map_a + energy_map_b

# -----------------------------
# 9. Plot beam energy and speaker arrays
# -----------------------------
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.imshow(energy_map_a.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Blues')
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.scatter(speakers_a[0,:], speakers_a[1,:], c='blue', marker='o', label='Speakers A')
plt.title("Beam Energy: Listener A")
plt.colorbar(label="Energy")
plt.legend()

plt.subplot(1,3,2)
plt.imshow(energy_map_b.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Greens')
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.scatter(speakers_b[0,:], speakers_b[1,:], c='orange', marker='o', label='Speakers B')
plt.title("Beam Energy: Listener B")
plt.colorbar(label="Energy")
plt.legend()

plt.subplot(1,3,3)
plt.imshow(energy_map_combined.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='Purples')
plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x', label='Listener A')
plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x', label='Listener B')
plt.scatter(speakers_a[0,:], speakers_a[1,:], c='blue', marker='o', label='Speakers A')
plt.scatter(speakers_b[0,:], speakers_b[1,:], c='orange', marker='o', label='Speakers B')
plt.title("Combined Beam Energy")
plt.colorbar(label="Energy")
plt.legend()

plt.tight_layout()
plt.show()
