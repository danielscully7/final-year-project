# ==========================================================
# directional_stereo_beamforming_N_arrays_null.py
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import librosa
import soundfile as sf

# ==========================================================
# 1. Load and Normalise Audio
# ==========================================================

def normalize_audio(signal):
    return signal / np.sqrt(np.mean(signal**2) + 1e-12)

click_a, sr_a = librosa.load("audio/beethoven.wav", sr=None)
click_b, sr_b = librosa.load("audio/metallica.wav", sr=None)

min_len = min(len(click_a), len(click_b))
click_a = normalize_audio(click_a[:min_len])
click_b = normalize_audio(click_b[:min_len])

# ==========================================================
# 2. Room Geometry
# ==========================================================

room_dim = [5.0, 4.0, 3.0]
absorption = 0.3

room = pra.ShoeBox(
    room_dim,
    fs=sr_a,
    materials=pra.Material(absorption),
    max_order=0
)

# ==========================================================
# 3. HARDWARE SPEAKER GEOMETRY 
# ==========================================================

z_height = 1.0

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

# Add height dimension
speakers_a = np.hstack(
    (speakers_a, np.ones((speakers_a.shape[0], 1)) * z_height)
)

speakers_b = np.hstack(
    (speakers_b, np.ones((speakers_b.shape[0], 1)) * z_height)
)

speakers = np.vstack((speakers_a, speakers_b))

weights = np.array([0.85, 1.0, 1.0, 0.85])

# ==========================================================
# 4. Listener Positions 
# ==========================================================

listener_a_pos = np.array([0.17, -0.8, 1.0])
listener_b_pos = np.array([0.71, -0.8, 1.0])

room.add_microphone(listener_a_pos)
room.add_microphone(listener_b_pos)

# ==========================================================
# 5. Beamforming Functions
# ==========================================================

def delayed_signal(signal, distance, sr):
    delay_samples = int(distance / 343 * sr)
    out = np.zeros(len(signal) + delay_samples)
    out[delay_samples:delay_samples+len(signal)] = signal
    return out[:len(signal)]


def apply_beamforming(listener_pos, speakers, signals,
                      sr, offaxis_scale=0.3, null_array=None):

    left = np.zeros(min_len)
    right = np.zeros(min_len)

    mid_x = np.mean([speakers_a[0,0], speakers_b[0,0]])

    for i, spk in enumerate(speakers.T):

        distance = np.linalg.norm(listener_pos - spk)
        sig = delayed_signal(signals[i], distance, sr)

        # Off-axis attenuation
        if listener_pos[0] < mid_x and spk[0] > mid_x:
            sig *= offaxis_scale
        elif listener_pos[0] > mid_x and spk[0] <= mid_x:
            sig *= offaxis_scale

        # Null steering
        if null_array is not None and i in null_array:
            sig *= 0.1

        if spk[0] <= mid_x:
            left += sig
        else:
            right += sig

    # Normalise
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if max_val > 0:
        left /= max_val
        right /= max_val

    return np.vstack((left, right)).T


# ==========================================================
# 6. Assign Audio To Hardware Speakers
# ==========================================================

speaker_signals = np.zeros((speakers.shape[1], min_len))

for i in range(speakers.shape[1]):

    if i < len(speakers_a):
        speaker_signals[i, :] = click_a
    else:
        speaker_signals[i, :] = click_b

    room.add_source(speakers[:, i], signal=speaker_signals[i, :])


# ==========================================================
# 7. Generate Stereo Outputs With Null Steering
# ==========================================================

null_a = np.arange(len(speakers_a), 2 * len(speakers_a))
null_b = np.arange(0, len(speakers_a))

stereo_a = apply_beamforming(
    listener_a_pos,
    speakers,
    speaker_signals,
    sr_a,
    null_array=null_a
)

stereo_b = apply_beamforming(
    listener_b_pos,
    speakers,
    speaker_signals,
    sr_a,
    null_array=null_b
)

stereo_a *= 0.25
stereo_b *= 0.25

sf.write("audio/listener_a_hw_locked.wav", stereo_a, sr_a)
sf.write("audio/listener_b_hw_locked.wav", stereo_b, sr_a)

print("Stereo files saved (hardware locked).")

# ==========================================================
# 8. Energy Map (Corrected Colourbars)
# ==========================================================

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


energy_map_a = compute_energy_map(
    speakers_a,
    speaker_signals[:len(speakers_a)]
)

energy_map_b = compute_energy_map(
    speakers_b,
    speaker_signals[len(speakers_a):]
)

energy_map_combined = energy_map_a + energy_map_b

# ==========================================================
# 9. Plot With Proper Colourbars
# ==========================================================

plt.figure(figsize=(18,5))

for idx, (data, title) in enumerate([
    (energy_map_a, "Beam Energy: Listener A"),
    (energy_map_b, "Beam Energy: Listener B"),
    (energy_map_combined, "Combined Beam Energy")
]):

    plt.subplot(1,3,idx+1)

    im = plt.imshow(
        data.T,
        origin='lower',
        extent=[0, room_dim[0], 0, room_dim[1]],
        cmap='inferno'
    )

    plt.scatter(listener_a_pos[0], listener_a_pos[1], c='red', marker='x')
    plt.scatter(listener_b_pos[0], listener_b_pos[1], c='green', marker='x')

    plt.scatter(speakers_a[:,0], speakers_a[:,1], c='blue')
    plt.scatter(speakers_b[:,0], speakers_b[:,1], c='orange')

    plt.colorbar(im, label="Spatial Energy")
    plt.title(title)

plt.tight_layout()
plt.show()