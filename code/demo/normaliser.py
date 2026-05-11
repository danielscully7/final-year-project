import librosa
import soundfile as sf
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================

# Input audio files
INPUT_FILES = [
    "audio/beethoven.wav",
    "audio/piano2.wav",
    "audio/talkA.wav",
    "audio/talkB.wav",
    "audio/radio.wav",
    "audio/weather.wav",
    "audio/metallica.wav",
    "audio/quietriot.wav"
]

# Normalized output audio files
OUTPUT_FILES = [
    "audio/normalized_beethoven.wav",
    "audio/normalized_piano2.wav",
    "audio/normalized_talkA.wav",
    "audio/normalized_talkB.wav",
    "audio/normalized_radio.wav",
    "audio/normalized_weather.wav",
    "audio/normalized_metallica.wav",
    "audio/normalized_quietriot.wav"
]

TARGET_RMS = 0.12          # Same loudness for all files
TARGET_DURATION = None     

# =====================================================
# FUNCTIONS
# =====================================================

def rms_normalize(x, target_rms=0.12, eps=1e-12):
    """Normalize audio to the target RMS level."""
    rms = np.sqrt(np.mean(x**2) + eps)
    if rms < eps:
        return x
    return x * (target_rms / rms)

def match_duration(x, sr, target_seconds):
    """Trim or pad to make all files the same duration."""
    if target_seconds is None:
        return x

    target_len = int(sr * target_seconds)

    if len(x) > target_len:
        return x[:target_len]

    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))

    return x

# =====================================================
# MAIN PROCESSING
# =====================================================

def process_file(in_path, out_path):
    print(f"Processing: {in_path}")

    # Load audio
    x, sr = librosa.load(in_path, sr=None, mono=True)

    # Optional duration matching
    x = match_duration(x, sr, TARGET_DURATION)

    # RMS normalize
    x = rms_normalize(x, TARGET_RMS)

    # Save
    sf.write(out_path, x.astype(np.float32), sr)
    print(f"  → Saved: {out_path}\n")

def main():
    print("=== Normalizing 6 audio files ===")

    for in_path, out_path in zip(INPUT_FILES, OUTPUT_FILES):
        process_file(in_path, out_path)

    print("Done. All 6 files normalized successfully.")

if __name__ == "__main__":
    main()