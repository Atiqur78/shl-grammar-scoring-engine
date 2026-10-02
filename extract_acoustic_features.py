from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\kagle_project\Dataset_Final")
TRAIN_DIR = ROOT / "train"

train_df = pd.read_csv(ROOT / "train.csv")


def summarize(x, prefix):
    """Return robust summary statistics for a frame-level feature."""
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_q25": float(np.percentile(x, 25)),
        f"{prefix}_q75": float(np.percentile(x, 75)),
    }


def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)

    features = {}

    # ---------------------------------------------------------
    # Basic audio properties
    # ---------------------------------------------------------

    duration = len(y) / sr

    features["duration"] = duration
    features["audio_rms"] = float(np.sqrt(np.mean(y ** 2)))
    features["audio_abs_mean"] = float(np.mean(np.abs(y)))

    # ---------------------------------------------------------
    # Frame-level features
    # ---------------------------------------------------------

    hop_length = 512
    n_fft = 2048

    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    # ---------------------------------------------------------
    # Aggregate spectral / energy features
    # ---------------------------------------------------------

    features.update(summarize(rms, "rms"))
    features.update(summarize(zcr, "zcr"))
    features.update(summarize(centroid, "centroid"))
    features.update(summarize(bandwidth, "bandwidth"))
    features.update(summarize(rolloff, "rolloff"))
    features.update(summarize(flatness, "flatness"))

    # ---------------------------------------------------------
    # MFCC
    # ---------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length
    )

    for i in range(mfcc.shape[0]):
        features.update(
            summarize(mfcc[i], f"mfcc_{i+1}")
        )

    # ---------------------------------------------------------
    # Spectral contrast
    # ---------------------------------------------------------

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    for i in range(contrast.shape[0]):
        features.update(
            summarize(contrast[i], f"contrast_{i+1}")
        )

    # ---------------------------------------------------------
    # Silence / low-energy statistics
    # ---------------------------------------------------------

    rms_threshold = np.percentile(rms, 20)

    low_energy_ratio = np.mean(rms <= rms_threshold)

    features["low_energy_ratio"] = float(low_energy_ratio)

    # Number of transitions between low/high energy regions
    low_energy = rms <= rms_threshold

    transitions = np.sum(
        low_energy[1:] != low_energy[:-1]
    )

    features["energy_transitions"] = float(transitions)

    # ---------------------------------------------------------
    # Pitch
    # ---------------------------------------------------------

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length
    )

    valid_f0 = f0[np.isfinite(f0)]

    if len(valid_f0) > 0:
        features["pitch_mean"] = float(np.mean(valid_f0))
        features["pitch_std"] = float(np.std(valid_f0))
        features["pitch_median"] = float(np.median(valid_f0))
        features["pitch_min"] = float(np.min(valid_f0))
        features["pitch_max"] = float(np.max(valid_f0))
        features["pitch_range"] = float(
            np.max(valid_f0) - np.min(valid_f0)
        )
    else:
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_median"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0
        features["pitch_range"] = 0.0

    features["voiced_ratio"] = float(
        np.mean(voiced_flag)
    )

    features["voiced_probability_mean"] = float(
        np.nanmean(voiced_prob)
    )

    return features


# =============================================================
# Extract all training features
# =============================================================

results = []

for _, row in tqdm(
    train_df.iterrows(),
    total=len(train_df)
):

    filename = row["filename"]
    label = row["label"]

    path = TRAIN_DIR / filename

    try:
        features = extract_features(path)

        features["filename"] = filename
        features["label"] = label

        results.append(features)

    except Exception as e:
        print(f"ERROR: {filename}: {e}")


features_df = pd.DataFrame(results)

# Put filename/label first
columns = (
    ["filename", "label"]
    + [
        c for c in features_df.columns
        if c not in ["filename", "label"]
    ]
)

features_df = features_df[columns]

output_path = ROOT / "acoustic_features.csv"

features_df.to_csv(
    output_path,
    index=False
)

print()
print("=" * 70)
print("ACOUSTIC FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print("Shape:", features_df.shape)

print()
print("Output:")
print(output_path)

print()
print("Missing values:")
print(
    features_df.isna().sum().sum()
)

print()
print("Top correlations with label:")

numeric_cols = features_df.select_dtypes(
    include=np.number
).columns

correlations = (
    features_df[numeric_cols]
    .corr()["label"]
    .drop("label")
    .sort_values(
        key=np.abs,
        ascending=False
    )
)

print(correlations.head(25))