"""
EXP-027A — Whisper encoder embeddings for Grammar Scoring Engine.
RTX 3050-safe starter experiment.

Requires:
    python -m pip install -U openai-whisper

Outputs:
    exp027a_whisper_encoder_train.npy
    exp027a_whisper_encoder_test.npy
    exp027a_whisper_oof_predictions.csv
    exp027a_whisper_results.csv
"""

import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 10
WHISPER_MODEL = "base"
SR = 16000
CHUNK = 30 * SR

TRAIN_DIR = Path("train")
TEST_DIR = Path("test")
TRAIN_CSV = Path("train.csv")
TEST_CSV = Path("test.csv")

OUT_TRAIN = Path("exp027a_whisper_encoder_train.npy")
OUT_TEST = Path("exp027a_whisper_encoder_test.npy")
OUT_OOF = Path("exp027a_whisper_oof_predictions.csv")
OUT_RESULTS = Path("exp027a_whisper_results.csv")

PCA_DIMS = [32, 48, 64, 96, 128]
ALPHAS = [0.1, 1.0, 10.0, 100.0]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 72)
print("EXP-027A — WHISPER ENCODER EMBEDDINGS")
print("=" * 72)
print("This experiment extracts Whisper's internal audio representation.")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Whisper model: {WHISPER_MODEL}")
print()

try:
    import whisper
except ImportError:
    print("openai-whisper is not installed.")
    print("Run:")
    print("python -m pip install -U openai-whisper")
    raise SystemExit(1)

try:
    import soundfile as sf
except ImportError:
    print("soundfile is not installed.")
    print("Run: python -m pip install soundfile")
    raise SystemExit(1)

train_df = pd.read_csv(TRAIN_CSV)[["filename", "label"]]
test_df = pd.read_csv(TEST_CSV)[["filename"]]
y = pd.to_numeric(train_df["label"], errors="coerce").to_numpy(np.float32)

print(f"Train: {len(train_df)}")
print(f"Test:  {len(test_df)}")
print()

print("Loading Whisper...")
whisper_model = whisper.load_model(
    WHISPER_MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
whisper_model.eval()

encoder_dim = int(whisper_model.dims.n_audio_state)
print(f"Whisper encoder hidden size: {encoder_dim}")
print()

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    if sr != SR:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(int(sr), SR)
        audio = resample_poly(
            audio, SR // g, int(sr) // g
        ).astype(np.float32)

    if len(audio) and np.max(np.abs(audio)) > 1:
        audio = audio / np.max(np.abs(audio))

    return audio

@torch.inference_mode()
def encode_chunk(audio):
    mel = whisper.log_mel_spectrogram(
        audio,
        n_mels=whisper_model.dims.n_mels,
    ).to(whisper_model.device)

    if mel.ndim == 2:
        mel = mel.unsqueeze(0)

    hidden = whisper_model.encoder(mel)[0]
    return hidden.float().cpu().numpy()

def extract_one(path):
    audio = load_audio(path)

    if len(audio) == 0:
        audio = np.zeros(CHUNK, dtype=np.float32)

    chunk_features = []

    for start in range(0, len(audio), CHUNK):
        chunk = audio[start:start + CHUNK]

        if len(chunk) < SR:
            continue

        if len(chunk) < CHUNK:
            padded = np.zeros(CHUNK, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded

        h = encode_chunk(chunk)

        # Frame-level statistics.
        chunk_features.append(np.concatenate([
            h.mean(axis=0),
            h.std(axis=0),
            h.min(axis=0),
            h.max(axis=0),
        ]))

    if not chunk_features:
        chunk_features = [
            np.zeros(encoder_dim * 4, dtype=np.float32)
        ]

    z = np.asarray(chunk_features, dtype=np.float32)

    # Global statistics across temporal chunks.
    feat = np.concatenate([
        z.mean(axis=0),
        z.std(axis=0),
    ]).astype(np.float32)

    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat /= norm

    return feat

def extract_split(df, audio_dir, out_path, name):
    if out_path.exists():
        old = np.load(out_path)
        if len(old) == len(df):
            print(f"Reusing {out_path} -> {old.shape}")
            return old

    print("=" * 72)
    print(f"EXTRACTING {name.upper()}")
    print("=" * 72)

    features = []
    errors = []

    for i, filename in enumerate(df["filename"]):
        try:
            features.append(
                extract_one(audio_dir / str(filename))
            )
        except Exception as exc:
            print(f"ERROR {filename}: {exc}")
            features.append(
                np.zeros(encoder_dim * 8, dtype=np.float32)
            )
            errors.append(str(filename))

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  {i + 1}/{len(df)}")

        if torch.cuda.is_available() and (i + 1) % 25 == 0:
            torch.cuda.empty_cache()

    X = np.asarray(features, dtype=np.float32)
    np.save(out_path, X)

    print(f"{name} shape: {X.shape}")
    print(f"Errors: {len(errors)}")
    print(f"Saved: {out_path}")
    print()

    return X

X_train = extract_split(
    train_df, TRAIN_DIR, OUT_TRAIN, "train"
)
X_test = extract_split(
    test_df, TEST_DIR, OUT_TEST, "test"
)

# Release Whisper before sklearn.
del whisper_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("=" * 72)
print("WHISPER ENCODER OOF MODEL SEARCH")
print("=" * 72)

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

results = []
best_oof = None
best_name = None
best_rmse = float("inf")

for pca_dim in PCA_DIMS:
    for alpha in ALPHAS:
        oof = np.zeros(len(y), dtype=np.float32)

        for tr, va in kf.split(X_train):
            ncomp = min(
                pca_dim,
                len(tr) - 1,
                X_train.shape[1],
            )

            pipe = make_pipeline(
                StandardScaler(),
                PCA(
                    n_components=ncomp,
                    random_state=SEED,
                ),
                Ridge(alpha=alpha),
            )

            pipe.fit(X_train[tr], y[tr])
            oof[va] = pipe.predict(X_train[va])

        clip = np.clip(oof, 0, 5)
        rmse = np.sqrt(mean_squared_error(y, oof))
        pearson = np.corrcoef(y, oof)[0, 1]
        rmse_clip = np.sqrt(mean_squared_error(y, clip))
        pearson_clip = np.corrcoef(y, clip)[0, 1]

        name = f"Whisper_PCA{pca_dim}_RidgeA{alpha}"

        results.append({
            "experiment": name,
            "rmse": rmse,
            "pearson": pearson,
            "rmse_clipped": rmse_clip,
            "pearson_clipped": pearson_clip,
        })

        print(
            f"{name:<35} "
            f"RMSE={rmse:.6f} "
            f"Pearson={pearson:.6f} "
            f"clipRMSE={rmse_clip:.6f}"
        )

        if rmse_clip < best_rmse:
            best_rmse = rmse_clip
            best_oof = oof.copy()
            best_name = name

print()
print("Testing raw Whisper ExtraTrees...")

et_oof = np.zeros(len(y), dtype=np.float32)

for tr, va in kf.split(X_train):
    et = ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.5,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    )
    et.fit(X_train[tr], y[tr])
    et_oof[va] = et.predict(X_train[va])

et_clip = np.clip(et_oof, 0, 5)
et_rmse = np.sqrt(mean_squared_error(y, et_oof))
et_p = np.corrcoef(y, et_oof)[0, 1]
et_clip_rmse = np.sqrt(mean_squared_error(y, et_clip))
et_clip_p = np.corrcoef(y, et_clip)[0, 1]

results.append({
    "experiment": "Whisper_Raw_ExtraTrees",
    "rmse": et_rmse,
    "pearson": et_p,
    "rmse_clipped": et_clip_rmse,
    "pearson_clipped": et_clip_p,
})

print(
    f"Whisper_Raw_ExtraTrees: "
    f"RMSE={et_rmse:.6f} "
    f"Pearson={et_p:.6f} "
    f"clipRMSE={et_clip_rmse:.6f}"
)

if et_clip_rmse < best_rmse:
    best_oof = et_oof.copy()
    best_rmse = et_clip_rmse
    best_name = "Whisper_Raw_ExtraTrees"

results_df = pd.DataFrame(results).sort_values(
    "rmse_clipped"
)
results_df.to_csv(OUT_RESULTS, index=False)

oof_df = pd.DataFrame({
    "filename": train_df["filename"],
    "actual": y,
    "prediction": best_oof,
})
oof_df.to_csv(OUT_OOF, index=False)

print()
print("=" * 72)
print("BEST WHISPER ENCODER MODEL")
print("=" * 72)
print(results_df.head(10).to_string(index=False))
print(f"\nBest: {best_name}")
print(f"Clipped RMSE: {best_rmse:.6f}")

# Compare with existing 10-fold OOF predictions.
base_path = Path("exp022_10fold_oof_predictions.csv")

if base_path.exists():
    print()
    print("=" * 72)
    print("WHISPER + EXISTING WAVLM PAIRWISE BLENDS")
    print("=" * 72)

    base = pd.read_csv(base_path)

    merged = base.merge(
        oof_df,
        on=["filename", "actual"],
        how="inner",
    )

    candidates = [
        "Combined_PCA64_RidgeA100",
        "MultiWavLM_PCA128_Ridge",
        "Existing_MultiWavLM_ExtraTrees",
        "Existing_HGB",
    ]

    for col in [c for c in candidates if c in merged]:
        a = merged[col].to_numpy()
        b = merged["prediction"].to_numpy()
        actual = merged["actual"].to_numpy()

        best = None

        for w in np.arange(0, 1.0001, 0.025):
            pred = np.clip(
                (1 - w) * a + w * b,
                0, 5,
            )
            rmse = np.sqrt(
                mean_squared_error(actual, pred)
            )
            p = np.corrcoef(actual, pred)[0, 1]

            if best is None or rmse < best[0]:
                best = (rmse, p, w)

        print(
            f"{col:<42} "
            f"Whisper={best[2]:.3f} "
            f"RMSE={best[0]:.6f} "
            f"Pearson={best[1]:.6f}"
        )

print()
print("Saved:")
print(OUT_TRAIN)
print(OUT_TEST)
print(OUT_OOF)
print(OUT_RESULTS)
print("DONE.")
