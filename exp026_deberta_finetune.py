"""
EXP-026 — Fine-tuned DeBERTa-v3-small for Grammar Score Regression

Purpose
-------
Fine-tune microsoft/deberta-v3-small directly on Whisper transcripts to predict
the 0–5 MOS grammar score. Uses 10-fold OOF predictions so the resulting
ensemble evaluation is leakage-safe.

Designed for:
- Windows
- RTX 3050 4 GB VRAM
- Python 3.12
- transformers 5.x
- sentencepiece 0.2.x

Outputs
-------
exp026_deberta_ft_oof_predictions.csv
exp026_results.csv
exp026_fold_models/            (one small checkpoint per fold; can be deleted later)
"""

import gc
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import nnls
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEED = 42
N_SPLITS = 10

MODEL_NAME = "microsoft/deberta-v3-small"
CACHE_DIR = Path("exp025_deberta_cache")

MAX_LENGTH = 192
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 2

# Small LR is important for pretrained DeBERTa on only 769 examples.
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

# Save each fold so a crash does not destroy completed work.
SAVE_FOLD_MODELS = True

TRAIN_CSV = Path("train.csv")
TRANSCRIPT_CSV = Path("whisper_transcripts.csv")

OOF_OUT = Path("exp026_deberta_ft_oof_predictions.csv")
RESULTS_OUT = Path("exp026_results.csv")
MODEL_DIR = Path("exp026_fold_models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

print("=" * 72)
print("EXP-026 — FINE-TUNED DeBERTa-v3-small GRAMMAR REGRESSION")
print("=" * 72)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Samples: loading...")
print(f"Max length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRAD_ACCUM}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print()

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
train = pd.read_csv(TRAIN_CSV)
tx = pd.read_csv(TRANSCRIPT_CSV)

# Support the existing transcript filename column convention.
if "filename" not in train.columns:
    raise ValueError("train.csv must contain filename")
if "filename" not in tx.columns:
    raise ValueError("whisper_transcripts.csv must contain filename")
if "label" not in train.columns:
    raise ValueError("train.csv must contain label")
if "transcript" not in tx.columns:
    raise ValueError("whisper_transcripts.csv must contain transcript")

df = train[["filename", "label"]].merge(
    tx[["filename", "transcript"]],
    on="filename",
    how="left",
)

df["transcript"] = df["transcript"].fillna("").astype(str)
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"]).reset_index(drop=True)

texts = df["transcript"].tolist()
y = df["label"].to_numpy(dtype=np.float32)

print(f"Samples: {len(df)}")
print(f"Empty transcripts: {(df['transcript'].str.strip() == '').sum()}")
print(f"Label range: {y.min():.2f} – {y.max():.2f}")
print()

# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=str(CACHE_DIR),
    use_fast=False,
)
print("Tokenizer loaded.")
print()

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class DebertaRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=str(CACHE_DIR),
        )
        self.encoder = self.encoder.float()

        hidden = self.encoder.config.hidden_size

        self.dropout = torch.nn.Dropout(0.15)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        pred = self.regressor(self.dropout(pooled)).squeeze(-1)

        # Keep predictions in the valid MOS interval.
        return 5.0 * torch.sigmoid(pred)


# ---------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------
def tokenize_batch(batch_texts):
    return tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------
def train_fold(train_idx, val_idx, fold_num):
    print("=" * 72)
    print(f"FOLD {fold_num}/{N_SPLITS}")
    print(f"Train: {len(train_idx)} | Validation: {len(val_idx)}")
    print("=" * 72)

    model = DebertaRegression().to(DEVICE)

    # AdamW is available directly from torch and avoids Trainer API changes.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    n_train_batches = int(np.ceil(len(train_idx) / BATCH_SIZE))
    total_steps = int(np.ceil(n_train_batches / GRAD_ACCUM) * EPOCHS)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # CUDA AMP. This materially reduces activation memory on the 4 GB GPU.
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    best_state = None
    best_val_rmse = float("inf")

    for epoch in range(EPOCHS):
        model.train()

        shuffled = train_idx.copy()
        rng = np.random.default_rng(SEED + fold_num * 100 + epoch)
        rng.shuffle(shuffled)

        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        batches = 0

        for start in range(0, len(shuffled), BATCH_SIZE):
            ids = shuffled[start:start + BATCH_SIZE]

            batch_text = [texts[i] for i in ids]
            labels = torch.tensor(y[ids], dtype=torch.float32, device=DEVICE)

            enc = tokenize_batch(batch_text)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            with torch.amp.autocast(
    device_type="cuda",
    dtype=torch.float16,
    enabled=use_amp,
):
                pred = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                loss = torch.nn.functional.mse_loss(pred, labels)
                scaled_loss = loss / GRAD_ACCUM

            scaler.scale(scaled_loss).backward()

            if ((batches + 1) % GRAD_ACCUM == 0) or (
                start + BATCH_SIZE >= len(shuffled)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item()
            batches += 1

        # Validation
        model.eval()
        val_preds = []

        with torch.no_grad():
            for start in range(0, len(val_idx), BATCH_SIZE):
                ids = val_idx[start:start + BATCH_SIZE]
                enc = tokenize_batch([texts[i] for i in ids])
                enc = {k: v.to(DEVICE) for k, v in enc.items()}

                with torch.amp.autocast(
                    device_type="cuda" if use_amp else "cpu",
                    dtype=torch.float16 if use_amp else torch.float32,
                    enabled=use_amp,
                ):
                    pred = model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    )

                val_preds.extend(pred.float().cpu().numpy())

        val_preds = np.asarray(val_preds, dtype=np.float32)
        val_true = y[val_idx]

        rmse = np.sqrt(mean_squared_error(val_true, val_preds))

        print(
            f"  Epoch {epoch + 1}/{EPOCHS}: "
            f"train_loss={running_loss / max(1, batches):.5f} "
            f"val_RMSE={rmse:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    # Restore best epoch.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final OOF prediction.
    model.eval()
    final_preds = []

    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH_SIZE):
            ids = val_idx[start:start + BATCH_SIZE]
            enc = tokenize_batch([texts[i] for i in ids])
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            with torch.amp.autocast(
                device_type="cuda" if use_amp else "cpu",
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                pred = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )

            final_preds.extend(pred.float().cpu().numpy())

    final_preds = np.clip(np.asarray(final_preds, dtype=np.float32), 0.0, 5.0)

    if SAVE_FOLD_MODELS:
        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold": fold_num,
                "best_val_rmse": best_val_rmse,
                "config": {
                    "model": MODEL_NAME,
                    "max_length": MAX_LENGTH,
                    "epochs": EPOCHS,
                    "lr": LEARNING_RATE,
                },
            },
            MODEL_DIR / f"fold_{fold_num:02d}.pt",
        )

    # Free GPU memory before next fold.
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Best fold RMSE: {best_val_rmse:.5f}")
    return val_idx, final_preds


# ---------------------------------------------------------------------
# 10-fold OOF
# ---------------------------------------------------------------------
kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

oof = np.zeros(len(df), dtype=np.float32)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
    vi, pred = train_fold(train_idx, val_idx, fold)
    oof[vi] = pred

    fold_rmse = np.sqrt(mean_squared_error(y[vi], pred))
    fold_scores.append(fold_rmse)

    print(f"FOLD {fold}: OOF RMSE={fold_rmse:.6f}")
    print()

# ---------------------------------------------------------------------
# DeBERTa OOF metrics
# ---------------------------------------------------------------------
overall_rmse = np.sqrt(mean_squared_error(y, oof))
pearson = np.corrcoef(y, oof)[0, 1]

print("=" * 72)
print("EXP-026 DeBERTa FINE-TUNED OOF")
print("=" * 72)
print(f"RMSE:    {overall_rmse:.6f}")
print(f"Pearson: {pearson:.6f}")
print(f"Fold mean RMSE: {np.mean(fold_scores):.6f}")
print(f"Fold std RMSE:  {np.std(fold_scores):.6f}")
print()

# ---------------------------------------------------------------------
# Save DeBERTa OOF
# ---------------------------------------------------------------------
out = pd.DataFrame({
    "filename": df["filename"],
    "actual": y,
    "DeBERTa_FT": oof,
})
out.to_csv(OOF_OUT, index=False)

# ---------------------------------------------------------------------
# Blend against the strongest existing 10-fold OOF models
# ---------------------------------------------------------------------
BASE_OOF = Path("exp022_10fold_oof_predictions.csv")

if not BASE_OOF.exists():
    print(f"WARNING: {BASE_OOF} not found.")
    print("DeBERTa OOF was saved; blend evaluation skipped.")
    print(f"Saved: {OOF_OUT}")
    raise SystemExit(0)

base = pd.read_csv(BASE_OOF)

# Align by filename rather than row order.
merged = base.merge(
    out,
    on=["filename", "actual"],
    how="inner",
)

# These are the models used by the EXP-025 optimized ensemble.
preferred = [
    "Combined_PCA64_RidgeA100",
    "MultiWavLM_PCA128_Ridge",
    "Existing_MultiWavLM_ExtraTrees",
    "Existing_HGB",
]

available = [c for c in preferred if c in merged.columns]

if not available:
    raise ValueError(
        "Could not find expected EXP-022 OOF columns. "
        f"Available columns: {list(merged.columns)}"
    )

X = merged[available + ["DeBERTa_FT"]].to_numpy(dtype=np.float64)
target = merged["actual"].to_numpy(dtype=np.float64)

# Fixed weights from EXP-025, renormalized if a column is unavailable.
exp025_weights = {
    "Combined_PCA64_RidgeA100": 0.182364,
    "MultiWavLM_PCA128_Ridge": 0.327537,
    "Existing_MultiWavLM_ExtraTrees": 0.152415,
    "Existing_HGB": 0.112116,
    "DeBERTa_FT": 0.225568,
}

fixed = np.array(
    [exp025_weights[c] for c in available + ["DeBERTa_FT"]],
    dtype=np.float64,
)
fixed /= fixed.sum()

fixed_pred = X @ fixed
fixed_rmse = np.sqrt(mean_squared_error(target, fixed_pred))
fixed_pearson = np.corrcoef(target, fixed_pred)[0, 1]

# Non-negative least squares.
# This is still evaluated on OOF predictions; no target leakage from model fitting.
nnls_w, _ = nnls(X, target)
if nnls_w.sum() > 0:
    nnls_w /= nnls_w.sum()

nnls_pred = X @ nnls_w
nnls_rmse = np.sqrt(mean_squared_error(target, nnls_pred))
nnls_pearson = np.corrcoef(target, nnls_pred)[0, 1]

# Also evaluate clipped predictions because competition predictions are 0–5.
fixed_clip = np.clip(fixed_pred, 0.0, 5.0)
nnls_clip = np.clip(nnls_pred, 0.0, 5.0)

results = [
    {
        "experiment": "DeBERTa_FT_alone",
        "rmse": overall_rmse,
        "pearson": pearson,
        "rmse_clipped": np.sqrt(mean_squared_error(target, np.clip(oof, 0, 5))),
        "pearson_clipped": np.corrcoef(target, np.clip(oof, 0, 5))[0, 1],
    },
    {
        "experiment": "EXP025_fixed_weights_plus_DeBERTa_FT",
        "rmse": fixed_rmse,
        "pearson": fixed_pearson,
        "rmse_clipped": np.sqrt(mean_squared_error(target, fixed_clip)),
        "pearson_clipped": np.corrcoef(target, fixed_clip)[0, 1],
    },
    {
        "experiment": "NNLS_plus_DeBERTa_FT",
        "rmse": nnls_rmse,
        "pearson": nnls_pearson,
        "rmse_clipped": np.sqrt(mean_squared_error(target, nnls_clip)),
        "pearson_clipped": np.corrcoef(target, nnls_clip)[0, 1],
    },
]

results_df = pd.DataFrame(results).sort_values("rmse_clipped")

print("=" * 72)
print("EXP-026 + EXISTING WAVLM ENSEMBLE")
print("=" * 72)

print("\nFixed EXP-025 weights:")
for c, w in zip(available + ["DeBERTa_FT"], fixed):
    print(f"  {c:<42} {w:.6f}")

print("\nNNLS optimized weights:")
for c, w in zip(available + ["DeBERTa_FT"], nnls_w):
    print(f"  {c:<42} {w:.6f}")

print("\nResults:")
print(results_df.to_string(index=False))

results_df.to_csv(RESULTS_OUT, index=False)

print()
print("Saved:")
print(f"  {OOF_OUT.resolve()}")
print(f"  {RESULTS_OUT.resolve()}")
if SAVE_FOLD_MODELS:
    print(f"  {MODEL_DIR.resolve()}")
print()
print("DONE.")
