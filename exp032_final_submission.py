
"""
EXP-032 — FINAL 216-ROW KAGGLE TEST INFERENCE

Uses test.csv ONLY as the authoritative test set.
sample_submission.csv is never read.

Final architecture = EXP-029 best stack:
  1. WavLM_Combined
  2. WavLM_Ridge
  3. WavLM_ExtraTrees
  4. HGB
  5. MiniLM
  6. DeBERTa_FT
  7. Whisper_Base

Meta model:
  ElasticNet(alpha=0.01, l1_ratio=0.10)

The seven base models are trained on all 769 training samples.
The meta model is fitted on the 769-row OOF base-prediction matrix.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gc
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"

OUT_BASE = ROOT / "exp032_test_base_predictions.csv"
OUT_SUBMISSION = ROOT / "submission_exp032.csv"

SEED = 42


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")


def load_features(path, names):
    require(path)
    df = pd.read_csv(path)
    if "filename" not in df.columns:
        raise RuntimeError(f"{path.name} has no filename column.")
    df["filename"] = df["filename"].astype(str)
    df = df.drop_duplicates("filename").set_index("filename")
    missing = [n for n in names if n not in df.index]
    if missing:
        raise RuntimeError(
            f"{path.name} is missing {len(missing)} required filenames."
        )
    cols = [
        c for c in df.columns
        if c not in {"filename", "label", "actual", "target"}
    ]
    return df.loc[names, cols].astype(float).values


def load_npy(path, expected_rows, name):
    require(path)
    x = np.load(path)
    if x.shape[0] != expected_rows:
        raise RuntimeError(
            f"{name}: expected {expected_rows} rows, got {x.shape[0]}."
        )
    return x.astype(np.float32)


def pca_ridge(Xtr, y, Xte, n_components, alpha):
    scaler = StandardScaler()
    a = scaler.fit_transform(Xtr)
    b = scaler.transform(Xte)

    pca = PCA(
        n_components=n_components,
        random_state=SEED,
    )
    a = pca.fit_transform(a)
    b = pca.transform(b)

    model = Ridge(alpha=alpha)
    model.fit(a, y)
    return model.predict(b)


def load_oof_column(path, column, names):
    require(path)
    df = pd.read_csv(path)
    if "filename" not in df.columns or column not in df.columns:
        raise RuntimeError(
            f"{path.name}: expected filename and {column}."
        )
    df["filename"] = df["filename"].astype(str)
    df = df.drop_duplicates("filename").set_index("filename")
    missing = [n for n in names if n not in df.index]
    if missing:
        raise RuntimeError(
            f"{path.name}: missing {len(missing)} OOF filenames."
        )
    return df.loc[names, column].astype(float).values


def deberta_test_predictions(test_names):
    """
    Generate test predictions from the 10 saved EXP-026 DeBERTa checkpoints.

    EXP-026 saved checkpoints as:
        exp026_fold_models/fold_01.pt
        ...
        exp026_fold_models/fold_10.pt

    Each checkpoint contains:
        {
            "model_state_dict": ...,
            "fold": ...,
            "best_val_rmse": ...,
            "config": ...
        }
    """

    print("  Loading EXP-026 DeBERTa fold checkpoints...")

    # ------------------------------------------------------------------
    # Exact EXP-026 configuration
    # ------------------------------------------------------------------
    MODEL_NAME = "microsoft/deberta-v3-small"
    CACHE_DIR = ROOT / "exp025_deberta_cache"
    MAX_LENGTH = 192

    MODEL_DIR = ROOT / "exp026_fold_models"

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"DeBERTa checkpoint directory not found: {MODEL_DIR}"
        )

    fold_files = sorted(MODEL_DIR.glob("fold_*.pt"))

    if len(fold_files) != 10:
        raise RuntimeError(
            f"Expected 10 DeBERTa fold checkpoints, "
            f"found {len(fold_files)} under {MODEL_DIR}"
        )

    print(f"  Found {len(fold_files)} fold checkpoints.")

    # ------------------------------------------------------------------
    # Load test transcripts
    # ------------------------------------------------------------------
    test_tx_path = ROOT / "test_whisper_transcripts.csv"

    if not test_tx_path.exists():
        raise FileNotFoundError(
            f"Missing test transcript file: {test_tx_path}"
        )

    test_tx = pd.read_csv(test_tx_path)

    if "filename" not in test_tx.columns:
        raise ValueError(
            "test_whisper_transcripts.csv must contain 'filename'"
        )

    if "transcript" not in test_tx.columns:
        raise ValueError(
            "test_whisper_transcripts.csv must contain 'transcript'"
        )

    transcript_map = dict(
        zip(
            test_tx["filename"].astype(str),
            test_tx["transcript"].fillna("").astype(str),
        )
    )

    texts = [
        transcript_map.get(str(name), "")
        for name in test_names
    ]

    if len(texts) != len(test_names):
        raise RuntimeError("Transcript/test filename count mismatch.")

    # ------------------------------------------------------------------
    # Exact EXP-026 tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        use_fast=False,
    )

    # ------------------------------------------------------------------
    # Exact EXP-026 model architecture
    # ------------------------------------------------------------------
    class DebertaRegression(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = AutoModel.from_pretrained(
                MODEL_NAME,
                cache_dir=str(CACHE_DIR),
            )

            # EXP-026 explicitly converted encoder to FP32.
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

            pooled = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1.0)
            )

            pred = self.regressor(
                self.dropout(pooled)
            ).squeeze(-1)

            # Exact EXP-026 output constraint.
            return 5.0 * torch.sigmoid(pred)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"  DeBERTa device: {device}")

    fold_predictions = []

    for fold_file in fold_files:

        print(f"    Loading {fold_file.name}...")

        # Checkpoint contains only tensors + simple metadata.
        checkpoint = torch.load(
            fold_file,
            map_location="cpu",
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"{fold_file} does not contain "
                "'model_state_dict'."
            )

        model = DebertaRegression()

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        model = model.to(device)
        model.eval()

        predictions = []

        with torch.no_grad():

            # EXP-026 used BATCH_SIZE = 2.
            for start in range(0, len(texts), 2):

                batch_texts = texts[start:start + 2]

                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )

                enc = {
                    k: v.to(device)
                    for k, v in enc.items()
                }

                pred = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )

                predictions.extend(
                    pred.float().cpu().numpy()
                )

        predictions = np.asarray(
            predictions,
            dtype=np.float32,
        )

        predictions = np.clip(
            predictions,
            0.0,
            5.0,
        )

        fold_predictions.append(predictions)

        print(
            f"      fold={checkpoint.get('fold', '?')} "
            f"best_val_rmse="
            f"{checkpoint.get('best_val_rmse', float('nan')):.6f} "
            f"mean={predictions.mean():.4f}"
        )

        # Important for the 4 GB RTX 3050:
        # release one DeBERTa model before loading the next.
        del model
        del checkpoint

        if device.type == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

    # ------------------------------------------------------------------
    # Average the 10 fold predictions
    # ------------------------------------------------------------------
    fold_predictions = np.stack(
        fold_predictions,
        axis=0,
    )

    if fold_predictions.shape != (
        10,
        len(test_names),
    ):
        raise RuntimeError(
            "Unexpected DeBERTa prediction shape: "
            f"{fold_predictions.shape}"
        )

    final_predictions = fold_predictions.mean(axis=0)

    final_predictions = np.clip(
        final_predictions,
        0.0,
        5.0,
    ).astype(np.float32)

    print(
        "  DeBERTa_FT test predictions:",
        final_predictions.shape,
    )

    print(
        f"  DeBERTa_FT stats: "
        f"mean={final_predictions.mean():.6f}, "
        f"std={final_predictions.std():.6f}, "
        f"min={final_predictions.min():.6f}, "
        f"max={final_predictions.max():.6f}"
    )

    return final_predictions

def main():
    print("=" * 78)
    print("EXP-032 — FINAL 216-ROW KAGGLE TEST INFERENCE")
    print("=" * 78)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    if "filename" not in train.columns or "label" not in train.columns:
        raise RuntimeError("train.csv must contain filename and label.")

    if "filename" not in test.columns:
        raise RuntimeError("test.csv must contain filename.")

    train_names = train["filename"].astype(str).tolist()
    test_names = test["filename"].astype(str).tolist()
    y = train["label"].astype(float).values

    if len(train_names) != 769:
        raise RuntimeError(f"Expected 769 train rows, got {len(train_names)}")
    if len(test_names) != 216:
        raise RuntimeError(f"Expected 216 test rows, got {len(test_names)}")

    if len(set(train_names)) != len(train_names):
        raise RuntimeError("Duplicate train filenames.")
    if len(set(test_names)) != len(test_names):
        raise RuntimeError("Duplicate test filenames.")

    print(f"Train rows: {len(train_names)}")
    print(f"Test rows : {len(test_names)}")
    print("Authoritative test source: test.csv")
    print("sample_submission.csv: NOT USED")

    # ---------------------------------------------------------------
    # Existing 240 features = acoustic 195 + linguistic 45
    # ---------------------------------------------------------------
    print("\nLoading existing acoustic + linguistic features...")

    A_tr = load_features(
        ROOT / "acoustic_features.csv",
        train_names,
    )
    A_te = load_features(
        ROOT / "test_acoustic_features.csv",
        test_names,
    )

    L_tr = load_features(
        ROOT / "linguistic_features.csv",
        train_names,
    )
    L_te = load_features(
        ROOT / "test_linguistic_features.csv",
        test_names,
    )

    EX_tr = np.hstack([A_tr, L_tr])
    EX_te = np.hstack([A_te, L_te])

    print("  Existing:", EX_tr.shape, EX_te.shape)

    # ---------------------------------------------------------------
    # WavLM multi-layer
    # ---------------------------------------------------------------
    print("\nLoading WavLM multi-layer embeddings...")

    W_tr = load_npy(
        ROOT / "exp020_wavlm_multilayer_train.npy",
        769,
        "WavLM train",
    )
    W_te = load_npy(
        ROOT / "exp020_wavlm_multilayer_test.npy",
        216,
        "WavLM test",
    )

    print("  WavLM:", W_tr.shape, W_te.shape)

    # ---------------------------------------------------------------
    # 1. WavLM Combined: Existing + MultiWavLM PCA64 Ridge A100
    # ---------------------------------------------------------------
    print("\n[1/7] WavLM_Combined")
    C_tr = np.hstack([EX_tr, W_tr])
    C_te = np.hstack([EX_te, W_te])

    p_combined = pca_ridge(
        C_tr, y, C_te,
        n_components=64,
        alpha=100,
    )

    # ---------------------------------------------------------------
    # 2. WavLM Ridge: MultiWavLM PCA128 Ridge
    # ---------------------------------------------------------------
    print("[2/7] WavLM_Ridge")

    p_wavlm_ridge = pca_ridge(
        W_tr, y, W_te,
        n_components=128,
        alpha=1,
    )

    # ---------------------------------------------------------------
    # 3. WavLM ExtraTrees
    # ---------------------------------------------------------------
    print("[3/7] WavLM_ExtraTrees")

    et = ExtraTreesRegressor(
        n_estimators=1000,
        max_features=0.7,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )
    et.fit(C_tr, y)
    p_wavlm_et = et.predict(C_te)

    # ---------------------------------------------------------------
    # 4. HGB on existing 240
    # ---------------------------------------------------------------
    print("[4/7] HGB")

    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=SEED,
    )
    hgb.fit(EX_tr, y)
    p_hgb = hgb.predict(EX_te)

    # ---------------------------------------------------------------
    # 5. MiniLM Ridge
    # ---------------------------------------------------------------
    print("[5/7] MiniLM")

    mini_tr = load_npy(
        ROOT / "exp013_sentence_embeddings.npy",
        769,
        "MiniLM train",
    )
    mini_te = load_npy(
        ROOT / "exp016_test_sentence_embeddings.npy",
        216,
        "MiniLM test",
    )

    mini = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10),
    )
    mini.fit(mini_tr, y)
    p_minilm = mini.predict(mini_te)

    # ---------------------------------------------------------------
    # 6. DeBERTa FT
    # ---------------------------------------------------------------
    print("[6/7] DeBERTa_FT")
    p_deberta = deberta_test_predictions(test_names)

    # ---------------------------------------------------------------
    # 7. Whisper Base: PCA128 Ridge A100
    # ---------------------------------------------------------------
    print("[7/7] Whisper_Base")

    wh_tr = load_npy(
        ROOT / "exp027a_whisper_encoder_train.npy",
        769,
        "Whisper train",
    )
    wh_te = load_npy(
        ROOT / "exp027a_whisper_encoder_test.npy",
        216,
        "Whisper test",
    )

    p_whisper = pca_ridge(
        wh_tr, y, wh_te,
        n_components=128,
        alpha=100,
    )

    # ---------------------------------------------------------------
    # Test base matrix
    # ---------------------------------------------------------------
    base_cols = [
        "WavLM_Combined",
        "WavLM_Ridge",
        "WavLM_ExtraTrees",
        "HGB",
        "MiniLM",
        "DeBERTa_FT",
        "Whisper_Base",
    ]

    X_test_meta = np.column_stack([
        p_combined,
        p_wavlm_ridge,
        p_wavlm_et,
        p_hgb,
        p_minilm,
        p_deberta,
        p_whisper,
    ]).astype(float)

    if X_test_meta.shape != (216, 7):
        raise RuntimeError(
            f"Expected test meta matrix (216, 7), got {X_test_meta.shape}"
        )

    if not np.isfinite(X_test_meta).all():
        raise RuntimeError("Non-finite base prediction detected.")

    print("\nTest meta matrix:", X_test_meta.shape)

    # ---------------------------------------------------------------
    # Full training OOF matrix for meta model
    # ---------------------------------------------------------------
    print("\nLoading 769-row OOF matrix for meta model...")

    oof_spec = [
        (
            "exp022_10fold_oof_predictions.csv",
            "Combined_PCA64_RidgeA100",
        ),
        (
            "exp022_10fold_oof_predictions.csv",
            "MultiWavLM_PCA128_Ridge",
        ),
        (
            "exp022_10fold_oof_predictions.csv",
            "Existing_MultiWavLM_ExtraTrees",
        ),
        (
            "exp022_10fold_oof_predictions.csv",
            "Existing_HGB",
        ),
        (
            "exp022_10fold_oof_predictions.csv",
            "MiniLM_Ridge",
        ),
        (
            "exp026_deberta_ft_oof_predictions.csv",
            "DeBERTa_FT",
        ),
        (
            "exp027a_whisper_oof_predictions.csv",
            "prediction",
        ),
    ]

    # Use the train.csv order as the authoritative OOF order.
    oof_parts = []

    for filename, col in oof_spec:
        path = ROOT / filename
        require(path)

        df = pd.read_csv(path)
        if "filename" not in df.columns or col not in df.columns:
            raise RuntimeError(
                f"{filename} missing filename/{col}"
            )

        df["filename"] = df["filename"].astype(str)
        df = df.drop_duplicates("filename").set_index("filename")

        missing = [n for n in train_names if n not in df.index]
        if missing:
            raise RuntimeError(
                f"{filename} missing {len(missing)} training filenames."
            )

        oof_parts.append(
            df.loc[train_names, col].astype(float).values
        )

    X_oof_meta = np.column_stack(oof_parts)

    if X_oof_meta.shape != (769, 7):
        raise RuntimeError(
            f"Expected OOF meta matrix (769, 7), got {X_oof_meta.shape}"
        )

    if not np.isfinite(X_oof_meta).all():
        raise RuntimeError("Non-finite OOF prediction detected.")

    # ---------------------------------------------------------------
    # EXP-030 best meta model
    # ---------------------------------------------------------------
    print("\nFitting final meta model:")
    print("  ElasticNet alpha=0.01")
    print("  ElasticNet l1_ratio=0.10")

    meta = ElasticNet(
        alpha=0.01,
        l1_ratio=0.10,
        max_iter=100000,
        random_state=SEED,
    )

    meta.fit(X_oof_meta, y)

    print("\nMeta coefficients:")
    for name, coef in zip(base_cols, meta.coef_):
        print(f"  {name:20s} {coef:+.8f}")
    print(f"  intercept             {meta.intercept_:+.8f}")

    final_pred = meta.predict(X_test_meta)
    final_pred = np.clip(final_pred, 0.0, 5.0)

    if not np.isfinite(final_pred).all():
        raise RuntimeError("Final predictions contain non-finite values.")

    # ---------------------------------------------------------------
    # Base prediction audit
    # ---------------------------------------------------------------
    base_df = pd.DataFrame({
        "filename": test_names,
    })

    for i, name in enumerate(base_cols):
        base_df[name] = X_test_meta[:, i]

    base_df["final_prediction"] = final_pred
    base_df.to_csv(OUT_BASE, index=False)

    # ---------------------------------------------------------------
    # Final submission: exactly test.csv
    # ---------------------------------------------------------------
    submission = pd.DataFrame({
        "filename": test_names,
        "label": final_pred,
    })

    if len(submission) != 216:
        raise RuntimeError(
            f"Submission has {len(submission)} rows, expected 216."
        )

    if submission["filename"].tolist() != test_names:
        raise RuntimeError("Submission ordering differs from test.csv.")

    if submission["filename"].duplicated().any():
        raise RuntimeError("Duplicate submission filenames.")

    if not np.isfinite(submission["label"]).all():
        raise RuntimeError("Non-finite submission predictions.")

    if submission["label"].min() < 0 or submission["label"].max() > 5:
        raise RuntimeError("Predictions outside [0, 5].")

    submission.to_csv(OUT_SUBMISSION, index=False)

    print("\n" + "=" * 78)
    print("EXP-032 COMPLETE")
    print("=" * 78)
    print(f"Train rows used: {len(train_names)}")
    print(f"Test rows predicted: {len(test_names)}")
    print(f"Prediction mean: {final_pred.mean():.6f}")
    print(f"Prediction std : {final_pred.std():.6f}")
    print(f"Prediction min : {final_pred.min():.6f}")
    print(f"Prediction max : {final_pred.max():.6f}")

    print("\nSaved:")
    print(f"  {OUT_BASE.name}")
    print(f"  {OUT_SUBMISSION.name}")

    print("\nFirst 10 submission rows:")
    print(submission.head(10).to_string(index=False))

    print("\nCONFIRMED:")
    print("  - 216 rows")
    print("  - filenames taken directly from test.csv")
    print("  - sample_submission.csv was NOT used")


if __name__ == "__main__":
    main()
