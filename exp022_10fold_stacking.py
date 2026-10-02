
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from scipy.optimize import minimize

BASE = Path(r"D:\kagle_project\Dataset_Final")

TRAIN_CSV = BASE / "train.csv"
ACOUSTIC_CSV = BASE / "acoustic_features.csv"
LING_CSV = BASE / "linguistic_features.csv"
WAVLM_MULTI = BASE / "exp020_wavlm_multilayer_train.npy"
WAVLM_TEST = BASE / "exp020_wavlm_multilayer_test.npy"
MINILM_TRAIN = BASE / "exp013_sentence_embeddings.npy"

OUT_OOF = BASE / "exp022_10fold_oof_predictions.csv"
OUT_RESULTS = BASE / "exp022_results.csv"
OUT_WEIGHTS = BASE / "exp022_blend_weights.csv"

SEED = 42
N_SPLITS = 10

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

def pearson(y, p):
    return float(np.corrcoef(y, p)[0, 1])

def report(name, y, p):
    p_clip = np.clip(p, 0.0, 5.0)
    print(f"{name}")
    print(f"  raw     RMSE={rmse(y,p):.6f} Pearson={pearson(y,p):.6f}")
    print(f"  clipped RMSE={rmse(y,p_clip):.6f} Pearson={pearson(y,p_clip):.6f}")
    return p_clip

def load_existing_features():
    ac = pd.read_csv(ACOUSTIC_CSV)
    li = pd.read_csv(LING_CSV)

    if "filename" not in ac.columns or "filename" not in li.columns:
        raise ValueError("Expected filename column in acoustic/linguistic feature files.")

    ac = ac.set_index("filename")
    li = li.set_index("filename")

    common = ac.index.intersection(li.index)
    train = pd.read_csv(TRAIN_CSV)

    # Preserve train.csv order exactly.
    train = train[train["filename"].isin(common)].copy()
    train = train.set_index("filename")

    ac = ac.loc[train.index]
    li = li.loc[train.index]

    # Drop labels if present; train.csv is authoritative.
    ac = ac.drop(columns=["label"], errors="ignore")
    li = li.drop(columns=["label"], errors="ignore")

    X_existing = pd.concat([ac, li], axis=1).astype(np.float32).values
    y = train["label"].astype(float).values
    filenames = train.index.to_numpy()

    return filenames, y, X_existing

def pca_ridge_oof(X, y, folds, n_components, alpha, name):
    pred = np.zeros(len(y), dtype=np.float64)

    for fold, (tr, va) in enumerate(folds, 1):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(
                n_components=n_components,
                random_state=SEED
            )),
            ("ridge", Ridge(alpha=alpha))
        ])
        pipe.fit(X[tr], y[tr])
        pred[va] = pipe.predict(X[va])

        print(
            f"  {name}: fold {fold:02d}/{N_SPLITS} "
            f"RMSE={rmse(y[va], pred[va]):.5f}"
        )

    return pred

def ridge_oof(X, y, folds, alpha, name):
    pred = np.zeros(len(y), dtype=np.float64)

    for fold, (tr, va) in enumerate(folds, 1):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])
        pipe.fit(X[tr], y[tr])
        pred[va] = pipe.predict(X[va])

    print(f"{name}: RMSE={rmse(y,pred):.6f} Pearson={pearson(y,pred):.6f}")
    return pred

def extra_trees_oof(X, y, folds, name, params):
    pred = np.zeros(len(y), dtype=np.float64)

    for fold, (tr, va) in enumerate(folds, 1):
        model = ExtraTreesRegressor(
            random_state=SEED + fold,
            n_jobs=-1,
            **params
        )
        model.fit(X[tr], y[tr])
        pred[va] = model.predict(X[va])
        print(
            f"  {name}: fold {fold:02d}/{N_SPLITS} "
            f"RMSE={rmse(y[va],pred[va]):.5f}"
        )

    print(f"{name}: RMSE={rmse(y,pred):.6f} Pearson={pearson(y,pred):.6f}")
    return pred

def hgb_oof(X, y, folds, name):
    pred = np.zeros(len(y), dtype=np.float64)

    for fold, (tr, va) in enumerate(folds, 1):
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            min_samples_leaf=12,
            random_state=SEED + fold
        )
        model.fit(X[tr], y[tr])
        pred[va] = model.predict(X[va])

    print(f"{name}: RMSE={rmse(y,pred):.6f} Pearson={pearson(y,pred):.6f}")
    return pred

def optimize_nonnegative_weights(P, y):
    n_models = P.shape[1]

    def objective(w):
        pred = P @ w
        return np.mean((y - pred) ** 2)

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_models
    x0 = np.ones(n_models) / n_models

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-12}
    )

    if not res.success:
        print("WARNING: optimizer did not fully converge:", res.message)

    w = np.maximum(res.x, 0)
    w = w / w.sum()
    return w

def evaluate_meta(P, y, names):
    results = []

    print("\n" + "="*70)
    print("INDIVIDUAL 10-FOLD OOF MODELS")
    print("="*70)

    for i, name in enumerate(names):
        p = P[:, i]
        results.append([
            name, rmse(y,p), pearson(y,p),
            rmse(y,np.clip(p,0,5)),
            pearson(y,np.clip(p,0,5))
        ])
        print(
            f"{name:40s} "
            f"RMSE={rmse(y,p):.6f} "
            f"Pearson={pearson(y,p):.6f}"
        )

    print("\n" + "="*70)
    print("NON-NEGATIVE OOF BLEND")
    print("="*70)

    w = optimize_nonnegative_weights(P, y)
    blend = P @ w

    for name, wi in zip(names, w):
        print(f"{name:40s} {wi:.6f}")

    print(
        f"\nOptimized blend: RMSE={rmse(y,blend):.6f} "
        f"Pearson={pearson(y,blend):.6f}"
    )

    blend_clip = np.clip(blend, 0, 5)
    print(
        f"Clipped blend:   RMSE={rmse(y,blend_clip):.6f} "
        f"Pearson={pearson(y,blend_clip):.6f}"
    )

    results.append([
        "Optimized_NonNegative_Blend",
        rmse(y,blend), pearson(y,blend),
        rmse(y,blend_clip), pearson(y,blend_clip)
    ])

    # Ridge meta learner
    meta_scaler = StandardScaler()
    Z = meta_scaler.fit_transform(P)
    meta = Ridge(alpha=10.0)
    meta.fit(Z, y)
    meta_pred = meta.predict(Z)

    print(
        f"\nRidge meta:      RMSE={rmse(y,meta_pred):.6f} "
        f"Pearson={pearson(y,meta_pred):.6f}"
    )

    print("Ridge meta coefficients:")
    for n, c in zip(names, meta.coef_):
        print(f"  {n:40s} {c:.6f}")

    results.append([
        "Ridge_Meta_InSample",
        rmse(y,meta_pred), pearson(y,meta_pred),
        rmse(y,np.clip(meta_pred,0,5)),
        pearson(y,np.clip(meta_pred,0,5))
    ])

    return pd.DataFrame(
        results,
        columns=["experiment","rmse","pearson","rmse_clipped","pearson_clipped"]
    ), w

def main():
    print("="*70)
    print("EXP-022 — 10-FOLD OOF STACKING")
    print("="*70)

    filenames, y, existing = load_existing_features()
    wavlm = np.load(WAVLM_MULTI).astype(np.float32)
    minilm = np.load(MINILM_TRAIN).astype(np.float32)

    if len(y) != len(wavlm) or len(y) != len(minilm):
        raise ValueError(
            f"Length mismatch: y={len(y)}, wavlm={len(wavlm)}, minilm={len(minilm)}"
        )

    combined = np.concatenate([existing, wavlm], axis=1)

    print("Existing:", existing.shape)
    print("MultiWavLM:", wavlm.shape)
    print("Combined:", combined.shape)
    print("MiniLM:", minilm.shape)
    print("Labels:", y.shape)

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )
    folds = list(kf.split(y))

    predictions = {}
    names = []

    # 1. Strongest model discovered so far.
    print("\n[1/7] Combined MultiWavLM PCA64 + Ridge alpha=100")
    predictions["Combined_PCA64_RidgeA100"] = pca_ridge_oof(
        combined, y, folds, 64, 100.0,
        "Combined_PCA64_RidgeA100"
    )
    names.append("Combined_PCA64_RidgeA100")

    # 2. MultiWavLM-only PCA128 Ridge.
    print("\n[2/7] MultiWavLM PCA128 + Ridge alpha=10")
    predictions["MultiWavLM_PCA128_Ridge"] = pca_ridge_oof(
        wavlm, y, folds, 128, 10.0,
        "MultiWavLM_PCA128_Ridge"
    )
    names.append("MultiWavLM_PCA128_Ridge")

    # 3. Diverse tree model.
    print("\n[3/7] Existing + MultiWavLM ExtraTrees")
    predictions["Existing_MultiWavLM_ExtraTrees"] = extra_trees_oof(
        combined, y, folds,
        "Existing_MultiWavLM_ExtraTrees",
        dict(
            n_estimators=700,
            max_features=0.7,
            min_samples_leaf=2,
            max_depth=None
        )
    )
    names.append("Existing_MultiWavLM_ExtraTrees")

    # 4. Existing-only ExtraTrees.
    print("\n[4/7] Existing features ExtraTrees")
    predictions["Existing_ET_leaf2"] = extra_trees_oof(
        existing, y, folds,
        "Existing_ET_leaf2",
        dict(
            n_estimators=700,
            max_features=0.7,
            min_samples_leaf=2,
            max_depth=None
        )
    )
    names.append("Existing_ET_leaf2")

    # 5. Existing-only Ridge.
    print("\n[5/7] Existing features Ridge")
    predictions["Existing_Ridge"] = ridge_oof(
        existing, y, folds, 10.0, "Existing_Ridge"
    )
    names.append("Existing_Ridge")

    # 6. HGB on existing features.
    print("\n[6/7] Existing features HGB")
    predictions["Existing_HGB"] = hgb_oof(
        existing, y, folds, "Existing_HGB"
    )
    names.append("Existing_HGB")

    # 7. MiniLM transcript embeddings.
    print("\n[7/7] MiniLM Ridge")
    predictions["MiniLM_Ridge"] = ridge_oof(
        minilm, y, folds, 10.0, "MiniLM_Ridge"
    )
    names.append("MiniLM_Ridge")

    P = np.column_stack([predictions[n] for n in names])

    # Save OOF predictions.
    oof = pd.DataFrame({
        "filename": filenames,
        "actual": y
    })
    for i, n in enumerate(names):
        oof[n] = P[:, i]
    oof.to_csv(OUT_OOF, index=False)

    results, weights = evaluate_meta(P, y, names)

    # Also test simple pairwise blends of the strongest candidates.
    print("\n" + "="*70)
    print("PAIRWISE BLEND SEARCH")
    print("="*70)

    pair_rows = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            best = None
            for w in np.linspace(0, 1, 101):
                p = w * P[:,i] + (1-w) * P[:,j]
                score = rmse(y,p)
                if best is None or score < best[0]:
                    best = (score, w, pearson(y,p))

            pair_rows.append([
                names[i], names[j], best[1],
                best[0], best[2]
            ])
            print(
                f"{names[i]} + {names[j]} | "
                f"w1={best[1]:.2f} "
                f"RMSE={best[0]:.6f} "
                f"Pearson={best[2]:.6f}"
            )

    pair_df = pd.DataFrame(
        pair_rows,
        columns=["model_a","model_b","weight_a","rmse","pearson"]
    )

    results = pd.concat([results, pair_df.rename(columns={
        "model_a":"experiment"
    })], ignore_index=True)

    results.to_csv(OUT_RESULTS, index=False)

    pd.DataFrame({
        "model": names,
        "weight": weights
    }).sort_values("weight", ascending=False).to_csv(
        OUT_WEIGHTS, index=False
    )

    print("\n" + "="*70)
    print("EXP-022 SUMMARY")
    print("="*70)

    print("\nTop pairwise blends:")
    print(
        pair_df.sort_values("rmse")
        .head(10)
        .to_string(index=False)
    )

    print("\nOptimized weights:")
    for n, w in sorted(zip(names, weights), key=lambda x: -x[1]):
        print(f"{n:40s} {w:.6f}")

    print("\nSaved:")
    print(OUT_OOF)
    print(OUT_RESULTS)
    print(OUT_WEIGHTS)
    print("\nDONE.")

if __name__ == "__main__":
    main()
