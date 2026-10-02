from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


ROOT = Path(r"D:\kagle_project\Dataset_Final")

DATA_PATH = ROOT / "acoustic_features.csv"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(
    columns=["filename", "label"]
)

y = df["label"].astype(float)


print("=" * 70)
print("ACOUSTIC BASELINE")
print("=" * 70)

print("Samples :", len(df))
print("Features:", X.shape[1])


# ============================================================
# Cross validation
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# Models
# ============================================================

models = {

    "Ridge": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(alpha=10.0)
        )
    ]),

    "ExtraTrees": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=500,
                max_features=0.7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]),

    "RandomForest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                max_features=0.7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]),

    "HistGradientBoosting": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                random_state=42
            )
        )
    ])
}


# ============================================================
# Evaluation
# ============================================================

results = []

oof_predictions = {}

for model_name, model in models.items():

    print()
    print("-" * 70)
    print(model_name)
    print("-" * 70)

    oof = np.zeros(len(df))

    fold_rmses = []
    fold_pearsons = []

    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(X),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_valid
        )

        # Keep predictions inside the valid score range.
        pred = np.clip(
            pred,
            0.0,
            5.0
        )

        oof[valid_idx] = pred

        rmse = np.sqrt(
            mean_squared_error(
                y_valid,
                pred
            )
        )

        pearson = pearsonr(
            y_valid,
            pred
        )[0]

        fold_rmses.append(rmse)
        fold_pearsons.append(pearson)

        print(
            f"Fold {fold}: "
            f"RMSE={rmse:.4f} | "
            f"Pearson={pearson:.4f}"
        )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y,
            oof
        )
    )

    overall_pearson = pearsonr(
        y,
        oof
    )[0]

    print()
    print(
        f"OOF RMSE    : {overall_rmse:.4f}"
    )

    print(
        f"OOF Pearson : {overall_pearson:.4f}"
    )

    print(
        f"RMSE std    : {np.std(fold_rmses):.4f}"
    )

    print(
        f"Pearson std : {np.std(fold_pearsons):.4f}"
    )

    results.append({
        "model": model_name,
        "rmse": overall_rmse,
        "pearson": overall_pearson,
        "rmse_std": np.std(fold_rmses),
        "pearson_std": np.std(fold_pearsons)
    })

    oof_predictions[model_name] = oof


# ============================================================
# Summary
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "rmse"
)

print()
print("=" * 70)
print("FINAL CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    ROOT / "acoustic_baseline_results.csv",
    index=False
)


# ============================================================
# Save OOF predictions
# ============================================================

oof_df = pd.DataFrame({
    "filename": df["filename"],
    "label": y
})

for model_name, pred in oof_predictions.items():
    oof_df[
        f"pred_{model_name}"
    ] = pred

oof_df.to_csv(
    ROOT / "acoustic_oof_predictions.csv",
    index=False
)

print()
print(
    "Saved:"
)

print(
    ROOT / "acoustic_baseline_results.csv"
)

print(
    ROOT / "acoustic_oof_predictions.csv"
)