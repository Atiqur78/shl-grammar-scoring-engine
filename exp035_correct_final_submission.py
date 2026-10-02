import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

from scipy.stats import pearsonr


# ============================================================
# EXP-035 — CORRECT FINAL SUBMISSION
#
# IMPORTANT:
# Proper OUTER 10-fold meta-CV.
#
# The base predictions are already OOF predictions.
# We must NOT evaluate a meta-model on the same rows used
# to fit that meta-model.
#
# Procedure:
#
#   base OOF predictions
#          ↓
#   outer 10-fold
#          ↓
#   fit meta-model on 90%
#          ↓
#   predict 10%
#          ↓
#   collect true meta OOF predictions
#          ↓
#   verify EXP-035 benchmark
#          ↓
#   fit final meta-model on all 769
#          ↓
#   predict 216 test rows
#
# ============================================================


SEED = 42
N_SPLITS = 10

EXPECTED_RMSE = 0.503684
EXPECTED_PEARSON = 0.913690

TEST_CSV = "test.csv"

EXP022_OOF = "exp022_10fold_oof_predictions.csv"
EXP026_OOF = "exp026_deberta_ft_oof_predictions.csv"
EXP027A_OOF = "exp027a_whisper_oof_predictions.csv"

# Existing test branch predictions.
#
# IMPORTANT:
# We will inspect this file and resolve the branch names.
BASE_TEST = "exp032_test_base_predictions.csv"

OUTPUT = "submission_exp035.csv"


print("=" * 80)
print("EXP-035 — PROPER META-CV FINAL SUBMISSION")
print("=" * 80)


# ============================================================
# 1. LOAD OOF FILES
# ============================================================

print("\nLoading OOF files...")

df22 = pd.read_csv(
    EXP022_OOF
)

df26 = pd.read_csv(
    EXP026_OOF
)

df27 = pd.read_csv(
    EXP027A_OOF
)


print(
    f"EXP-022: {df22.shape}"
)

print(
    f"EXP-026: {df26.shape}"
)

print(
    f"EXP-027A: {df27.shape}"
)


# ============================================================
# 2. ALIGN BY FILENAME
# ============================================================

if "filename" not in df22.columns:
    raise RuntimeError(
        "EXP-022 OOF has no filename column."
    )

if "actual" not in df22.columns:
    raise RuntimeError(
        "EXP-022 OOF has no actual column."
    )


filenames = (
    df22["filename"]
    .astype(str)
    .tolist()
)


y = (
    df22["actual"]
    .astype(float)
    .to_numpy()
)


def resolve_oof_column(
    df,
    column,
    filenames
):

    if "filename" not in df.columns:
        raise RuntimeError(
            f"Missing filename column."
        )

    if column not in df.columns:
        raise RuntimeError(
            f"Missing OOF column: {column}"
        )

    lookup = (
        df
        .drop_duplicates("filename")
        .set_index("filename")
    )

    missing = [
        f
        for f in filenames
        if f not in lookup.index
    ]

    if missing:

        raise RuntimeError(
            f"{column}: missing "
            f"{len(missing)} filenames."
        )

    return (
        lookup
        .loc[filenames, column]
        .astype(float)
        .to_numpy()
    )


# ============================================================
# 3. SEVEN EXP-035 BASE BRANCHES
# ============================================================

meta_df = pd.DataFrame(
    {
        "WavLM_Combined":
            resolve_oof_column(
                df22,
                "Combined_PCA64_RidgeA100",
                filenames
            ),

        "WavLM_Ridge":
            resolve_oof_column(
                df22,
                "MultiWavLM_PCA128_Ridge",
                filenames
            ),

        "WavLM_ExtraTrees":
            resolve_oof_column(
                df22,
                "Existing_MultiWavLM_ExtraTrees",
                filenames
            ),

        "HGB":
            resolve_oof_column(
                df22,
                "Existing_HGB",
                filenames
            ),

        "MiniLM":
            resolve_oof_column(
                df22,
                "MiniLM_Ridge",
                filenames
            ),

        "DeBERTa_FT":
            resolve_oof_column(
                df26,
                "DeBERTa_FT",
                filenames
            ),

        "Whisper_Base":
            resolve_oof_column(
                df27,
                "prediction",
                filenames
            ),
    }
)


X = meta_df.to_numpy(
    dtype=np.float64
)


print(
    "\nMeta matrix:",
    X.shape
)

print(
    "Target:",
    y.shape
)


# ============================================================
# 4. META MODELS
# ============================================================

def make_models():

    return {

        "ET_500":
            ExtraTreesRegressor(
                n_estimators=500,
                max_features=0.8,
                min_samples_leaf=6,
                max_depth=None,
                criterion="squared_error",
                random_state=SEED,
                n_jobs=-1
            ),

        "ET_700":
            ExtraTreesRegressor(
                n_estimators=700,
                max_features=0.8,
                min_samples_leaf=6,
                max_depth=None,
                criterion="squared_error",
                random_state=SEED,
                n_jobs=-1
            ),

        "ET_1000":
            ExtraTreesRegressor(
                n_estimators=1000,
                max_features=0.8,
                min_samples_leaf=6,
                max_depth=None,
                criterion="squared_error",
                random_state=SEED,
                n_jobs=-1
            ),

        "RF_700":
            RandomForestRegressor(
                n_estimators=700,
                max_features=0.8,
                min_samples_leaf=4,
                max_depth=None,
                criterion="squared_error",
                random_state=SEED,
                n_jobs=-1
            ),

        "HGB":
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.03,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=SEED
            ),

        "Ridge_1":
            Ridge(
                alpha=1.0
            ),

        "Ridge_10":
            Ridge(
                alpha=10.0
            ),

        "ElasticNet":
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.10,
                max_iter=10000,
                random_state=SEED
            )
    }


# ============================================================
# 5. PROPER OUTER 10-FOLD META-CV
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


model_names = list(
    make_models().keys()
)


meta_oof = {
    name:
        np.zeros(
            len(y),
            dtype=np.float64
        )
    for name in model_names
}


print(
    "\n" + "=" * 80
)

print(
    "PROPER OUTER 10-FOLD META-CV"
)

print(
    "=" * 80
)


for fold, (
    train_idx,
    valid_idx
) in enumerate(
    kf.split(X),
    start=1
):

    print(
        f"\nFOLD {fold:02d}"
    )

    print(
        f"  train={len(train_idx)} "
        f"valid={len(valid_idx)}"
    )


    models = make_models()


    for name in model_names:

        model = models[name]


        model.fit(
            X[train_idx],
            y[train_idx]
        )


        pred = model.predict(
            X[valid_idx]
        )


        meta_oof[
            name
        ][valid_idx] = pred


        fold_rmse = np.sqrt(
            mean_squared_error(
                y[valid_idx],
                pred
            )
        )


        print(
            f"  {name:<12} "
            f"RMSE={fold_rmse:.6f}"
        )


# ============================================================
# 6. EVALUATE INDIVIDUAL META MODELS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "PROPER META-CV RESULTS"
)

print(
    "=" * 80
)


results = []


for name in model_names:

    pred = np.clip(
        meta_oof[name],
        0,
        5
    )


    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred
        )
    )


    pearson = pearsonr(
        y,
        pred
    )[0]


    results.append(
        {
            "model": name,
            "rmse": rmse,
            "pearson": pearson
        }
    )


    print(
        f"{name:<12} "
        f"RMSE={rmse:.6f} "
        f"Pearson={pearson:.6f}"
    )


results_df = (
    pd.DataFrame(results)
    .sort_values("rmse")
)


# ============================================================
# 7. EXP-035 GREEDY ENSEMBLE
# ============================================================
#
# EXP-035 reported:
#
#   ET_1000
#   Ridge_1
#   HGB
#   RF_700
#   ElasticNet
#
# with greedy forward blending.
#
# We now perform the SAME greedy process using the
# properly generated meta OOF predictions.
#
# ============================================================


candidate_names = [
    "ET_1000",
    "Ridge_1",
    "HGB",
    "RF_700",
    "ElasticNet"
]


def evaluate_prediction(
    pred
):

    pred = np.clip(
        pred,
        0,
        5
    )


    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred
        )
    )


    pearson = pearsonr(
        y,
        pred
    )[0]


    return rmse, pearson


# Start with ET_1000.
selected = [
    "ET_1000"
]

current = (
    meta_oof[
        "ET_1000"
    ].copy()
)


rmse, pearson = evaluate_prediction(
    current
)


print(
    "\n" + "=" * 80
)

print(
    "GREEDY FORWARD ENSEMBLE"
)

print(
    "=" * 80
)

print(
    f"START ET_1000: "
    f"RMSE={rmse:.6f} "
    f"Pearson={pearson:.6f}"
)


# ------------------------------------------------------------
# At each step, search weight from 0 to 1.
# ------------------------------------------------------------

ensemble_weights = {
    "ET_1000": 1.0
}


remaining = [
    x
    for x in candidate_names
    if x != "ET_1000"
]


for candidate in remaining:

    candidate_pred = (
        meta_oof[
            candidate
        ]
    )


    best_rmse = rmse
    best_weight = 0.0
    best_prediction = current


    # Fine grid.
    for weight in np.linspace(
        0.0,
        1.0,
        1001
    ):

        trial = (
            (1.0 - weight)
            *
            current
            +
            weight
            *
            candidate_pred
        )


        trial_rmse, _ = evaluate_prediction(
            trial
        )


        if trial_rmse < best_rmse:

            best_rmse = trial_rmse

            best_weight = weight

            best_prediction = trial


    if best_weight > 0.0:

        old_weights = dict(
            ensemble_weights
        )


        for name in old_weights:

            ensemble_weights[name] *= (
                1.0
                -
                best_weight
            )


        ensemble_weights[
            candidate
        ] = best_weight


        current = best_prediction

        rmse, pearson = evaluate_prediction(
            current
        )


        selected.append(
            candidate
        )


        print(
            f"\nADDED {candidate}"
        )

        print(
            f"  weight={best_weight:.4f}"
        )

        print(
            f"  RMSE={rmse:.6f}"
        )

        print(
            f"  Pearson={pearson:.6f}"
        )

    else:

        print(
            f"\nSKIPPED {candidate}"
        )

        print(
            f"  no RMSE improvement"
        )


# ============================================================
# 8. NORMALIZE WEIGHTS
# ============================================================

weight_sum = sum(
    ensemble_weights.values()
)


for name in ensemble_weights:

    ensemble_weights[name] /= weight_sum


# Recalculate.
ensemble_oof = np.zeros(
    len(y),
    dtype=float
)


for name, weight in (
    ensemble_weights.items()
):

    ensemble_oof += (
        weight
        *
        meta_oof[name]
    )


ensemble_oof = np.clip(
    ensemble_oof,
    0,
    5
)


ensemble_rmse, ensemble_pearson = (
    evaluate_prediction(
        ensemble_oof
    )
)


print(
    "\n" + "=" * 80
)

print(
    "FINAL EXP-035 RECONSTRUCTION"
)

print(
    "=" * 80
)


print(
    "\nWeights:"
)

for name, weight in (
    ensemble_weights.items()
):

    print(
        f"  {name:<12} "
        f"{weight:.8f}"
    )


print(
    f"\nRMSE:    {ensemble_rmse:.6f}"
)

print(
    f"Pearson: {ensemble_pearson:.6f}"
)


# ============================================================
# 9. BENCHMARK CHECK
# ============================================================

rmse_difference = abs(
    ensemble_rmse
    -
    EXPECTED_RMSE
)


pearson_difference = abs(
    ensemble_pearson
    -
    EXPECTED_PEARSON
)


print(
    "\n" + "=" * 80
)

print(
    "BENCHMARK CHECK"
)

print(
    "=" * 80
)


print(
    f"Expected RMSE:    {EXPECTED_RMSE:.6f}"
)

print(
    f"Actual RMSE:      {ensemble_rmse:.6f}"
)

print(
    f"RMSE difference:  {rmse_difference:.6f}"
)

print()

print(
    f"Expected Pearson: {EXPECTED_PEARSON:.6f}"
)

print(
    f"Actual Pearson:   {ensemble_pearson:.6f}"
)

print(
    f"Pearson diff:     {pearson_difference:.6f}"
)


# ------------------------------------------------------------
# IMPORTANT:
#
# We allow a small tolerance because implementation details
# can cause tiny differences.
# ------------------------------------------------------------

if rmse_difference > 0.002:

    print(
        "\n" + "!" * 80
    )

    print(
        "BENCHMARK NOT REPRODUCED"
    )

    print(
        "NO SUBMISSION CREATED."
    )

    print(
        "!" * 80
    )

    results_df.to_csv(
        "exp035_correct_meta_results.csv",
        index=False
    )

    raise SystemExit(1)


print(
    "\n✓ EXP-035 benchmark reproduced."
)

print(
    "Proceeding to final test inference."
)


# ============================================================
# 10. LOAD TEST BRANCH PREDICTIONS
# ============================================================

test = pd.read_csv(
    TEST_CSV
)


if len(test) != 216:

    raise RuntimeError(
        f"Expected 216 test rows, got {len(test)}"
    )


test_filenames = (
    test["filename"]
    .astype(str)
    .tolist()
)


base_test = pd.read_csv(
    BASE_TEST
)


print(
    "\nTest prediction source:",
    BASE_TEST
)

print(
    "Shape:",
    base_test.shape
)


print(
    "\nAvailable test columns:"
)

for c in base_test.columns:

    print(
        " ",
        c
    )


# ============================================================
# 11. RESOLVE TEST BRANCH COLUMNS
# ============================================================

lookup = (
    base_test
    .drop_duplicates("filename")
    .set_index("filename")
)


if len(lookup) != len(base_test):

    print(
        "Warning: duplicate filenames existed "
        "in test prediction file."
    )


aliases = {

    "WavLM_Combined": [
        "WavLM_Combined",
        "Combined_PCA64_RidgeA100"
    ],

    "WavLM_Ridge": [
        "WavLM_Ridge",
        "MultiWavLM_PCA128_Ridge"
    ],

    "WavLM_ExtraTrees": [
        "WavLM_ExtraTrees",
        "Existing_MultiWavLM_ExtraTrees"
    ],

    "HGB": [
        "HGB",
        "Existing_HGB"
    ],

    "MiniLM": [
        "MiniLM",
        "MiniLM_Ridge"
    ],

    "DeBERTa_FT": [
        "DeBERTa_FT"
    ],

    "Whisper_Base": [
        "Whisper_Base",
        "Whisper_Base_Ridge",
        "prediction"
    ]
}


def find_test_column(
    branch
):

    for candidate in aliases[
        branch
    ]:

        if candidate in lookup.columns:

            return candidate

    return None


test_meta_df = pd.DataFrame(
    index=test_filenames
)


for branch in meta_df.columns:

    column = find_test_column(
        branch
    )


    if column is None:

        raise RuntimeError(
            "\nCould not resolve test column for "
            f"{branch}.\n\n"
            "Available columns:\n"
            +
            "\n".join(
                str(c)
                for c in lookup.columns
            )
        )


    print(
        f"{branch:<22} <- {column}"
    )


    test_meta_df[
        branch
    ] = (
        lookup
        .loc[
            test_filenames,
            column
        ]
        .astype(float)
        .to_numpy()
    )


X_test = test_meta_df[
    meta_df.columns
].to_numpy(
    dtype=float
)


print(
    "\nTest meta matrix:",
    X_test.shape
)


# ============================================================
# 12. TRAIN FINAL META MODELS ON ALL 769 SAMPLES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "TRAINING FINAL META MODELS"
)

print(
    "=" * 80
)


final_models = make_models()

test_meta_predictions = {}


for name in ensemble_weights:

    model = clone(
        final_models[name]
    )


    print(
        f"Training {name}..."
    )


    model.fit(
        X,
        y
    )


    test_meta_predictions[
        name
    ] = model.predict(
        X_test
    )


# ============================================================
# 13. FINAL ENSEMBLE
# ============================================================

test_prediction = np.zeros(
    len(test),
    dtype=float
)


for name, weight in (
    ensemble_weights.items()
):

    test_prediction += (
        weight
        *
        test_meta_predictions[name]
    )


test_prediction = np.clip(
    test_prediction,
    0,
    5
)


# ============================================================
# 14. BUILD SUBMISSION
# ============================================================

submission = pd.DataFrame(
    {
        "filename":
            test_filenames,

        "label":
            test_prediction
    }
)


# ============================================================
# 15. SAFETY CHECKS
# ============================================================

assert len(
    submission
) == 216


assert np.array_equal(
    submission["filename"].to_numpy(),
    test["filename"]
    .astype(str)
    .to_numpy()
)


assert not submission[
    "label"
].isna().any()


assert np.isfinite(
    submission["label"]
    .to_numpy()
).all()


assert (
    submission["label"].min()
    >= 0
)


assert (
    submission["label"].max()
    <= 5
)


# ============================================================
# 16. SAVE
# ============================================================

submission.to_csv(
    OUTPUT,
    index=False
)


# Also save weights/results for reproducibility.
pd.DataFrame(
    [
        {
            "model": name,
            "weight": weight
        }

        for name, weight
        in ensemble_weights.items()
    ]
).to_csv(
    "exp035_final_weights.csv",
    index=False
)


results_df.to_csv(
    "exp035_correct_meta_results.csv",
    index=False
)


# ============================================================
# 17. FINAL REPORT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "SUBMISSION CREATED"
)

print(
    "=" * 80
)

print(
    f"File: {OUTPUT}"
)

print(
    f"Rows: {len(submission)}"
)

print(
    f"Mean prediction: "
    f"{submission['label'].mean():.6f}"
)

print(
    f"Std prediction:  "
    f"{submission['label'].std():.6f}"
)

print(
    f"Min prediction:  "
    f"{submission['label'].min():.6f}"
)

print(
    f"Max prediction:  "
    f"{submission['label'].max():.6f}"
)


print(
    "\nFirst 10 predictions:"
)

print(
    submission.head(10).to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "READY FOR KAGGLE"
)

print(
    "=" * 80
)

print(
    "✓ Proper outer 10-fold meta-CV"
)

print(
    "✓ Benchmark checked before submission"
)

print(
    "✓ 216 rows from test.csv"
)

print(
    "✓ sample_submission.csv not used"
)

print(
    "✓ Predictions clipped to [0, 5]"
)

print(
    f"✓ Local RMSE = {ensemble_rmse:.6f}"
)

print(
    f"✓ Local Pearson = {ensemble_pearson:.6f}"
)

print(
    "=" * 80
)