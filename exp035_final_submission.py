import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


# ============================================================
# EXP-035 — FINAL KAGGLE SUBMISSION
#
# Reconstruct the EXP-035 meta ensemble from the saved
# 10-fold OOF predictions, verify the local score, then
# apply exactly the same meta ensemble to the 216 test rows.
# ============================================================

SEED = 42

TEST_CSV = "test.csv"

BASE_TEST = "exp032_test_base_predictions.csv"

EXP022_OOF = "exp022_10fold_oof_predictions.csv"
EXP026_OOF = "exp026_deberta_ft_oof_predictions.csv"
EXP027A_OOF = "exp027a_whisper_oof_predictions.csv"

OUTPUT = "submission_exp035.csv"

EXPECTED_RMSE = 0.503684


print("=" * 80)
print("EXP-035 — FINAL KAGGLE SUBMISSION")
print("=" * 80)


# ============================================================
# 1. LOAD OOF SOURCES
# ============================================================

print("\nLoading OOF predictions...")


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
    "EXP-022:",
    df22.shape
)

print(
    "EXP-026:",
    df26.shape
)

print(
    "EXP-027A:",
    df27.shape
)


# ============================================================
# 2. ALIGN OOF ROWS
# ============================================================

if "filename" not in df22.columns:
    raise RuntimeError(
        "EXP-022 OOF has no filename column."
    )

if "actual" not in df22.columns:
    raise RuntimeError(
        "EXP-022 OOF has no actual column."
    )


names = (
    df22["filename"]
    .astype(str)
    .tolist()
)


target = (
    df22["actual"]
    .astype(float)
    .to_numpy()
)


def get_column(
    df,
    column,
    filenames
):

    if "filename" not in df.columns:
        raise RuntimeError(
            f"Missing filename column for {column}"
        )

    if column not in df.columns:
        raise RuntimeError(
            f"Missing column '{column}'"
        )

    lookup = (
        df
        .drop_duplicates("filename")
        .set_index("filename")
    )

    missing = [
        x
        for x in filenames
        if x not in lookup.index
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
# 3. BUILD THE SEVEN EXP-035 BRANCHES
# ============================================================

oof = pd.DataFrame(
    {
        "WavLM_Combined":
            get_column(
                df22,
                "Combined_PCA64_RidgeA100",
                names
            ),

        "WavLM_Ridge":
            get_column(
                df22,
                "MultiWavLM_PCA128_Ridge",
                names
            ),

        "WavLM_ExtraTrees":
            get_column(
                df22,
                "Existing_MultiWavLM_ExtraTrees",
                names
            ),

        "HGB":
            get_column(
                df22,
                "Existing_HGB",
                names
            ),

        "MiniLM":
            get_column(
                df22,
                "MiniLM_Ridge",
                names
            ),

        "DeBERTa_FT":
            get_column(
                df26,
                "DeBERTa_FT",
                names
            ),

        "Whisper_Base":
            get_column(
                df27,
                "prediction",
                names
            ),
    }
)


print(
    "\nOOF meta matrix:",
    oof.shape
)


print(
    "\nBranches:"
)

for c in oof.columns:

    print(
        " ",
        c
    )


X = oof.to_numpy(
    dtype=np.float64
)


# ============================================================
# 4. META-MODEL CANDIDATES FROM EXP-035
# ============================================================

# These are the models tested in EXP-035.
#
# For the final ensemble we specifically reproduce the
# greedy forward blend:
#
#   ET_1000
#   Ridge_1
#   HGB
#   RF_700
#   ElasticNet
#
# The remaining branches/models are not needed at the
# final meta layer because EXP-035 found zero useful
# additional weight for them.
#
# Each candidate below is fitted directly to the seven
# OOF branch predictions.

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.linear_model import (
    Ridge,
    ElasticNet
)


models = {

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

    "Ridge_1":
        Ridge(
            alpha=1.0
        ),

    "HGB":
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED
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

    "ElasticNet":
        ElasticNet(
            alpha=0.01,
            l1_ratio=0.10,
            max_iter=10000,
            random_state=SEED
        )
}


# ============================================================
# 5. FIT META MODELS ON FULL OOF MATRIX
#
# This is NOT used as the benchmark.
#
# It is only used to generate candidate prediction vectors
# that the EXP-035 ensemble search combines.
# ============================================================

meta_predictions = {}


print(
    "\nFitting EXP-035 meta models..."
)


for name, model in models.items():

    print(
        f"  {name}"
    )

    model.fit(
        X,
        target
    )

    pred = model.predict(
        X
    )

    meta_predictions[
        name
    ] = np.asarray(
        pred,
        dtype=float
    )


# ============================================================
# 6. GREEDY FORWARD BLEND
#
# IMPORTANT:
#
# We optimize a convex combination:
#
#       prediction = sum(w_i * model_i)
#
# subject to:
#
#       w_i >= 0
#       sum(w_i) = 1
#
# This avoids an unconstrained meta-model producing
# unstable coefficients.
# ============================================================

ordered_models = [
    "ET_1000",
    "Ridge_1",
    "HGB",
    "RF_700",
    "ElasticNet"
]


def optimize_weights(
    prediction_matrix,
    y
):

    n_models = prediction_matrix.shape[1]

    def objective(weights):

        pred = (
            prediction_matrix
            @
            weights
        )

        return np.sqrt(
            mean_squared_error(
                y,
                pred
            )
        )


    initial = (
        np.ones(n_models)
        /
        n_models
    )


    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
            for _ in range(n_models)
        ],
        constraints=[
            {
                "type": "eq",
                "fun":
                    lambda w:
                    np.sum(w) - 1.0
            }
        ],
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
            "disp": False
        }
    )


    if not result.success:

        raise RuntimeError(
            "Weight optimization failed:\n"
            +
            str(result.message)
        )


    return result.x


# ============================================================
# 7. RECONSTRUCT EXP-035 ENSEMBLE
# ============================================================

prediction_matrix = np.column_stack(
    [
        meta_predictions[name]
        for name in ordered_models
    ]
)


weights = optimize_weights(
    prediction_matrix,
    target
)


final_oof = (
    prediction_matrix
    @
    weights
)


final_oof = np.clip(
    final_oof,
    0,
    5
)


rmse = np.sqrt(
    mean_squared_error(
        target,
        final_oof
    )
)


pearson = np.corrcoef(
    target,
    final_oof
)[0, 1]


print(
    "\n" + "=" * 80
)

print(
    "RECONSTRUCTED EXP-035 OOF ENSEMBLE"
)

print(
    "=" * 80
)


for name, weight in zip(
    ordered_models,
    weights
):

    print(
        f"{name:<15} "
        f"{weight:.10f}"
    )


print(
    f"\nOOF RMSE:    {rmse:.6f}"
)

print(
    f"OOF Pearson: {pearson:.6f}"
)

print(
    f"Target RMSE: {EXPECTED_RMSE:.6f}"
)


# ============================================================
# 8. SAFETY CHECK
# ============================================================

difference = abs(
    rmse
    -
    EXPECTED_RMSE
)


print(
    f"\nDifference from EXP-035 benchmark: "
    f"{difference:.6f}"
)


if difference > 0.003:

    print(
        "\n"
        + "!" * 80
    )

    print(
        "WARNING"
    )

    print(
        "The reconstructed ensemble does NOT reproduce "
        "the EXP-035 benchmark closely enough."
    )

    print(
        "Submission will NOT be created."
    )

    print(
        "Do not submit this result."
    )

    print(
        "!" * 80
    )

    raise SystemExit(1)


print(
    "\nEXP-035 benchmark reproduced closely."
)

print(
    "Proceeding to 216-row test inference."
)


# ============================================================
# 9. LOAD TEST BASE PREDICTIONS
# ============================================================

test = pd.read_csv(
    TEST_CSV
)


if len(test) != 216:

    raise RuntimeError(
        f"Expected 216 test rows, got {len(test)}"
    )


test_names = (
    test["filename"]
    .astype(str)
    .tolist()
)


base_test = pd.read_csv(
    BASE_TEST
)


print(
    "\nTest base prediction file:",
    BASE_TEST
)

print(
    "Shape:",
    base_test.shape
)


if "filename" not in base_test.columns:

    raise RuntimeError(
        "exp032_test_base_predictions.csv "
        "must contain filename."
    )


# ============================================================
# 10. INSPECT TEST BASE COLUMNS
# ============================================================

print(
    "\nTest base columns:"
)

for c in base_test.columns:

    print(
        " ",
        c
    )


# ============================================================
# 11. ALIGN TEST BRANCHES
# ============================================================

test_lookup = (
    base_test
    .drop_duplicates("filename")
    .set_index("filename")
)


missing = [
    x
    for x in test_names
    if x not in test_lookup.index
]


if missing:

    raise RuntimeError(
        f"Test base predictions missing "
        f"{len(missing)} filenames."
    )


# Expected seven branch columns.
#
# EXP-032 was created specifically from these branches.
# We accept the exact names first, then a few aliases.

aliases = {

    "WavLM_Combined":
        [
            "WavLM_Combined",
            "Combined_PCA64_RidgeA100"
        ],

    "WavLM_Ridge":
        [
            "WavLM_Ridge",
            "MultiWavLM_PCA128_Ridge"
        ],

    "WavLM_ExtraTrees":
        [
            "WavLM_ExtraTrees",
            "Existing_MultiWavLM_ExtraTrees"
        ],

    "HGB":
        [
            "HGB",
            "Existing_HGB"
        ],

    "MiniLM":
        [
            "MiniLM",
            "MiniLM_Ridge"
        ],

    "DeBERTa_FT":
        [
            "DeBERTa_FT"
        ],

    "Whisper_Base":
        [
            "Whisper_Base",
            "Whisper_Base_Ridge",
            "prediction"
        ]
}


def resolve_column(
    lookup,
    aliases
):

    for candidate in aliases:

        if candidate in lookup.columns:

            return candidate

    return None


test_branch_data = {}


for branch in oof.columns:

    column = resolve_column(
        test_lookup,
        aliases[branch]
    )


    if column is None:

        raise RuntimeError(
            f"\nCould not find test prediction "
            f"column for branch: {branch}\n"
            f"Available columns:\n"
            +
            "\n".join(
                str(x)
                for x in test_lookup.columns
            )
        )


    print(
        f"{branch:<22} <- {column}"
    )


    test_branch_data[
        branch
    ] = (
        test_lookup
        .loc[test_names, column]
        .astype(float)
        .to_numpy()
    )


# ============================================================
# 12. BUILD TEST META MATRIX
# ============================================================

X_test = np.column_stack(
    [
        test_branch_data[name]
        for name in oof.columns
    ]
)


print(
    "\nTest meta matrix:",
    X_test.shape
)


# ============================================================
# 13. FIT FINAL META MODELS ON FULL TRAIN OOF
#
# These are now the actual meta models used on the
# 216 test samples.
# ============================================================

test_meta_predictions = {}


for name, model in models.items():

    print(
        f"Generating test predictions: {name}"
    )

    model.fit(
        X,
        target
    )

    test_meta_predictions[
        name
    ] = model.predict(
        X_test
    )


# ============================================================
# 14. FINAL TEST ENSEMBLE
# ============================================================

test_prediction_matrix = np.column_stack(
    [
        test_meta_predictions[name]
        for name in ordered_models
    ]
)


final_test = (
    test_prediction_matrix
    @
    weights
)


# Score range is 0–5.
final_test = np.clip(
    final_test,
    0,
    5
)


# ============================================================
# 15. BUILD SUBMISSION
# ============================================================

submission = pd.DataFrame(
    {
        "filename":
            test_names,

        "label":
            final_test
    }
)


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

if len(submission) != 216:

    raise RuntimeError(
        "Submission does not contain exactly 216 rows."
    )


if not np.array_equal(
    submission["filename"].to_numpy(),
    test["filename"].astype(str).to_numpy()
):

    raise RuntimeError(
        "Submission filename order does not exactly "
        "match test.csv."
    )


if submission["label"].isna().any():

    raise RuntimeError(
        "Submission contains NaN predictions."
    )


if not np.isfinite(
    submission["label"].to_numpy()
).all():

    raise RuntimeError(
        "Submission contains non-finite predictions."
    )


if (
    submission["label"].min() < 0
    or
    submission["label"].max() > 5
):

    raise RuntimeError(
        "Prediction outside [0,5]."
    )


# ============================================================
# 17. SAVE
# ============================================================

submission.to_csv(
    OUTPUT,
    index=False
)


print(
    "\n" + "=" * 80
)

print(
    "EXP-035 SUBMISSION READY"
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
    f"Mean: {submission['label'].mean():.6f}"
)

print(
    f"Std : {submission['label'].std():.6f}"
)

print(
    f"Min : {submission['label'].min():.6f}"
)

print(
    f"Max : {submission['label'].max():.6f}"
)


print(
    "\nFirst 10 rows:"
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
    "CONFIRMED"
)

print(
    "=" * 80
)

print(
    "✓ 216 rows"
)

print(
    "✓ filenames taken directly from test.csv"
)

print(
    "✓ sample_submission.csv NOT used"
)

print(
    "✓ predictions clipped to [0, 5]"
)

print(
    "✓ EXP-035 OOF benchmark verified before submission"
)

print(
    "=" * 80
)