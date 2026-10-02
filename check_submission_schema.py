import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\kagle_project\Dataset_Final")

train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")
sample = pd.read_csv(ROOT / "sample_submission.csv")

print("=" * 70)
print("SUBMISSION SCHEMA CHECK")
print("=" * 70)

print("\nTRAIN")
print(train.shape)
print(train.columns.tolist())
print(train.head())

print("\nTEST")
print(test.shape)
print(test.columns.tolist())
print(test.head())

print("\nSAMPLE SUBMISSION")
print(sample.shape)
print(sample.columns.tolist())
print(sample.head())

test_files = set(test["filename"])
sample_files = set(sample["filename"])

print("\n" + "=" * 70)
print("FILENAME COMPARISON")
print("=" * 70)

print("Test filenames:", len(test_files))
print("Sample filenames:", len(sample_files))
print("Overlap:", len(test_files & sample_files))

missing_from_sample = sorted(test_files - sample_files)
extra_in_sample = sorted(sample_files - test_files)

print("\nTest files missing from sample_submission:")
print(len(missing_from_sample))

for x in missing_from_sample:
    print(" ", x)

print("\nSample files NOT present in test.csv:")
print(len(extra_in_sample))

for x in extra_in_sample:
    print(" ", x)

print("\n" + "=" * 70)
print("ORDER CHECK")
print("=" * 70)

print(
    "Test first 10:",
    test["filename"].head(10).tolist()
)

print(
    "Sample first 10:",
    sample["filename"].head(10).tolist()
)

print("\nDone.")