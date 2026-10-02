from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\kagle_project\Dataset_Final")

train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")
sample = pd.read_csv(ROOT / "sample_submission.csv")

train_files = {x.name for x in (ROOT / "train").glob("*.wav")}
test_audio_files = {x.name for x in (ROOT / "test").glob("*.wav")}

print("=" * 70)
print("DATASET / AUDIO CONSISTENCY CHECK")
print("=" * 70)

print("\nTrain CSV rows:", len(train))
print("Train audio files:", len(train_files))

print("\nTest CSV rows:", len(test))
print("Test audio files:", len(test_audio_files))

print("\nSample submission rows:", len(sample))

print("\n" + "=" * 70)
print("TRAIN CSV vs TRAIN AUDIO")
print("=" * 70)

train_csv_files = set(train["filename"])

print(
    "CSV filenames found in audio:",
    len(train_csv_files & train_files)
)

print(
    "CSV filenames missing from audio:",
    len(train_csv_files - train_files)
)

print(
    "Audio files missing from CSV:",
    len(train_files - train_csv_files)
)

print("\n" + "=" * 70)
print("TEST CSV vs TEST AUDIO")
print("=" * 70)

test_csv_files = set(test["filename"])

print(
    "CSV filenames found in audio:",
    len(test_csv_files & test_audio_files)
)

print(
    "CSV filenames missing from audio:",
    len(test_csv_files - test_audio_files)
)

print(
    "Audio files missing from CSV:",
    len(test_audio_files - test_csv_files)
)

print("\n" + "=" * 70)
print("SAMPLE SUBMISSION vs TEST AUDIO")
print("=" * 70)

sample_files = set(sample["filename"])

print(
    "Sample filenames found in test audio:",
    len(sample_files & test_audio_files)
)

print(
    "Sample filenames NOT in test audio:",
    len(sample_files - test_audio_files)
)

print("\n" + "=" * 70)
print("ACTUAL TEST AUDIO SAMPLE")
print("=" * 70)

print(
    sorted(test_audio_files)[:30]
)

print("\n" + "=" * 70)
print("SAMPLE SUBMISSION SAMPLE")
print("=" * 70)

print(
    sample["filename"].head(30).tolist()
)

print("\nDone.")