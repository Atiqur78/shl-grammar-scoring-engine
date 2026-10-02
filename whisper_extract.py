import json
import re
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\kagle_project\Dataset_Final")

TRAIN_DIR = ROOT / "train"
TRAIN_CSV = ROOT / "train.csv"
OUTPUT_CSV = ROOT / "whisper_transcripts.csv"


# ============================================================
# SETTINGS
# ============================================================

MODEL_SIZE = "base"

# Your GPU is now working correctly.
DEVICE = "cuda"
COMPUTE_TYPE = "float16"


# ============================================================
# LOAD TRAIN CSV
# ============================================================

print("=" * 70)
print("WHISPER TRANSCRIPTION")
print("=" * 70)

train_df = pd.read_csv(TRAIN_CSV)

print(f"Training rows: {len(train_df)}")

# Use filenames from train.csv instead of assuming
# audio_0.wav ... audio_768.wav exist consecutively.
filenames = train_df["filename"].astype(str).tolist()

label_map = dict(
    zip(
        train_df["filename"].astype(str),
        train_df["label"]
    )
)


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

if OUTPUT_CSV.exists():

    existing_df = pd.read_csv(OUTPUT_CSV)

    completed = set(
        existing_df["filename"].astype(str)
    )

    print(f"Existing results: {len(completed)}")
    print("Already completed files will be skipped.")

else:

    completed = set()

    print("No previous results found.")
    print("Starting from the beginning.")


remaining = [
    f for f in filenames
    if f not in completed
]

print(f"Remaining files: {len(remaining)}")


if len(remaining) == 0:

    print("\nAll files are already processed.")
    print(f"Output: {OUTPUT_CSV}")
    raise SystemExit


# ============================================================
# LOAD WHISPER
# ============================================================

print("\nLoading Whisper model...")

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("Whisper model loaded successfully.")


# ============================================================
# PROCESS FILES
# ============================================================

start_all = time.time()

for position, filename in enumerate(remaining, start=1):

    audio_path = TRAIN_DIR / filename

    print()
    print("-" * 70)
    print(f"[{position}/{len(remaining)}] {filename}")

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

        # Convert stereo -> mono if necessary
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio_duration = len(audio) / sample_rate

        # ----------------------------------------------------
        # Whisper transcription
        # ----------------------------------------------------

        start_time = time.time()

        segments_generator, info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False
        )

        # IMPORTANT:
        # segments is a generator, so force it to execute.
        segments = list(segments_generator)

        processing_time = time.time() - start_time

        # ----------------------------------------------------
        # Extract segment information
        # ----------------------------------------------------

        segment_records = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            segment_records.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text
                }
            )

        # ----------------------------------------------------
        # Full transcript
        # ----------------------------------------------------

        transcript = " ".join(
            x["text"]
            for x in segment_records
        )

        # ----------------------------------------------------
        # Basic word statistics
        # ----------------------------------------------------

        words = re.findall(
            r"\b[\w']+\b",
            transcript.lower()
        )

        word_count = len(words)

        unique_word_count = len(set(words))

        # ----------------------------------------------------
        # Speech duration
        # ----------------------------------------------------

        speech_duration = sum(
            max(0, x["end"] - x["start"])
            for x in segment_records
        )

        # ----------------------------------------------------
        # Inter-segment gaps
        # ----------------------------------------------------

        gaps = []

        for i in range(1, len(segment_records)):

            previous_end = segment_records[i - 1]["end"]
            current_start = segment_records[i]["start"]

            gap = max(
                0,
                current_start - previous_end
            )

            gaps.append(gap)

        if gaps:

            total_gap_duration = sum(gaps)
            mean_gap_duration = sum(gaps) / len(gaps)
            max_gap_duration = max(gaps)

        else:

            total_gap_duration = 0.0
            mean_gap_duration = 0.0
            max_gap_duration = 0.0

        # ----------------------------------------------------
        # Words per minute
        # ----------------------------------------------------

        if speech_duration > 0:

            words_per_minute = (
                word_count / speech_duration
            ) * 60

        else:

            words_per_minute = 0.0

        # ----------------------------------------------------
        # Real-time factor
        # ----------------------------------------------------

        if audio_duration > 0:

            rtf = processing_time / audio_duration

        else:

            rtf = 0.0

        # ----------------------------------------------------
        # Create result row
        # ----------------------------------------------------

        row = {

            "filename": filename,

            "label": label_map.get(
                filename,
                None
            ),

            "transcript": transcript,

            "audio_duration": audio_duration,

            "sample_rate": sample_rate,

            "segment_count": len(segment_records),

            "word_count": word_count,

            "unique_word_count": unique_word_count,

            "speech_duration": speech_duration,

            "total_gap_duration": total_gap_duration,

            "mean_gap_duration": mean_gap_duration,

            "max_gap_duration": max_gap_duration,

            "words_per_minute": words_per_minute,

            "processing_time": processing_time,

            "rtf": rtf,

            "segments_json": json.dumps(
                segment_records,
                ensure_ascii=False
            ),

            "status": "success",

            "error": ""

        }

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        row_df = pd.DataFrame([row])

        file_exists = OUTPUT_CSV.exists()

        row_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=not file_exists,
            index=False
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"Audio: {audio_duration:.1f}s | "
            f"Time: {processing_time:.2f}s | "
            f"RTF: {rtf:.3f} | "
            f"Segments: {len(segment_records)} | "
            f"Words: {word_count}"
        )

        print(
            f"Transcript: {transcript[:180]}"
        )

    except Exception as e:

        print(f"ERROR: {e}")

        # Save failures too, so the script knows
        # what happened and doesn't lose progress.

        error_row = {

            "filename": filename,

            "label": label_map.get(
                filename,
                None
            ),

            "transcript": "",

            "audio_duration": 0,

            "sample_rate": 0,

            "segment_count": 0,

            "word_count": 0,

            "unique_word_count": 0,

            "speech_duration": 0,

            "total_gap_duration": 0,

            "mean_gap_duration": 0,

            "max_gap_duration": 0,

            "words_per_minute": 0,

            "processing_time": 0,

            "rtf": 0,

            "segments_json": "[]",

            "status": "error",

            "error": str(e)

        }

        error_df = pd.DataFrame([error_row])

        file_exists = OUTPUT_CSV.exists()

        error_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=not file_exists,
            index=False
        )


# ============================================================
# FINISHED
# ============================================================

total_time = time.time() - start_all

print()
print("=" * 70)
print("TRANSCRIPTION COMPLETE")
print("=" * 70)

print(f"Total processing time: {total_time / 60:.2f} minutes")
print(f"Output file: {OUTPUT_CSV}")

if OUTPUT_CSV.exists():

    result_df = pd.read_csv(OUTPUT_CSV)

    print(f"Total saved rows: {len(result_df)}")
    print(
        "Successful:",
        (result_df["status"] == "success").sum()
    )
    print(
        "Errors:",
        (result_df["status"] == "error").sum()
    )