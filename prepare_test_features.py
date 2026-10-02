import json
import re
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from faster_whisper import WhisperModel


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\kagle_project\Dataset_Final")

TEST_DIR = ROOT / "test"
TEST_CSV = ROOT / "test.csv"

WHISPER_OUTPUT = ROOT / "test_whisper_transcripts.csv"
ACOUSTIC_OUTPUT = ROOT / "test_acoustic_features.csv"
LINGUISTIC_OUTPUT = ROOT / "test_linguistic_features.csv"


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

WHISPER_MODEL = "base"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"


# ============================================================
# LOAD TEST CSV
# ============================================================

test_df = pd.read_csv(TEST_CSV)

test_filenames = test_df["filename"].tolist()

print("=" * 70)
print("TEST FEATURE PREPARATION")
print("=" * 70)

print(
    "Test rows:",
    len(test_df)
)


# ============================================================
# VERIFY AUDIO FILES
# ============================================================

missing_audio = [
    filename
    for filename in test_filenames
    if not (TEST_DIR / filename).exists()
]

if missing_audio:

    print()
    print("Missing test audio files:")

    for filename in missing_audio:
        print(" ", filename)

    raise FileNotFoundError(
        f"{len(missing_audio)} test audio files are missing."
    )


print(
    "All test audio files found: OK"
)


# ============================================================
# PART 1 — WHISPER TRANSCRIPTION
# ============================================================

print()
print("=" * 70)
print("PART 1 — WHISPER TEST TRANSCRIPTION")
print("=" * 70)

print(
    "Model:",
    WHISPER_MODEL
)

print(
    "Device:",
    WHISPER_DEVICE
)

print(
    "Compute type:",
    WHISPER_COMPUTE_TYPE
)


model = WhisperModel(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)


whisper_results = []

start_time = time.time()


for index, filename in enumerate(
    tqdm(
        test_filenames,
        total=len(test_filenames)
    ),
    start=1
):

    audio_path = TEST_DIR / filename

    try:

        # ----------------------------------------------------
        # Load WAV ourselves.
        #
        # This avoids the PyAV metadata_errors compatibility
        # issue encountered during the original extraction.
        # ----------------------------------------------------

        import soundfile as sf

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1
            )

        audio_duration = (
            len(audio) / sample_rate
        )

        # ----------------------------------------------------
        # Same Whisper configuration used previously
        # ----------------------------------------------------

        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False
        )

        segments = list(segments)

        transcript_parts = []

        speech_duration = 0.0

        for segment in segments:

            text = segment.text.strip()

            if text:
                transcript_parts.append(text)

            speech_duration += max(
                0.0,
                segment.end - segment.start
            )

        transcript = " ".join(
            transcript_parts
        ).strip()

        # ----------------------------------------------------
        # Word statistics
        # ----------------------------------------------------

        words = re.findall(
            r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
            transcript.lower()
        )

        word_count = len(words)

        words_per_minute = (
            word_count /
            audio_duration *
            60.0
            if audio_duration > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Segment timing
        # ----------------------------------------------------

        gap_durations = []

        for i in range(
            1,
            len(segments)
        ):

            gap = (
                segments[i].start
                -
                segments[i - 1].end
            )

            if gap > 0:
                gap_durations.append(gap)

        total_gap_duration = sum(
            gap_durations
        )

        mean_gap_duration = (
            np.mean(gap_durations)
            if gap_durations
            else 0.0
        )

        max_gap_duration = (
            np.max(gap_durations)
            if gap_durations
            else 0.0
        )

        whisper_results.append({

            "filename":
                filename,

            "transcript":
                transcript,

            "audio_duration":
                audio_duration,

            "sample_rate":
                sample_rate,

            "segment_count":
                len(segments),

            "word_count":
                word_count,

            "unique_word_count":
                len(set(words)),

            "speech_duration":
                speech_duration,

            "total_gap_duration":
                total_gap_duration,

            "mean_gap_duration":
                mean_gap_duration,

            "max_gap_duration":
                max_gap_duration,

            "words_per_minute":
                words_per_minute,

            "segments_json":
                json.dumps([
                    {
                        "start":
                            float(s.start),

                        "end":
                            float(s.end),

                        "text":
                            s.text.strip()
                    }

                    for s in segments
                ])
        })

    except Exception as e:

        print()
        print(
            f"ERROR: {filename}: {e}"
        )

        whisper_results.append({

            "filename":
                filename,

            "transcript":
                "",

            "audio_duration":
                0.0,

            "sample_rate":
                16000,

            "segment_count":
                0,

            "word_count":
                0,

            "unique_word_count":
                0,

            "speech_duration":
                0.0,

            "total_gap_duration":
                0.0,

            "mean_gap_duration":
                0.0,

            "max_gap_duration":
                0.0,

            "words_per_minute":
                0.0,

            "segments_json":
                "[]"
        })


whisper_df = pd.DataFrame(
    whisper_results
)


whisper_df.to_csv(
    WHISPER_OUTPUT,
    index=False
)


elapsed = (
    time.time()
    -
    start_time
)


print()
print(
    "Whisper rows:",
    len(whisper_df)
)

print(
    "Processing time:",
    f"{elapsed / 60:.2f} minutes"
)

print(
    "Saved:",
    WHISPER_OUTPUT
)


# ============================================================
# WHISPER VALIDATION
# ============================================================

assert len(whisper_df) == len(test_df)

assert set(
    whisper_df["filename"]
) == set(test_filenames)

assert (
    whisper_df["transcript"]
    .notna()
    .all()
)

print(
    "Whisper filename validation: OK"
)


# ============================================================
# PART 2 — ACOUSTIC FEATURES
# ============================================================

print()
print("=" * 70)
print("PART 2 — ACOUSTIC TEST FEATURES")
print("=" * 70)


def summarize(x, prefix):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return {

            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0
        }

    return {

        f"{prefix}_mean":
            float(np.mean(x)),

        f"{prefix}_std":
            float(np.std(x)),

        f"{prefix}_median":
            float(np.median(x)),

        f"{prefix}_min":
            float(np.min(x)),

        f"{prefix}_max":
            float(np.max(x)),

        f"{prefix}_q25":
            float(np.percentile(x, 25)),

        f"{prefix}_q75":
            float(np.percentile(x, 75))
    }


def extract_acoustic_features(path):

    y, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    features = {}

    # --------------------------------------------------------
    # Basic audio properties
    # --------------------------------------------------------

    duration = len(y) / sr

    features["duration"] = duration

    features["audio_rms"] = float(
        np.sqrt(np.mean(y ** 2))
    )

    features["audio_abs_mean"] = float(
        np.mean(np.abs(y))
    )

    # --------------------------------------------------------
    # Frame-level features
    # --------------------------------------------------------

    hop_length = 512
    n_fft = 2048

    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    # --------------------------------------------------------
    # Aggregate features
    # --------------------------------------------------------

    features.update(
        summarize(rms, "rms")
    )

    features.update(
        summarize(zcr, "zcr")
    )

    features.update(
        summarize(centroid, "centroid")
    )

    features.update(
        summarize(bandwidth, "bandwidth")
    )

    features.update(
        summarize(rolloff, "rolloff")
    )

    features.update(
        summarize(flatness, "flatness")
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length
    )

    for i in range(
        mfcc.shape[0]
    ):

        features.update(
            summarize(
                mfcc[i],
                f"mfcc_{i + 1}"
            )
        )

    # --------------------------------------------------------
    # Spectral contrast
    # --------------------------------------------------------

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    for i in range(
        contrast.shape[0]
    ):

        features.update(
            summarize(
                contrast[i],
                f"contrast_{i + 1}"
            )
        )

    # --------------------------------------------------------
    # Silence / energy
    # --------------------------------------------------------

    rms_threshold = np.percentile(
        rms,
        20
    )

    low_energy_ratio = np.mean(
        rms <= rms_threshold
    )

    features[
        "low_energy_ratio"
    ] = float(
        low_energy_ratio
    )

    low_energy = (
        rms <= rms_threshold
    )

    transitions = np.sum(
        low_energy[1:]
        !=
        low_energy[:-1]
    )

    features[
        "energy_transitions"
    ] = float(
        transitions
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    f0, voiced_flag, voiced_prob = (
        librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048,
            hop_length=hop_length
        )
    )

    valid_f0 = f0[
        np.isfinite(f0)
    ]

    if len(valid_f0) > 0:

        features["pitch_mean"] = float(
            np.mean(valid_f0)
        )

        features["pitch_std"] = float(
            np.std(valid_f0)
        )

        features["pitch_median"] = float(
            np.median(valid_f0)
        )

        features["pitch_min"] = float(
            np.min(valid_f0)
        )

        features["pitch_max"] = float(
            np.max(valid_f0)
        )

        features["pitch_range"] = float(
            np.max(valid_f0)
            -
            np.min(valid_f0)
        )

    else:

        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_median"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0
        features["pitch_range"] = 0.0

    features["voiced_ratio"] = float(
        np.mean(voiced_flag)
    )

    features[
        "voiced_probability_mean"
    ] = float(
        np.nanmean(voiced_prob)
    )

    return features


acoustic_results = []

acoustic_start = time.time()


for filename in tqdm(
    test_filenames,
    total=len(test_filenames)
):

    path = TEST_DIR / filename

    try:

        features = extract_acoustic_features(
            path
        )

        features[
            "filename"
        ] = filename

        acoustic_results.append(
            features
        )

    except Exception as e:

        print(
            f"ERROR: {filename}: {e}"
        )


test_acoustic_df = pd.DataFrame(
    acoustic_results
)


# Put filename first
acoustic_columns = (
    ["filename"]
    +
    [
        c
        for c in test_acoustic_df.columns
        if c != "filename"
    ]
)

test_acoustic_df = (
    test_acoustic_df[
        acoustic_columns
    ]
)


test_acoustic_df.to_csv(
    ACOUSTIC_OUTPUT,
    index=False
)


print()
print(
    "Acoustic shape:",
    test_acoustic_df.shape
)

print(
    "Saved:",
    ACOUSTIC_OUTPUT
)

print(
    "Acoustic processing time:",
    f"{(time.time() - acoustic_start) / 60:.2f} minutes"
)


# ============================================================
# ACOUSTIC VALIDATION
# ============================================================

assert len(
    test_acoustic_df
) == len(test_df)

assert set(
    test_acoustic_df["filename"]
) == set(test_filenames)

assert (
    test_acoustic_df.isna().sum().sum()
    == 0
)

print(
    "Acoustic filename validation: OK"
)


# ============================================================
# PART 3 — LINGUISTIC FEATURES
# ============================================================

print()
print("=" * 70)
print("PART 3 — LINGUISTIC TEST FEATURES")
print("=" * 70)


FILLER_WORDS = {
    "um", "uh", "er", "erm", "hmm", "hmmm",
    "like", "actually", "basically", "you know"
}


PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves"
}


CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so",
    "because", "although", "though", "while",
    "if", "when", "where", "unless", "since",
    "before", "after", "until", "whether"
}


MODAL_WORDS = {
    "can", "could", "may", "might", "must",
    "shall", "should", "will", "would"
}


COMMON_VERBS = {
    "be", "am", "is", "are", "was", "were",
    "have", "has", "had",
    "do", "does", "did",
    "go", "went", "come", "came",
    "make", "made", "get", "got",
    "take", "took", "give", "gave",
    "see", "saw", "know", "knew",
    "think", "thought", "want", "wanted",
    "like", "liked", "love", "loved",
    "play", "played", "use", "used"
}


def safe_ratio(a, b):

    if b == 0:
        return 0.0

    return float(a) / float(b)


def extract_linguistic_features(row):

    text = str(
        row["transcript"]
    ).strip()

    text_lower = text.lower()

    # --------------------------------------------------------
    # Words
    # --------------------------------------------------------

    words = re.findall(
        r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
        text_lower
    )

    word_count = len(words)

    unique_words = set(words)

    unique_word_count = len(
        unique_words
    )

    # --------------------------------------------------------
    # Sentences
    # --------------------------------------------------------

    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    sentence_count = len(
        sentences
    )

    sentence_lengths = []

    for sentence in sentences:

        sentence_words = re.findall(
            r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
            sentence.lower()
        )

        if sentence_words:

            sentence_lengths.append(
                len(sentence_words)
            )

    if sentence_lengths:

        avg_sentence_length = np.mean(
            sentence_lengths
        )

        std_sentence_length = np.std(
            sentence_lengths
        )

        max_sentence_length = np.max(
            sentence_lengths
        )

        min_sentence_length = np.min(
            sentence_lengths
        )

    else:

        avg_sentence_length = 0
        std_sentence_length = 0
        max_sentence_length = 0
        min_sentence_length = 0

    # --------------------------------------------------------
    # Word length
    # --------------------------------------------------------

    word_lengths = [
        len(w.replace("'", ""))
        for w in words
    ]

    if word_lengths:

        avg_word_length = np.mean(
            word_lengths
        )

        std_word_length = np.std(
            word_lengths
        )

        max_word_length = np.max(
            word_lengths
        )

    else:

        avg_word_length = 0
        std_word_length = 0
        max_word_length = 0

    # --------------------------------------------------------
    # Vocabulary diversity
    # --------------------------------------------------------

    type_token_ratio = safe_ratio(
        unique_word_count,
        word_count
    )

    # --------------------------------------------------------
    # Hapax
    # --------------------------------------------------------

    word_counts = {}

    for word in words:

        word_counts[word] = (
            word_counts.get(word, 0)
            + 1
        )

    hapax_count = sum(
        1
        for count in word_counts.values()
        if count == 1
    )

    repeated_word_count = sum(
        1
        for count in word_counts.values()
        if count > 1
    )

    # --------------------------------------------------------
    # Adjacent repetition
    # --------------------------------------------------------

    adjacent_repetitions = 0

    for i in range(
        1,
        len(words)
    ):

        if words[i] == words[i - 1]:

            adjacent_repetitions += 1

    # --------------------------------------------------------
    # Repeated bigrams
    # --------------------------------------------------------

    bigrams = [
        (
            words[i],
            words[i + 1]
        )
        for i in range(
            len(words) - 1
        )
    ]

    bigram_counts = {}

    for bg in bigrams:

        bigram_counts[bg] = (
            bigram_counts.get(bg, 0)
            + 1
        )

    repeated_bigram_count = sum(
        1
        for count in bigram_counts.values()
        if count > 1
    )

    # --------------------------------------------------------
    # Word groups
    # --------------------------------------------------------

    filler_count = sum(
        1
        for w in words
        if w in FILLER_WORDS
    )

    pronoun_count = sum(
        1
        for w in words
        if w in PRONOUNS
    )

    conjunction_count = sum(
        1
        for w in words
        if w in CONJUNCTIONS
    )

    modal_count = sum(
        1
        for w in words
        if w in MODAL_WORDS
    )

    common_verb_count = sum(
        1
        for w in words
        if w in COMMON_VERBS
    )

    # --------------------------------------------------------
    # Grammar proxies
    # --------------------------------------------------------

    past_tense_like = sum(
        1
        for w in words
        if w.endswith("ed")
    )

    plural_like = sum(
        1
        for w in words
        if len(w) > 3
        and w.endswith("s")
    )

    ing_words = sum(
        1
        for w in words
        if w.endswith("ing")
    )

    # --------------------------------------------------------
    # Contractions
    # --------------------------------------------------------

    contraction_count = len(
        re.findall(
            r"\b[a-zA-Z]+['’][a-zA-Z]+\b",
            text
        )
    )

    # --------------------------------------------------------
    # Punctuation
    # --------------------------------------------------------

    comma_count = text.count(",")

    semicolon_count = text.count(";")

    question_count = text.count("?")

    exclamation_count = text.count("!")

    # --------------------------------------------------------
    # Discourse markers
    # --------------------------------------------------------

    discourse_markers = {
        "well",
        "so",
        "then",
        "also",
        "because",
        "however",
        "therefore",
        "first",
        "finally",
        "actually",
        "basically"
    }

    discourse_marker_count = sum(
        1
        for w in words
        if w in discourse_markers
    )

    # --------------------------------------------------------
    # Character statistics
    # --------------------------------------------------------

    character_count = len(text)

    # --------------------------------------------------------
    # Whisper statistics
    # --------------------------------------------------------

    audio_duration = float(
        row["audio_duration"]
    )

    speech_duration = float(
        row["speech_duration"]
    )

    whisper_wpm = float(
        row["words_per_minute"]
    )

    segment_count = int(
        row["segment_count"]
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "filename":
            row["filename"],

        # Basic lexical
        "word_count":
            word_count,

        "unique_word_count":
            unique_word_count,

        "type_token_ratio":
            type_token_ratio,

        "hapax_count":
            hapax_count,

        "repeated_word_count":
            repeated_word_count,

        "adjacent_repetitions":
            adjacent_repetitions,

        "repeated_bigram_count":
            repeated_bigram_count,

        # Sentence
        "sentence_count":
            sentence_count,

        "avg_sentence_length":
            avg_sentence_length,

        "std_sentence_length":
            std_sentence_length,

        "max_sentence_length":
            max_sentence_length,

        "min_sentence_length":
            min_sentence_length,

        # Word
        "avg_word_length":
            avg_word_length,

        "std_word_length":
            std_word_length,

        "max_word_length":
            max_word_length,

        # Grammar proxies
        "conjunction_count":
            conjunction_count,

        "conjunction_ratio":
            safe_ratio(
                conjunction_count,
                word_count
            ),

        "pronoun_count":
            pronoun_count,

        "pronoun_ratio":
            safe_ratio(
                pronoun_count,
                word_count
            ),

        "modal_count":
            modal_count,

        "modal_ratio":
            safe_ratio(
                modal_count,
                word_count
            ),

        "common_verb_count":
            common_verb_count,

        "past_tense_like":
            past_tense_like,

        "past_tense_ratio":
            safe_ratio(
                past_tense_like,
                word_count
            ),

        "plural_like":
            plural_like,

        "ing_words":
            ing_words,

        "ing_ratio":
            safe_ratio(
                ing_words,
                word_count
            ),

        "contraction_count":
            contraction_count,

        # Fluency
        "filler_count":
            filler_count,

        "filler_ratio":
            safe_ratio(
                filler_count,
                word_count
            ),

        "discourse_marker_count":
            discourse_marker_count,

        # Punctuation
        "comma_count":
            comma_count,

        "comma_per_sentence":
            safe_ratio(
                comma_count,
                sentence_count
            ),

        "semicolon_count":
            semicolon_count,

        "question_count":
            question_count,

        "exclamation_count":
            exclamation_count,

        # Text
        "character_count":
            character_count,

        "avg_chars_per_word":
            safe_ratio(
                character_count,
                word_count
            ),

        # Whisper timing
        "audio_duration":
            audio_duration,

        "speech_duration":
            speech_duration,

        "segment_count":
            segment_count,

        "whisper_wpm":
            whisper_wpm,

        # Derived
        "words_per_audio_second":
            safe_ratio(
                word_count,
                audio_duration
            ),

        "words_per_speech_second":
            safe_ratio(
                word_count,
                speech_duration
            ),

        "speech_ratio":
            safe_ratio(
                speech_duration,
                audio_duration
            )
    }


linguistic_results = []

for _, row in tqdm(
    whisper_df.iterrows(),
    total=len(whisper_df)
):

    try:

        linguistic_results.append(
            extract_linguistic_features(
                row
            )
        )

    except Exception as e:

        print(
            f"ERROR: {row['filename']}: {e}"
        )


test_linguistic_df = pd.DataFrame(
    linguistic_results
)


test_linguistic_df.to_csv(
    LINGUISTIC_OUTPUT,
    index=False
)


print()
print(
    "Linguistic shape:",
    test_linguistic_df.shape
)

print(
    "Saved:",
    LINGUISTIC_OUTPUT
)


# ============================================================
# LINGUISTIC VALIDATION
# ============================================================

assert len(
    test_linguistic_df
) == len(test_df)

assert set(
    test_linguistic_df["filename"]
) == set(test_filenames)

assert (
    test_linguistic_df.isna().sum().sum()
    == 0
)

print(
    "Linguistic filename validation: OK"
)


# ============================================================
# FINAL FEATURE VALIDATION
# ============================================================

print()
print("=" * 70)
print("TEST FEATURE PREPARATION COMPLETE")
print("=" * 70)

print(
    "Test CSV:",
    len(test_df)
)

print(
    "Whisper rows:",
    len(whisper_df)
)

print(
    "Acoustic rows:",
    len(test_acoustic_df)
)

print(
    "Linguistic rows:",
    len(test_linguistic_df)
)

assert (
    len(test_df)
    ==
    len(whisper_df)
    ==
    len(test_acoustic_df)
    ==
    len(test_linguistic_df)
    ==
    216
)

assert (
    set(test_filenames)
    ==
    set(whisper_df["filename"])
    ==
    set(test_acoustic_df["filename"])
    ==
    set(test_linguistic_df["filename"])
)

print()
print("216/216 test samples have all features.")
print("Ready for final model training.")