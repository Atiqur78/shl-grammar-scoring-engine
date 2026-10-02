import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\kagle_project\Dataset_Final")

INPUT_CSV = ROOT / "whisper_transcripts.csv"
OUTPUT_CSV = ROOT / "linguistic_features.csv"


# ============================================================
# WORD LISTS
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def safe_ratio(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


def extract_features(row):

    text = str(row["transcript"]).strip()

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

    unique_word_count = len(unique_words)

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

    sentence_count = len(sentences)

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

        avg_word_length = np.mean(word_lengths)
        std_word_length = np.std(word_lengths)
        max_word_length = np.max(word_lengths)

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
    # Hapax / rare words
    # --------------------------------------------------------

    word_counts = {}

    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

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

    for i in range(1, len(words)):

        if words[i] == words[i - 1]:
            adjacent_repetitions += 1

    # --------------------------------------------------------
    # Repeated bigrams
    # --------------------------------------------------------

    bigrams = [
        (words[i], words[i + 1])
        for i in range(len(words) - 1)
    ]

    bigram_counts = {}

    for bg in bigrams:
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1

    repeated_bigram_count = sum(
        1
        for count in bigram_counts.values()
        if count > 1
    )

    # --------------------------------------------------------
    # Special word groups
    # --------------------------------------------------------

    filler_count = sum(
        1 for w in words
        if w in FILLER_WORDS
    )

    pronoun_count = sum(
        1 for w in words
        if w in PRONOUNS
    )

    conjunction_count = sum(
        1 for w in words
        if w in CONJUNCTIONS
    )

    modal_count = sum(
        1 for w in words
        if w in MODAL_WORDS
    )

    common_verb_count = sum(
        1 for w in words
        if w in COMMON_VERBS
    )

    # --------------------------------------------------------
    # Specific grammatical proxies
    # --------------------------------------------------------

    past_tense_like = sum(
        1
        for w in words
        if w.endswith("ed")
    )

    plural_like = sum(
        1
        for w in words
        if len(w) > 3 and w.endswith("s")
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
    # Audio statistics already produced by Whisper
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
    # Return feature dictionary
    # --------------------------------------------------------

    return {

        "filename": row["filename"],

        "label": row["label"],

        # Basic lexical features
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "type_token_ratio": type_token_ratio,
        "hapax_count": hapax_count,
        "repeated_word_count": repeated_word_count,
        "adjacent_repetitions": adjacent_repetitions,
        "repeated_bigram_count": repeated_bigram_count,

        # Sentence features
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "std_sentence_length": std_sentence_length,
        "max_sentence_length": max_sentence_length,
        "min_sentence_length": min_sentence_length,

        # Word features
        "avg_word_length": avg_word_length,
        "std_word_length": std_word_length,
        "max_word_length": max_word_length,

        # Grammar-related proxies
        "conjunction_count": conjunction_count,
        "conjunction_ratio": safe_ratio(
            conjunction_count,
            word_count
        ),

        "pronoun_count": pronoun_count,
        "pronoun_ratio": safe_ratio(
            pronoun_count,
            word_count
        ),

        "modal_count": modal_count,
        "modal_ratio": safe_ratio(
            modal_count,
            word_count
        ),

        "common_verb_count": common_verb_count,

        "past_tense_like": past_tense_like,
        "past_tense_ratio": safe_ratio(
            past_tense_like,
            word_count
        ),

        "plural_like": plural_like,

        "ing_words": ing_words,
        "ing_ratio": safe_ratio(
            ing_words,
            word_count
        ),

        "contraction_count": contraction_count,

        # Fluency / repetition
        "filler_count": filler_count,
        "filler_ratio": safe_ratio(
            filler_count,
            word_count
        ),

        "discourse_marker_count": discourse_marker_count,

        # Punctuation / sentence structure
        "comma_count": comma_count,
        "comma_per_sentence": safe_ratio(
            comma_count,
            sentence_count
        ),

        "semicolon_count": semicolon_count,

        "question_count": question_count,

        "exclamation_count": exclamation_count,

        # Text size
        "character_count": character_count,
        "avg_chars_per_word": safe_ratio(
            character_count,
            word_count
        ),

        # Whisper timing
        "audio_duration": audio_duration,
        "speech_duration": speech_duration,
        "segment_count": segment_count,
        "whisper_wpm": whisper_wpm,

        # Derived speaking measures
        "words_per_audio_second": safe_ratio(
            word_count,
            audio_duration
        ),

        "words_per_speech_second": safe_ratio(
            word_count,
            speech_duration
        ),

        "speech_ratio": safe_ratio(
            speech_duration,
            audio_duration
        )
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("LINGUISTIC FEATURE EXTRACTION")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)

print(f"Input rows: {len(df)}")

features = []

for i, (_, row) in enumerate(df.iterrows(), start=1):

    features.append(
        extract_features(row)
    )

    if i % 100 == 0:
        print(f"Processed: {i}/{len(df)}")


features_df = pd.DataFrame(features)

features_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(f"Rows: {len(features_df)}")
print(f"Columns: {len(features_df.columns)}")
print(f"Output: {OUTPUT_CSV}")

print()
print("Feature columns:")

for col in features_df.columns:
    print(" ", col)