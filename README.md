# SHL Grammar Scoring Engine

Machine learning pipeline for predicting continuous spoken-English grammar scores from short audio responses.

## Problem

The task is to predict a continuous grammar score on a 0–5 scale from approximately 45–60 second spoken-English responses.

Evaluation uses:

- RMSE
- Pearson correlation

## Approach

The solution combines information from both speech and language representations.

### 1. Speech transcription

Spoken responses are transcribed using Whisper.

### 2. Acoustic features

Audio-level and prosodic characteristics are extracted from the speech signal, including timing, energy, pitch and related acoustic statistics.

### 3. Language representations

The pipeline uses multiple complementary text/speech representations, including:

- WavLM embeddings
- Whisper encoder representations
- sentence embeddings
- DeBERTa representations
- linguistic/text features

### 4. Out-of-fold stacking

Base-model predictions are generated with leakage-safe cross-validation.

The final prediction is obtained by combining several complementary base models through a validated ensemble.

## Validation

Training evaluation uses out-of-fold predictions rather than in-sample predictions.

The main validation metrics are:

- RMSE
- Pearson correlation

All train/test records are matched by filename.

## Final inference

The final inference pipeline:

1. Loads the trained feature representations / model outputs.
2. Generates predictions for the complete test set.
3. Combines the base-model predictions.
4. Clips predictions to the valid 0–5 score range.
5. Writes the final submission in the exact order of `test.csv`.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── final_submission.py
├── exp035_final_submission.py
├── exp035_correct_final_submission.py
├── exp032_final_submission.py
├── exp022_10fold_stacking.py
├── exp026_deberta_finetune.py
├── exp027a_whisper_encoder.py
├── baseline_acoustic.py
├── extract_acoustic_features.py
├── extract_linguistic_features.py
├── prepare_test_features.py
├── whisper_extract.py
├── check_submission_schema.py
└── check_audio_consistency.py