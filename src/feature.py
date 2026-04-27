from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================
# INDSTILLINGER
# =========================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR

TARGET_SR = 8000
N_SEGMENTS = 3

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 5

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def get_wav_files(folder):
    """
    Finder .wav-filer uden at tælle dem dobbelt.
    Virker både for .wav og .WAV.
    """
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".wav"
    ])


def load_audio(wav_path):
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    # Fjern stilhed
    y, _ = librosa.effects.trim(y, top_db=25)

    # Fjern DC og normalisér
    y = y - np.mean(y)
    y = y / (np.max(np.abs(y)) + 1e-12)

    return y, sr


def split_audio(y):
    """
    Deler én lydfil op i N_SEGMENTS dele.
    """
    segments = np.array_split(y, N_SEGMENTS)

    fixed_segments = []

    for seg in segments:
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))

        fixed_segments.append(seg)

    return fixed_segments


def extract_features(y, sr):
    features = {}

    y_harm = librosa.effects.harmonic(y)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    features["spectral_centroid"] = float(np.mean(centroid))

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    for i, val in enumerate(np.mean(mfcc, axis=1), start=1):
        features[f"mfcc_{i}"] = float(val)

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(
        y=y_harm,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_chroma=12
    )

    for note, val in zip(NOTE_NAMES, np.mean(chroma_stft, axis=1)):
        features[f"chroma_stft_{note}"] = float(val)

    # Chroma CENS
    chroma_cens = librosa.feature.chroma_cens(
        y=y_harm,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=12,
        bins_per_octave=36,
        n_octaves=6
    )

    for note, val in zip(NOTE_NAMES, np.mean(chroma_cens, axis=1)):
        features[f"chroma_cens_{note}"] = float(val)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(
        chroma=chroma_stft,
        sr=sr
    )

    for i, val in enumerate(np.mean(tonnetz, axis=1), start=1):
        features[f"tonnetz_{i}"] = float(val)

    return features


def build_dataset():
    rows = []

    for class_dir in sorted(DATASET_PATH.iterdir()):
        if not class_dir.is_dir():
            continue

        if class_dir.name.startswith(".") or class_dir.name == "src":
            continue

        wav_files = get_wav_files(class_dir)

        if len(wav_files) == 0:
            continue

        label = class_dir.name
        print(f"{label}: {len(wav_files)} filer")

        for wav_file in wav_files:
            y, sr = load_audio(wav_file)
            segments = split_audio(y)

            for segment_id, segment in enumerate(segments):
                features = extract_features(segment, sr)

                features["label"] = label
                features["filename"] = f"{wav_file.stem}_seg{segment_id}.wav"
                features["source_file"] = f"{label}/{wav_file.name}"
                features["segment_id"] = segment_id

                rows.append(features)

    return pd.DataFrame(rows)


# =========================
# KØR FEATURE EXTRACTION
# =========================
print("\n===== BYGGER FEATURE-DATASET =====")

df = build_dataset()

if df.empty:
    raise ValueError(
        "Ingen WAV-filer fundet. Tjek at akkordmapperne ligger direkte i projektmappen."
    )

output_path = PROJECT_DIR / "feature_dataset.csv"
df.to_csv(output_path, index=False)

print("\nFeature-datasæt gemt:")
print(output_path)

print("\nAntal samples:", len(df))

print("\nKlasser:")
print(df["label"].value_counts())