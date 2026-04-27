from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import warnings

warnings.filterwarnings("ignore")

# =========================
# INDSTILLINGER
# =========================
DATASET_PATH = Path(".")
TARGET_SR = 8000
SEGMENT_LENGTH = 2.5
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 5

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# =========================
# HJÆLPEFUNKTIONER
# =========================
def load_audio_segment(wav_path, target_sr=8000, segment_length=2.0):
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    # Trim stilhed væk
    y, _ = librosa.effects.trim(y, top_db=25)

    needed_samples = int(segment_length * sr)

    if len(y) < needed_samples:
        y = np.pad(y, (0, needed_samples - len(y)))
    else:
        y = y[:needed_samples]

    # DC-fjernelse + normalisering
    y = y - np.mean(y)
    y = y / (np.max(np.abs(y)) + 1e-12)

    return y, sr


def extract_features(y, sr):
    features = {}

    # Harmonisk del af signalet
    y_harm = librosa.effects.harmonic(y)

    # 1) Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    features["spectral_centroid"] = float(np.mean(centroid))

    # 2) MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    for i, val in enumerate(mfcc_mean, start=1):
        features[f"mfcc_{i}"] = float(val)

    # 3) Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(
        y=y_harm,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_chroma=12
    )
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    for note, val in zip(NOTE_NAMES, chroma_stft_mean):
        features[f"chroma_stft_{note}"] = float(val)

    # 4) Chroma CENS
    chroma_cens = librosa.feature.chroma_cens(
        y=y_harm,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=12,
        bins_per_octave=36,
        n_octaves=6
    )
    chroma_cens_mean = np.mean(chroma_cens, axis=1)
    for note, val in zip(NOTE_NAMES, chroma_cens_mean):
        features[f"chroma_cens_{note}"] = float(val)

    # 5) Tonnetz
    tonnetz = librosa.feature.tonnetz(chroma=chroma_stft, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    for i, val in enumerate(tonnetz_mean, start=1):
        features[f"tonnetz_{i}"] = float(val)

    return features


def build_dataset(dataset_path, target_sr=8000, segment_length=2.0):
    rows = []

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("."):
            continue

        wav_files = sorted(class_dir.glob("*.wav"))
        if len(wav_files) == 0:
            continue

        label = class_dir.name

        for wav_file in wav_files:
            y, sr = load_audio_segment(
                wav_file,
                target_sr=target_sr,
                segment_length=segment_length
            )
            feat = extract_features(y, sr)
            feat["label"] = label
            feat["filename"] = wav_file.name
            rows.append(feat)

    return pd.DataFrame(rows)


def feature_class_summary(df):
    feature_cols = [c for c in df.columns if c not in ["label", "filename"]]

    mean_table = df.groupby("label")[feature_cols].mean().T
    std_table = df.groupby("label")[feature_cols].std().T

    mean_table.columns = [f"{c}_mean" for c in mean_table.columns]
    std_table.columns = [f"{c}_std" for c in std_table.columns]

    summary = pd.concat([mean_table, std_table], axis=1)
    return summary


def feature_separation_score(df):
    feature_cols = [c for c in df.columns if c not in ["label", "filename"]]
    labels = sorted(df["label"].unique())

    results = []

    for feat in feature_cols:
        class_means = []
        class_stds = []

        for label in labels:
            values = df[df["label"] == label][feat].values
            class_means.append(np.mean(values))
            class_stds.append(np.std(values))

        between_class = np.std(class_means)
        within_class = np.mean(class_stds) + 1e-12
        score = between_class / within_class

        results.append({
            "feature": feat,
            "between_class_std": between_class,
            "within_class_std": within_class,
            "separation_score": score,
            "min_class_mean": np.min(class_means),
            "max_class_mean": np.max(class_means),
            "mean_range": np.max(class_means) - np.min(class_means)
        })

    results_df = pd.DataFrame(results).sort_values(
        "separation_score", ascending=False
    )

    return results_df


# =========================
# 1. TJEK DATASET
# =========================
print("\n===== ANTAL FILER PR. KLASSE =====")
for class_dir in sorted(DATASET_PATH.iterdir()):
    if not class_dir.is_dir():
        continue
    if class_dir.name.startswith("."):
        continue

    wav_files = list(class_dir.glob("*.wav"))
    if len(wav_files) == 0:
        continue

    print(f"{class_dir.name}: {len(wav_files)}")

# =========================
# 2. BYG FEATURE-DATASET
# =========================
print("\n===== BYGGER FEATURE-DATASET =====")
df = build_dataset(
    DATASET_PATH,
    target_sr=TARGET_SR,
    segment_length=SEGMENT_LENGTH
)

print(df.head())

# =========================
# 3. BESKRIVELSE
# =========================
print("\n===== BESKRIVELSE AF FEATURES =====")
print(df.describe())

# =========================
# 4. FEATURE-SUMMARY PR. KLASSE
# =========================
print("\n===== FEATURE-SUMMARY PR. KLASSE =====")
summary_df = feature_class_summary(df)
print(summary_df.head(20))

# =========================
# 5. FEATURES SORTERET EFTER ADSKILLELSE
# =========================
print("\n===== FEATURES SORTERET EFTER ADSKILLELSE =====")
separation_df = feature_separation_score(df)
print(separation_df.head(20))

# =========================
# 6. GEM RESULTATER
# =========================
df.to_csv("feature_dataset.csv", index=False)
summary_df.to_csv("feature_class_summary.csv")
separation_df.to_csv("feature_separation_scores.csv", index=False)

print("\nFiler gemt:")
print("- feature_dataset.csv")
print("- feature_class_summary.csv")
print("- feature_separation_scores.csv")