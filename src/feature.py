from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import warnings

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =========================
# INDSTILLINGER
# =========================
DATASET_PATH = Path(".")
SEGMENT_LENGTHS = [0.5, 1.0, 2.0, 2.5]
TARGET_SAMPLE_RATES = [8000]
N_FFT = 2048
HOP_LENGTH = 512
RANDOM_STATE = 42

# =========================
# HJÆLPEFUNKTIONER
# =========================
def load_audio_segment(wav_path, target_sr=8000, segment_length=2.0):
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    # trim stilhed væk
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

    # 1) RMS
    rms = np.sqrt(np.mean(y**2))
    features["rms"] = float(rms)

    # 2) Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr"] = float(np.mean(zcr))

    # 3) Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    features["spectral_centroid"] = float(np.mean(centroid))

    # 4) MFCC (brug gennemsnit af de første 5)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=5, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    for i, val in enumerate(mfcc_mean, start=1):
        features[f"mfcc_{i}"] = float(val)

    # 5) Chroma STFT i stedet for Chroma CQT
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_chroma=12
    )
    chroma_mean = np.mean(chroma, axis=1)

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    for note, val in zip(note_names, chroma_mean):
        features[f"chroma_{note}"] = float(val)

    return features

def build_dataset(dataset_path, target_sr=8000, segment_length=2.0):
    rows = []

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for wav_file in sorted(class_dir.glob("*.wav")):
            y, sr = load_audio_segment(wav_file, target_sr=target_sr, segment_length=segment_length)
            feat = extract_features(y, sr)
            feat["label"] = label
            feat["filename"] = wav_file.name
            rows.append(feat)

    return pd.DataFrame(rows)


def evaluate_dataset(df):
    feature_cols = [c for c in df.columns if c not in ["label", "filename"]]
    X = df[feature_cols].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")

    return scores.mean(), scores.std(), feature_cols


def top_features(df, n_top=5):
    feature_cols = [c for c in df.columns if c not in ["label", "filename"]]
    X = df[feature_cols].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE
    )

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False)

    return imp.head(n_top), imp


# =========================
# 1. TJEK DATASET
# =========================
print("\n===== ANTAL FILER PR. KLASSE =====")
for class_dir in sorted(DATASET_PATH.iterdir()):
    if class_dir.is_dir():
        n_files = len(list(class_dir.glob("*.wav")))
        print(f"{class_dir.name}: {n_files}")

# =========================
# 2. TEST SAMPLELÆNGDE
# =========================
print("\n===== TEST AF SAMPLELÆNGDE =====")
length_results = []

for seg_len in SEGMENT_LENGTHS:
    df = build_dataset(DATASET_PATH, target_sr=8000, segment_length=seg_len)
    mean_acc, std_acc, _ = evaluate_dataset(df)
    length_results.append((seg_len, mean_acc, std_acc))
    print(f"{seg_len:.1f} s -> accuracy = {mean_acc:.3f} ± {std_acc:.3f}")

best_segment = max(length_results, key=lambda x: x[1])[0]
print(f"\nBedste sample-længde: {best_segment:.1f} s")

# =========================
# 3. TEST SAMPLE RATE
# =========================
print("\n===== TEST AF SAMPLE RATE =====")
sr_results = []

for sr_test in TARGET_SAMPLE_RATES:
    df = build_dataset(DATASET_PATH, target_sr=sr_test, segment_length=best_segment)
    mean_acc, std_acc, _ = evaluate_dataset(df)
    sr_results.append((sr_test, mean_acc, std_acc))
    print(f"{sr_test} Hz -> accuracy = {mean_acc:.3f} ± {std_acc:.3f}")

best_sr = max(sr_results, key=lambda x: x[1])[0]
print(f"\nBedste sample rate: {best_sr} Hz")

# =========================
# 4. ENDELIGT DATASET
# =========================
print("\n===== BYGGER ENDELIGT DATASET =====")
df_final = build_dataset(DATASET_PATH, target_sr=best_sr, segment_length=best_segment)

print(df_final.head())
print("\nBeskrivelse:")
print(df_final.describe())

# =========================
# 5. MEST BESKRIVENDE FEATURES
# =========================
print("\n===== TOP 5 FEATURES =====")
top5, all_features = top_features(df_final, n_top=5)
print(top5)

# =========================
# 6. ENDELIG MODELKVALITET
# =========================
print("\n===== ENDELIG EVALUERING =====")
mean_acc, std_acc, feature_cols = evaluate_dataset(df_final)
print(f"Accuracy = {mean_acc:.3f} ± {std_acc:.3f}")

# =========================
# 7. SVAR PÅ 'FULL CYCLE'
# =========================
print("\n===== FULL CYCLE =====")
lowest_guitar_freq = 82.4  # E2
samples_per_cycle = best_sr / lowest_guitar_freq
print(f"Laveste guitarfrekvens antaget: {lowest_guitar_freq} Hz")
print(f"Samples pr. periode ved {best_sr} Hz: {samples_per_cycle:.1f}")

# =========================
# 8. WINDOWING / OVERLAP
# =========================
print("\n===== WINDOWING =====")
window_duration = N_FFT / best_sr
overlap = 1 - (HOP_LENGTH / N_FFT)
print(f"Vindueslængde: {window_duration:.3f} s")
print(f"Overlap: {overlap*100:.1f}%")

# =========================
# 9. GEM FEATURES
# =========================
df_final.to_csv("feature_dataset.csv", index=False)
print("\nFeature-datasæt gemt som feature_dataset.csv")