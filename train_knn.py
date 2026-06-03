from pathlib import Path
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# =========================
# PATHS
# =========================
PROJECT_DIR = Path(__file__).resolve().parent

# Hvis scriptet ligger i src, så gå én mappe tilbage til projektroden
if PROJECT_DIR.name == "src":
    PROJECT_DIR = PROJECT_DIR.parent

# Sørger for at relative output-filer ender i projektroden
os.chdir(PROJECT_DIR)

CSV_PATH = PROJECT_DIR / "c_feature2_dataset.csv"

PREDICTIONS_OUT = PROJECT_DIR / "knn_live_18_predictions.csv"
SUMMARY_OUT = PROJECT_DIR / "knn_live_18_summary.csv"
BEST_K_OUT = PROJECT_DIR / "knn_live_18_best_k.txt"

META_COLS = ["label", "filename", "source_file", "segment_id", "label_encoded"]

# Samme kvantisering som bruges i export_knn_emlearn.py
QUANT_SCALE = 100.0

# Samme k-værdier som testes i export-scriptet
K_VALUES = [1, 3, 5, 7]


# =========================
# FEATURE SELECTION
# =========================
def select_live_features(df):
    """
    Vælger de 18 features, som live feature_extractor på Photon2 bruger:

    spectral_centroid
    mfcc_1 ... mfcc_5
    chroma_stft_C ... chroma_stft_B

    Fjerner:
    chroma_cens
    tonnetz
    """

    feature_names = []

    for col in df.columns:
        if col in META_COLS:
            continue

        if col.startswith("chroma_cens"):
            continue

        if col.startswith("tonnetz"):
            continue

        feature_names.append(col)

    return feature_names


# =========================
# TRAIN / TEST SPLIT
# =========================
def split_train_test(df):
    """
    Splitter 80/20.

    Hvis source_file findes, splitter vi på filniveau.
    Det betyder, at segmenter fra samme WAV-fil ikke både kommer i train og test.
    """

    if "source_file" in df.columns:
        file_df = df[["source_file", "label"]].drop_duplicates()

        train_files, test_files = train_test_split(
            file_df,
            test_size=0.2,
            random_state=42,
            stratify=file_df["label"]
        )

        train_df = df[df["source_file"].isin(train_files["source_file"])].copy()
        test_df = df[df["source_file"].isin(test_files["source_file"])].copy()

    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label"]
        )

    return train_df, test_df


# =========================
# QUANTIZATION
# =========================
def quantize_features(X_scaled):
    """
    KNN bruger afstande, så vi scaler først med StandardScaler.
    Derefter kvantiserer vi til int16_t, så Python-testen matcher emlearn/Photon-versionen.
    """

    Xq = np.round(X_scaled * QUANT_SCALE)
    Xq = np.clip(Xq, -32768, 32767)

    return Xq.astype(np.int16)


# =========================
# PRINT HELPERS
# =========================
def print_prediction_distribution(name, pred):
    values, counts = np.unique(pred, return_counts=True)

    print(f"\n{name} prediction distribution:")
    for v, c in zip(values, counts):
        print(f"  class {v}: {c}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CSV_PATH}\n"
            "c_feature2_dataset.csv skal ligge i projektroden, ikke i src."
        )

    df = pd.read_csv(CSV_PATH)

    feature_names = select_live_features(df)

    print("\n===== KNN LIVE 18 FEATURES - TRAIN/ANALYSE =====")
    print("CSV:", CSV_PATH)
    print("Number of samples:", len(df))
    print("Number of features:", len(feature_names))

    print("\nSelected features:")
    for name in feature_names:
        print(" -", name)

    if len(feature_names) != 18:
        print("\nWARNING:")
        print("Expected exactly 18 features.")
        print("Check that chroma_cens and tonnetz are removed.")

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])
    labels = list(encoder.classes_)

    train_df, test_df = split_train_test(df)

    X_train_raw = train_df[feature_names].values.astype(np.float32)
    X_test_raw = test_df[feature_names].values.astype(np.float32)

    y_train = train_df["label_encoded"].values
    y_test = test_df["label_encoded"].values

    print("\n===== TRAIN / TEST SPLIT =====")
    print("Training samples:", len(X_train_raw))
    print("Test samples:", len(X_test_raw))

    if "source_file" in df.columns:
        print("Training files:", train_df["source_file"].nunique())
        print("Test files:", test_df["source_file"].nunique())

    print("Classes:", labels)

    # KNN kræver scaling, fordi den bruger afstande.
    # Scaler fittes kun på train-data for at undgå data leakage.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train = quantize_features(X_train_scaled)
    X_test = quantize_features(X_test_scaled)

    best_acc = -1.0
    best_k = None
    best_model = None
    best_pred = None

    results = []

    for k in K_VALUES:
        print(f"\n===== Testing KNN k={k} =====")

        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            metric="euclidean"
        )

        knn.fit(X_train, y_train)

        pred = knn.predict(X_test)
        acc = accuracy_score(y_test, pred)

        print(f"Python KNN test accuracy: {acc * 100:.2f}%")
        print_prediction_distribution("Python KNN", pred)

        results.append({
            "k": k,
            "accuracy": acc,
            "features": len(feature_names),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "quant_scale": QUANT_SCALE,
        })

        # Hvis to modeller har samme accuracy, beholdes laveste k,
        # fordi den er hurtigere og simplere på mikrocontrolleren.
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = knn
            best_pred = pred

    print("\n===== BEST KNN MODEL =====")
    print("Best k:", best_k)
    print(f"Best Python KNN test accuracy: {best_acc * 100:.2f}%")

    cm = confusion_matrix(y_test, best_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels]
    )

    print("\nConfusion matrix:")
    print(cm_df)

    print("\nClassification report:")
    print(classification_report(y_test, best_pred, target_names=labels))

    results_df = pd.DataFrame(results)
    results_df.to_csv(SUMMARY_OUT, index=False)

    predictions_df = pd.DataFrame({
        "source_file": test_df["source_file"].values if "source_file" in test_df.columns else "",
        "segment_id": test_df["segment_id"].values if "segment_id" in test_df.columns else "",
        "true_label": encoder.inverse_transform(y_test),
        "predicted_label": encoder.inverse_transform(best_pred),
    })

    predictions_df.to_csv(PREDICTIONS_OUT, index=False)
    BEST_K_OUT.write_text(str(best_k), encoding="utf-8")

    print("\nSaved:")
    print(" -", SUMMARY_OUT)
    print(" -", PREDICTIONS_OUT)
    print(" -", BEST_K_OUT)

    print("\nDone. This file does NOT export C headers. Use export_knn_emlearn.py for that.")


if __name__ == "__main__":
    main()
