from pathlib import Path
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# PATHS / SETTINGS
# =========================
PROJECT_DIR = Path(__file__).resolve().parent

if PROJECT_DIR.name == "src":
    PROJECT_DIR = PROJECT_DIR.parent

os.chdir(PROJECT_DIR)

CSV_PATH = PROJECT_DIR / "c_feature2_dataset.csv"

PREDICTIONS_OUT = PROJECT_DIR / "knn_live_18_predictions.csv"
SUMMARY_OUT = PROJECT_DIR / "knn_live_18_summary.csv"
BEST_K_OUT = PROJECT_DIR / "knn_live_18_best_k.txt"
CONFUSION_OUT = PROJECT_DIR / "knn_live_18_confusion_matrix.csv"

META_COLS = ["label", "filename", "source_file", "segment_id", "label_encoded"]

QUANT_SCALE = 100.0
K_VALUES = [1, 3, 5, 7]

TEST_FILES_PER_CLASS = 2
BALANCE_TOTAL_FILES = True


# =========================
# FEATURE SELECTION
# =========================
def select_live_features(df):
    """
    Vælger de samme 18 live-features som bruges på Photon2:

    spectral_centroid
    mfcc_1 ... mfcc_5
    chroma_stft_C ... chroma_stft_B

    chroma_cens og tonnetz fjernes.
    """

    return [
        col for col in df.columns
        if col not in META_COLS
        and not col.startswith("chroma_cens")
        and not col.startswith("tonnetz")
    ]


# =========================
# BALANCED FILE SPLIT
# =========================
def make_balanced_file_split(df):
   
    file_df = df[["source_file", "label"]].drop_duplicates()

    files_by_label = {}

    for label in sorted(file_df["label"].unique()):
        label_files = sorted(
            file_df[file_df["label"] == label]["source_file"].tolist()
        )
        files_by_label[label] = label_files

    min_files_per_class = min(len(files) for files in files_by_label.values())

    if BALANCE_TOTAL_FILES:
        files_per_class = min_files_per_class
    else:
        files_per_class = None

    train_files = []
    test_files = []

    print("\n===== FILE SPLIT =====")

    for label in sorted(files_by_label):
        files = files_by_label[label]

        if BALANCE_TOTAL_FILES:
            files = files[:files_per_class]

        if len(files) <= TEST_FILES_PER_CLASS:
            raise ValueError(
                f"Klassen {label} har kun {len(files)} filer. "
                f"TEST_FILES_PER_CLASS={TEST_FILES_PER_CLASS} er for højt."
            )

        label_test_files = files[-TEST_FILES_PER_CLASS:]
        label_train_files = files[:-TEST_FILES_PER_CLASS]

        train_files.extend(label_train_files)
        test_files.extend(label_test_files)

        print(
            f"{label}: "
            f"train WAV-filer = {len(label_train_files)}, "
            f"test WAV-filer = {len(label_test_files)}"
        )

    train_df = df[df["source_file"].isin(train_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()

    return train_df, test_df


# =========================
# QUANTIZATION
# =========================
def quantize_features(x_scaled):
    """
    KNN bruger afstande, så features skaleres først med StandardScaler.
    Derefter kvantiseres de til int16_t, så Python-testen matcher Photon/emlearn-versionen.
    """

    xq = np.round(x_scaled * QUANT_SCALE)
    xq = np.clip(xq, -32768, 32767)

    return xq.astype(np.int16)


def print_prediction_distribution(name, pred):
    values, counts = np.unique(pred, return_counts=True)

    print(f"\n{name} prediction distribution:")
    for value, count in zip(values, counts):
        print(f"  class {value}: {count}")


# =========================
# MAIN
# =========================
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CSV_PATH}\n"
            "c_feature2_dataset.csv skal ligge i projektroden."
        )

    df = pd.read_csv(CSV_PATH)

    print("\n===== DATASET =====")
    print("Antal samples før balancering:", len(df))

    print("\nKlasser før balancering:")
    print(df["label"].value_counts().sort_index())

    feature_names = select_live_features(df)

    print("\nFeatures brugt:")
    for name in feature_names:
        print(" -", name)

    print("\nAntal features:", len(feature_names))

    if len(feature_names) != 18:
        print("\nWARNING:")
        print("Expected exactly 18 features.")
        print("Check that chroma_cens and tonnetz are removed.")

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])
    labels = list(encoder.classes_)

    train_df, test_df = make_balanced_file_split(df)

    x_train_raw = train_df[feature_names].values.astype(np.float32)
    x_test_raw = test_df[feature_names].values.astype(np.float32)

    y_train = train_df["label_encoded"].values
    y_test = test_df["label_encoded"].values

    print("\n===== TRAIN / TEST DATA =====")
    print("Train samples:", len(x_train_raw))
    print("Test samples:", len(x_test_raw))

    print("\nTrain-fordeling:")
    print(train_df["label"].value_counts().sort_index())

    print("\nTest-fordeling:")
    print(test_df["label"].value_counts().sort_index())

    print("\nTrain WAV-filer pr. klasse:")
    print(train_df[["source_file", "label"]].drop_duplicates()["label"].value_counts().sort_index())

    print("\nTest WAV-filer pr. klasse:")
    print(test_df[["source_file", "label"]].drop_duplicates()["label"].value_counts().sort_index())

    print("\nClasses:", labels)

    # KNN kræver scaling, fordi den bruger afstande.
    # Scaler fittes kun på train-data for at undgå data leakage.
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    x_train = quantize_features(x_train_scaled)
    x_test = quantize_features(x_test_scaled)

    best_acc = -1.0
    best_k = None
    best_pred = None

    results = []

    for k in K_VALUES:
        print(f"\n===== Testing KNN k={k} =====")

        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            metric="euclidean"
        )

        knn.fit(x_train, y_train)

        pred = knn.predict(x_test)
        acc = accuracy_score(y_test, pred)

        print(f"Python KNN test accuracy: {acc * 100:.2f}%")
        print_prediction_distribution("Python KNN", pred)

        results.append({
            "k": k,
            "accuracy": acc,
            "features": len(feature_names),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "quant_scale": QUANT_SCALE,
            "test_files_per_class": TEST_FILES_PER_CLASS,
        })

        if acc > best_acc:
            best_acc = acc
            best_k = k
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
    print(cm_df.to_string())

    print("\nClassification report:")
    print(classification_report(y_test, best_pred, target_names=labels))

    results_df = pd.DataFrame(results)
    results_df.to_csv(SUMMARY_OUT, index=False)

    predictions_df = pd.DataFrame({
        "source_file": test_df["source_file"].values,
        "segment_id": test_df["segment_id"].values,
        "true_label": encoder.inverse_transform(y_test),
        "predicted_label": encoder.inverse_transform(best_pred),
    })

    predictions_df.to_csv(PREDICTIONS_OUT, index=False)
    cm_df.to_csv(CONFUSION_OUT)
    BEST_K_OUT.write_text(str(best_k), encoding="utf-8")

    print("\nSaved:")
    print(" -", SUMMARY_OUT)
    print(" -", PREDICTIONS_OUT)
    print(" -", CONFUSION_OUT)
    print(" -", BEST_K_OUT)

if __name__ == "__main__":
    main()
