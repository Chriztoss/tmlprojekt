from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# SETTINGS
# =========================
PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "c_feature2_dataset.csv"

META_COLS = ["label", "filename", "source_file", "segment_id", "label_encoded"]

TEST_FILES_PER_CLASS = 2

# Hvis True bruges samme antal WAV-filer fra hver klasse.
# Det betyder fx, at hvis nogle klasser har 11 eller 12 filer,
# men de fleste har 10, så bruges kun 10 fra hver klasse.
BALANCE_TOTAL_FILES = True


def is_live_feature(column_name):
    """
    Brug samme reducerede feature-set som live-koden på Photon2:
    spectral_centroid + 5 MFCC + 12 chroma_stft = 18 features.
    """
    if column_name in META_COLS:
        return False

    if column_name.startswith("chroma_cens"):
        return False

    if column_name.startswith("tonnetz"):
        return False

    return True


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

        # Deterministisk valg:
        # De sidste TEST_FILES_PER_CLASS filer bruges til test.
        # Resten bruges til træning.
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
# LOAD DATASET
# =========================
df = pd.read_csv(CSV_PATH)

print("\n===== DATASET =====")
print("Antal samples før balancering:", len(df))

print("\nKlasser før balancering:")
print(df["label"].value_counts().sort_index())


# =========================
# FEATURE SELECTION
# =========================
feature_columns = [c for c in df.columns if is_live_feature(c)]

print("\nFeatures brugt:")
for feature in feature_columns:
    print(" -", feature)

print("\nAntal features:", len(feature_columns))


# =========================
# BALANCED FILE SPLIT
# =========================
train_df, test_df = make_balanced_file_split(df)

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df["label"]
y_test = test_df["label"]

print("\n===== TRAIN / TEST DATA =====")
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTrain-fordeling:")
print(y_train.value_counts().sort_index())

print("\nTest-fordeling:")
print(y_test.value_counts().sort_index())

print("\nTrain WAV-filer pr. klasse:")
print(train_df[["source_file", "label"]].drop_duplicates()["label"].value_counts().sort_index())

print("\nTest WAV-filer pr. klasse:")
print(test_df[["source_file", "label"]].drop_duplicates()["label"].value_counts().sort_index())


# =========================
# RANDOM FOREST MODEL
# =========================
rf_clf = RandomForestClassifier(
    n_estimators=30,
    max_depth=8,
    random_state=42
)

rf_clf.fit(X_train, y_train)


# =========================
# TEST MODELLEN
# =========================
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===== RANDOM FOREST RESULTAT =====")
print("n_estimators = 30")
print("max_depth = 8")
print(f"Accuracy = {accuracy:.3f}")


# =========================
# CONFUSION MATRIX
# =========================
labels = sorted(y_train.unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in labels],
    columns=[f"pred_{label}" for label in labels]
)

print("\nConfusion matrix:")
print(cm_df.to_string())


# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification report:")
print(classification_report(y_test, y_pred, labels=labels))


# =========================
# FEATURE IMPORTANCE
# =========================
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": rf_clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 vigtigste features:")
print(importance_df.head(10).to_string(index=False))


# =========================
# GEM RESULTATER
# =========================
predictions_path = PROJECT_DIR / "random_forest_predictions.csv"
confusion_matrix_path = PROJECT_DIR / "random_forest_confusion_matrix.csv"
importance_path = PROJECT_DIR / "random_forest_feature_importance.csv"

results_df = pd.DataFrame({
    "source_file": test_df["source_file"].values,
    "segment_id": test_df["segment_id"].values,
    "true_label": y_test.values,
    "predicted_label": y_pred
})

results_df.to_csv(predictions_path, index=False)
cm_df.to_csv(confusion_matrix_path)
importance_df.to_csv(importance_path, index=False)

print("\nResultater gemt:")
print(predictions_path)
print(confusion_matrix_path)
print(importance_path)
