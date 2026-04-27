from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# FIND FEATURE-DATASET
# =========================
PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "feature_dataset.csv"

df = pd.read_csv(CSV_PATH)


# =========================
# VIS DATASET INFO
# =========================
print("\n===== DATASET =====")
print("Antal samples:", len(df))

print("\nKlasser:")
print(df["label"].value_counts())


# =========================
# X = FEATURES, y = LABELS
# =========================
X = df.drop(
    columns=["label", "filename", "source_file", "segment_id"],
    errors="ignore"
)

y = df["label"]


# =========================
# GROUP SPLIT
# =========================
# Sikrer at segmenter fra samme WAV-fil ikke ender i både train og test
groups = df["source_file"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))

# =========================
# K-NN MODEL
# =========================
k = 4
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=k)
)

knn.fit(X_train, y_train)


# =========================
# TEST MODELLEN
# =========================
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===== K-NN RESULTAT =====")
print(f"k = {k}")
print(f"Accuracy = {accuracy:.3f}")


# =========================
# CONFUSION MATRIX
# =========================
labels = sorted(y.unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in labels],
    columns=[f"pred_{label}" for label in labels]
)

print("\nConfusion matrix:")
print(cm_df)


# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# =========================
# GEM RESULTATER
# =========================
results_path = PROJECT_DIR / "knn_test_predictions_k3.csv"

results_df = pd.DataFrame({
    "source_file": df.iloc[test_idx]["source_file"].values,
    "segment_id": df.iloc[test_idx]["segment_id"].values,
    "true_label": y_test.values,
    "predicted_label": y_pred
})

results_df.to_csv(results_path, index=False)

print("\nResultater gemt:")
print(results_path)