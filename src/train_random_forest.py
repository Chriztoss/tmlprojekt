from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# FIND FEATURE-DATASET
PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "feature_dataset.csv"

df = pd.read_csv(CSV_PATH)


# DATASET INFO
print("\n===== DATASET =====")
print("Antal samples:", len(df))

print("\nKlasser:")
print(df["label"].value_counts())


# X = FEATURES, y = LABELS
X = df.drop(
    columns=["label", "filename", "source_file", "segment_id"],
    errors="ignore"
)

y = df["label"]

# GROUP SPLIT
# Holder segmenter fra samme WAV-fil samlet i enten train eller test
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


# RANDOM FOREST MODEL
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_clf.fit(X_train, y_train)

# TEST MODELLEN
y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===== RANDOM FOREST RESULTAT =====")
print("n_estimators = 100")
print(f"Accuracy = {accuracy:.3f}")

# CONFUSION MATRIX
labels = sorted(y.unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in labels],
    columns=[f"pred_{label}" for label in labels]
)

print("\nConfusion matrix:")
print(cm_df)


# CLASSIFICATION REPORT
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 vigtigste features:")
print(importance_df.head(10))


# GEM RESULTATER
predictions_path = PROJECT_DIR / "random_forest_predictions.csv"
importance_path = PROJECT_DIR / "random_forest_feature_importance.csv"

results_df = pd.DataFrame({
    "source_file": df.iloc[test_idx]["source_file"].values,
    "segment_id": df.iloc[test_idx]["segment_id"].values,
    "true_label": y_test.values,
    "predicted_label": y_pred
})

results_df.to_csv(predictions_path, index=False)
importance_df.to_csv(importance_path, index=False)

print("\nResultater gemt:")
print(predictions_path)
print(importance_path)