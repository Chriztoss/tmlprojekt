from pathlib import Path
from contextlib import contextmanager
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import emlearn


# =========================
# SETTINGS
# =========================

def find_project_dir():
    here = Path(__file__).resolve().parent

    if (here / "c_feature2_dataset.csv").exists():
        return here

    if (here.parent / "c_feature2_dataset.csv").exists():
        return here.parent

    return here.parent


PROJECT_DIR = find_project_dir()
CSV_PATH = PROJECT_DIR / "c_feature2_dataset.csv"

OUT_DIR = PROJECT_DIR / "src"
OUT_DIR.mkdir(exist_ok=True)

MODEL_OUT = OUT_DIR / "chord_rf_model.h"
INFO_OUT = OUT_DIR / "chord_model_info.h"

PREDICTIONS_OUT = PROJECT_DIR / "random_forest_export_predictions.csv"
PYTHON_CM_OUT = PROJECT_DIR / "random_forest_export_python_confusion_matrix.csv"
EMLEARN_CM_OUT = PROJECT_DIR / "random_forest_export_emlearn_confusion_matrix.csv"
PYTHON_VS_EMLEARN_CM_OUT = PROJECT_DIR / "random_forest_python_vs_emlearn_confusion_matrix.csv"
SUMMARY_OUT = PROJECT_DIR / "random_forest_export_summary.txt"
IMPORTANCE_OUT = PROJECT_DIR / "random_forest_export_feature_importance.csv"

TEST_FILES_PER_CLASS = 2
BALANCE_TOTAL_FILES = True
REQUIRED_MATCH = 1.0

N_ESTIMATORS = 30
MAX_DEPTH = 8
RANDOM_STATE = 42
LEAF_BITS_TO_TEST = [0, 3, 4, 5, 6, 7, 8]

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

FEATURE_NAMES = (
    ["spectral_centroid"] +
    [f"mfcc_{i}" for i in range(1, 6)] +
    [f"chroma_stft_{note}" for note in NOTE_NAMES]
)


# =========================
# HELPERS
# =========================

@contextmanager
def suppress_console_output():
    """Skjuler støj fra emlearn/subprocess under conversion og test."""
    sys.stdout.flush()
    sys.stderr.flush()

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def select_live_features(df):
    missing = [name for name in FEATURE_NAMES if name not in df.columns]

    if missing:
        raise ValueError(f"CSV mangler disse features: {missing}")

    return FEATURE_NAMES


def make_balanced_file_split(df):
    if "source_file" not in df.columns:
        raise ValueError("CSV mangler 'source_file'. Kan ikke lave korrekt filbaseret split.")

    file_df = df[["source_file", "label"]].drop_duplicates()
    files_by_label = {}

    for label in sorted(file_df["label"].unique()):
        label_files = sorted(
            file_df[file_df["label"] == label]["source_file"].tolist()
        )
        files_by_label[label] = label_files

    min_files_per_class = min(len(files) for files in files_by_label.values())
    files_per_class = min_files_per_class if BALANCE_TOTAL_FILES else None

    train_files = []
    test_files = []

    for label in sorted(files_by_label):
        files = files_by_label[label]

        if BALANCE_TOTAL_FILES:
            files = files[:files_per_class]

        if len(files) <= TEST_FILES_PER_CLASS:
            raise ValueError(
                f"Klassen {label} har kun {len(files)} filer. "
                f"TEST_FILES_PER_CLASS={TEST_FILES_PER_CLASS} er for højt."
            )

        test_part = files[-TEST_FILES_PER_CLASS:]
        train_part = files[:-TEST_FILES_PER_CLASS]

        train_files.extend(train_part)
        test_files.extend(test_part)

    train_df = df[df["source_file"].isin(train_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()

    return train_df, test_df


def save_model_info(labels, feature_names):
    lines = []
    lines.append("#ifndef CHORD_MODEL_INFO_H")
    lines.append("#define CHORD_MODEL_INFO_H\n")
    lines.append(f"#define CHORD_NUM_CLASSES {len(labels)}")
    lines.append(f"#define CHORD_NUM_FEATURES {len(feature_names)}\n")

    lines.append("static const char *CHORD_LABELS[CHORD_NUM_CLASSES] = {")
    for label in labels:
        lines.append(f'    "{label}",')
    lines.append("};\n")

    lines.append("static const char *CHORD_FEATURE_NAMES[CHORD_NUM_FEATURES] = {")
    for name in feature_names:
        lines.append(f'    "{name}",')
    lines.append("};\n")

    lines.append("#endif\n")

    INFO_OUT.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix(path, y_true, y_pred, labels, prefix_true="true", prefix_pred="pred"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    cm_df = pd.DataFrame(
        cm,
        index=[f"{prefix_true}_{label}" for label in labels],
        columns=[f"{prefix_pred}_{label}" for label in labels]
    )

    cm_df.to_csv(path)
    return cm_df


def save_python_vs_emlearn_confusion(path, python_pred, emlearn_pred, labels):
    cm = confusion_matrix(python_pred, emlearn_pred, labels=np.arange(len(labels)))

    cm_df = pd.DataFrame(
        cm,
        index=[f"python_{label}" for label in labels],
        columns=[f"emlearn_{label}" for label in labels]
    )

    cm_df.to_csv(path)
    return cm_df


def main():
    print("\n===== RANDOM FOREST EXPORT + EMLEARN-VERIFIKATION =====")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Kunne ikke finde {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    feature_names = select_live_features(df)

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])
    labels = list(encoder.classes_)

    train_df, test_df = make_balanced_file_split(df)

    x_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["label_encoded"].values

    x_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["label_encoded"].values

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )

    rf.fit(x_train, y_train)

    python_pred = rf.predict(x_test).astype(int)
    python_acc = accuracy_score(y_test, python_pred)

    best = None
    conversion_errors = []

    for leaf_bits in LEAF_BITS_TO_TEST:
        try:
            with suppress_console_output():
                cmodel = emlearn.convert(
                    rf,
                    method="inline",
                    dtype="float",
                    leaf_bits=leaf_bits
                )
                emlearn_pred = np.asarray(cmodel.predict(x_test), dtype=int)

            match = float(np.mean(emlearn_pred == python_pred))
            emlearn_acc = accuracy_score(y_test, emlearn_pred)

            candidate = {
                "leaf_bits": leaf_bits,
                "cmodel": cmodel,
                "emlearn_pred": emlearn_pred,
                "match": match,
                "emlearn_acc": emlearn_acc,
            }

            if best is None or match > best["match"]:
                best = candidate

            if match >= REQUIRED_MATCH:
                best = candidate
                break

        except Exception as e:
            conversion_errors.append(f"leaf_bits={leaf_bits}: {e}")

    if best is None:
        error_text = "\n".join(conversion_errors)
        raise RuntimeError("emlearn-konvertering fejlede for alle leaf_bits.\n" + error_text)

    best_match = best["match"]
    best_leaf_bits = best["leaf_bits"]
    emlearn_pred = best["emlearn_pred"]
    emlearn_acc = best["emlearn_acc"]

    status_ok = best_match >= REQUIRED_MATCH

    if status_ok:
        save_model_info(labels, feature_names)
        best["cmodel"].save(name="chord_rf", file=str(MODEL_OUT))

    predictions_df = pd.DataFrame({
        "source_file": test_df["source_file"].values,
        "segment_id": test_df["segment_id"].values,
        "true_label": encoder.inverse_transform(y_test),
        "python_prediction": encoder.inverse_transform(python_pred),
        "emlearn_prediction": encoder.inverse_transform(emlearn_pred),
        "python_equals_emlearn": python_pred == emlearn_pred,
    })
    predictions_df.to_csv(PREDICTIONS_OUT, index=False)

    save_confusion_matrix(PYTHON_CM_OUT, y_test, python_pred, labels)
    save_confusion_matrix(EMLEARN_CM_OUT, y_test, emlearn_pred, labels)
    save_python_vs_emlearn_confusion(PYTHON_VS_EMLEARN_CM_OUT, python_pred, emlearn_pred, labels)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(IMPORTANCE_OUT, index=False)

    summary_lines = [
        "RANDOM FOREST EXPORT + EMLEARN-VERIFIKATION",
        f"Features: {len(feature_names)}",
        f"Split: {len(x_train)} train-samples / {len(x_test)} test-samples",
        f"Train WAV-filer: {train_df['source_file'].nunique()}",
        f"Test WAV-filer: {test_df['source_file'].nunique()}",
        f"Test-WAV pr. klasse: {TEST_FILES_PER_CLASS}",
        f"n_estimators: {N_ESTIMATORS}",
        f"max_depth: {MAX_DEPTH}",
        f"Bedste leaf_bits: {best_leaf_bits}",
        f"Python test accuracy: {python_acc * 100:.2f} %",
        f"emlearn/C-model test accuracy: {emlearn_acc * 100:.2f} %",
        f"emlearn/C-model vs Python match: {best_match * 100:.2f} %",
        "Status: " + ("OK - C-modellen matcher Python-modellen 100 %" if status_ok else "IKKE OK - C-modellen matcher ikke Python-modellen 100 %"),
    ]

    SUMMARY_OUT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Features: {len(feature_names)}")
    print(f"Split: {len(x_train)} train-samples / {len(x_test)} test-samples")
    print(f"Test-WAV pr. klasse: {TEST_FILES_PER_CLASS}")
    print(f"n_estimators: {N_ESTIMATORS}")
    print(f"max_depth: {MAX_DEPTH}")
    print(f"Bedste leaf_bits: {best_leaf_bits}")
    print(f"Python test accuracy: {python_acc * 100:.2f} %")
    print(f"c test accuracy: {emlearn_acc * 100:.2f} %")
    print(f"c vs Python match: {best_match * 100:.2f} %")

    if status_ok:
        print("Status: OK - C-modellen matcher Python-modellen 100 %")
        print("Gemte filer:")
        print(f" - {MODEL_OUT}")
        print(f" - {INFO_OUT}")
        print(f" - {PREDICTIONS_OUT}")
        print(f" - {PYTHON_CM_OUT}")
        print(f" - {EMLEARN_CM_OUT}")
        print(f" - {PYTHON_VS_EMLEARN_CM_OUT}")
        print(f" - {SUMMARY_OUT}")
    else:
        print("Status: IKKE OK - C-modellen matcher ikke Python-modellen 100 %")
        print("chord_rf_model.h blev ikke overskrevet, fordi exporten ikke er verificeret.")
        print(f"Se detaljer i: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
