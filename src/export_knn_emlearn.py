from pathlib import Path
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import emlearn


PROJECT_DIR = Path(__file__).resolve().parent
if PROJECT_DIR.name == "src":
    PROJECT_DIR = PROJECT_DIR.parent

os.chdir(PROJECT_DIR)

CSV_PATH = PROJECT_DIR / "c_feature2_dataset.csv"

OUT_DIR = PROJECT_DIR / "src"
OUT_DIR.mkdir(exist_ok=True)

MODEL_OUT = OUT_DIR / "chord_knn_model.h"
INFO_OUT = OUT_DIR / "chord_knn_model_info.h"

BEST_K_PATH = PROJECT_DIR / "knn_live_18_best_k.txt"
PREDICTIONS_OUT = PROJECT_DIR / "knn_live_18_export_predictions.csv"

META_COLS = ["label", "filename", "source_file", "segment_id", "label_encoded"]
QUANT_SCALE = 100.0


def select_live_features(df):
    return [
        col for col in df.columns
        if col not in META_COLS
        and not col.startswith("chroma_cens")
        and not col.startswith("tonnetz")
    ]


def split_train_test(df):
    if "source_file" not in df.columns:
        return train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label"]
        )

    file_df = df[["source_file", "label"]].drop_duplicates()

    train_files, test_files = train_test_split(
        file_df,
        test_size=0.2,
        random_state=42,
        stratify=file_df["label"]
    )

    train_df = df[df["source_file"].isin(train_files["source_file"])].copy()
    test_df = df[df["source_file"].isin(test_files["source_file"])].copy()

    return train_df, test_df


def read_best_k():
    if BEST_K_PATH.exists():
        return int(BEST_K_PATH.read_text(encoding="utf-8").strip())

    print("WARNING: knn_live_18_best_k.txt not found. Using k=1.")
    return 1


def quantize_features(x_scaled):
    xq = np.round(x_scaled * QUANT_SCALE)
    xq = np.clip(xq, -32768, 32767)
    return xq.astype(np.int16)


def c_label_array(labels):
    lines = ["static const char * const CHORD_KNN_LABELS[CHORD_KNN_NUM_CLASSES] = {"]
    for label in labels:
        lines.append(f'    "{label}",')
    lines.append("};\n")
    return "\n".join(lines)


def c_float_array(name, values):
    lines = [f"static const float {name}[CHORD_KNN_NUM_FEATURES] = {{"]
    for value in values:
        lines.append(f"    {float(value):.9g}f,")
    lines.append("};\n")
    return "\n".join(lines)


def save_knn_info(labels, feature_names, mean, scale):
    content = f"""#ifndef CHORD_KNN_MODEL_INFO_H
#define CHORD_KNN_MODEL_INFO_H

#include <stdint.h>

#ifndef EML_NEIGHBORS_MAX_CLASSES
#define EML_NEIGHBORS_MAX_CLASSES {len(labels)}
#endif

#define CHORD_KNN_NUM_CLASSES {len(labels)}
#define CHORD_KNN_NUM_FEATURES {len(feature_names)}
#define CHORD_KNN_QUANT_SCALE {QUANT_SCALE:.1f}f

{c_label_array(labels)}
{c_float_array("CHORD_KNN_MEAN", mean)}
{c_float_array("CHORD_KNN_SCALE", scale)}
static void chord_knn_prepare_features(const float *raw, int16_t *out)
{{
    for (int i = 0; i < CHORD_KNN_NUM_FEATURES; i++)
    {{
        float z = (raw[i] - CHORD_KNN_MEAN[i]) / CHORD_KNN_SCALE[i];
        float qf = z * CHORD_KNN_QUANT_SCALE;

        if (qf > 32767.0f) qf = 32767.0f;
        if (qf < -32768.0f) qf = -32768.0f;

        if (qf >= 0.0f)
            out[i] = (int16_t)(qf + 0.5f);
        else
            out[i] = (int16_t)(qf - 0.5f);
    }}
}}

#endif
"""

    INFO_OUT.write_text(content, encoding="utf-8")
    print("Saved:", INFO_OUT)


def patch_knn_model_header(num_classes):
    text = MODEL_OUT.read_text(encoding="utf-8")

    macro = (
        "#ifndef EML_NEIGHBORS_MAX_CLASSES\n"
        f"#define EML_NEIGHBORS_MAX_CLASSES {num_classes}\n"
        "#endif\n\n"
    )

    if "#include <eml_neighbors.h>" in text:
        text = text.replace("#include <eml_neighbors.h>", macro + "#include <eml_neighbors.h>", 1)
    elif '#include "eml_neighbors.h"' in text:
        text = text.replace('#include "eml_neighbors.h"', macro + '#include "eml_neighbors.h"', 1)
    else:
        text = macro + text

    MODEL_OUT.write_text(text, encoding="utf-8")

    print("Patched:", MODEL_OUT)
    print("Set EML_NEIGHBORS_MAX_CLASSES to", num_classes)


def print_distribution(name, pred):
    values, counts = np.unique(pred, return_counts=True)

    print(f"\n{name} prediction distribution:")
    for value, count in zip(values, counts):
        print(f"  class {value}: {count}")


def main():
    print("\n===== EXPORT SCRIPT VERSION =====")
    print("KNN_INT16_RAM_OPTIMIZED_INFO_HEADER")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    feature_names = select_live_features(df)

    print("\n===== KNN LIVE 18 FEATURES - EMLEARN EXPORT =====")
    print("CSV:", CSV_PATH)
    print("Number of samples:", len(df))
    print("Number of features:", len(feature_names))

    print("\nSelected features:")
    for name in feature_names:
        print(" -", name)

    if len(feature_names) != 18:
        raise ValueError(f"Expected 18 features, got {len(feature_names)}")

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])
    labels = list(encoder.classes_)

    train_df, test_df = split_train_test(df)

    x_train_raw = train_df[feature_names].values.astype(np.float32)
    x_test_raw = test_df[feature_names].values.astype(np.float32)

    y_train = train_df["label_encoded"].values
    y_test = test_df["label_encoded"].values

    print("\n===== TRAIN / TEST SPLIT =====")
    print("Training samples:", len(x_train_raw))
    print("Test samples:", len(x_test_raw))

    if "source_file" in df.columns:
        print("Training files:", train_df["source_file"].nunique())
        print("Test files:", test_df["source_file"].nunique())

    print("Classes:", labels)

    k = read_best_k()
    print("\nSelected k for export:", k)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    x_train = quantize_features(x_train_scaled)
    x_test = quantize_features(x_test_scaled)

    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform",
        metric="euclidean"
    )

    knn.fit(x_train, y_train)

    pred = knn.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print(f"\nPython KNN test accuracy before export: {acc * 100:.2f}%")
    print_distribution("Python KNN", pred)

    results = pd.DataFrame({
        "source_file": test_df["source_file"].values if "source_file" in test_df.columns else "",
        "segment_id": test_df["segment_id"].values if "segment_id" in test_df.columns else "",
        "true_label": encoder.inverse_transform(y_test),
        "predicted_label_python": encoder.inverse_transform(pred),
    })

    results.to_csv(PREDICTIONS_OUT, index=False)
    print("Saved:", PREDICTIONS_OUT)

    save_knn_info(labels, feature_names, scaler.mean_, scaler.scale_)

    print("\nConverting KNN model with emlearn...")

    cmodel = emlearn.convert(knn, method=None)

    cmodel.save(
        name="chord_knn",
        file=str(MODEL_OUT)
    )

    patch_knn_model_header(len(labels))

    print("\nSaved:")
    print(" -", MODEL_OUT)
    print(" -", INFO_OUT)
    print(" -", PREDICTIONS_OUT)

    print("\nImportant:")
    print("This version removes CHORD_KNN_FEATURE_NAMES to save SRAM.")
    print("Use chord_knn_model_info.h before chord_knn_model.h in main.cpp.")
    print("Delete src/tmp before Particle compile if it exists.")

    print("\nDone.")


if __name__ == "__main__":
    main()