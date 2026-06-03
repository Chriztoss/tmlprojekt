from pathlib import Path
import os
import sys
import subprocess
import contextlib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import emlearn


# =========================
# PATHS / SETTINGS
# =========================
PROJECT_DIR = Path(__file__).resolve().parent
if PROJECT_DIR.name == "src":
    PROJECT_DIR = PROJECT_DIR.parent

os.chdir(PROJECT_DIR)

CSV_PATH = PROJECT_DIR / "c_feature2_dataset.csv"
OUT_DIR = PROJECT_DIR / "src"
OUT_DIR.mkdir(exist_ok=True)

MODEL_OUT = OUT_DIR / "chord_knn_model.h"
INFO_OUT = OUT_DIR / "chord_knn_model_info.h"
VERIFY_C_OUT = OUT_DIR / "knn_export_verify.c"
VERIFY_EXE_OUT = PROJECT_DIR / "knn_export_verify.exe"

SUMMARY_OUT = PROJECT_DIR / "knn_export_summary.txt"
PY_PRED_OUT = PROJECT_DIR / "knn_export_predictions.csv"
C_PRED_OUT = PROJECT_DIR / "knn_export_c_predictions.csv"
PY_CM_OUT = PROJECT_DIR / "knn_export_python_confusion_matrix.csv"
C_CM_OUT = PROJECT_DIR / "knn_export_c_confusion_matrix.csv"
K_OUT = PROJECT_DIR / "knn_live_18_best_k.txt"

TEST_FILES_PER_CLASS = 2
BALANCE_TOTAL_FILES = True
QUANT_SCALE = 100.0
K_VALUES = [1, 3, 5, 7]

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
# SMALL HELPERS
# =========================
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress noisy output from emlearn internals and compiler checks."""
    sys.stdout.flush()
    sys.stderr.flush()

    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def c_float_literal(value):
    s = f"{float(value):.9g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s + "f"


def c_float_array(name, values):
    lines = [f"static const float {name}[CHORD_KNN_NUM_FEATURES] = {{"]
    for value in values:
        lines.append(f"    {c_float_literal(value)},")
    lines.append("};\n")
    return "\n".join(lines)


def c_string_array(name, size_macro, values):
    lines = [f"static const char *{name}[{size_macro}] = {{"]
    for value in values:
        escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'    "{escaped}",')
    lines.append("};\n")
    return "\n".join(lines)


def save_confusion_csv(path, y_true, y_pred, labels, label_names):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in label_names],
        columns=[f"pred_{name}" for name in label_names]
    )
    cm_df.to_csv(path)


def try_save_confusion_png(path, y_true, y_pred, labels, label_names, title):
    """Optional PNG output. Skips silently if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(np.sum(cm))
    acc = accuracy_score(y_true, y_pred) * 100.0

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{title}\n{total} tests, samlet accuracy: {acc:.1f} %", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predikteret akkord", fontsize=12)
    ax.set_ylabel("Faktisk akkord", fontsize=12)

    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Antal tests")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


# =========================
# DATA / SPLIT
# =========================
def select_live_features(df):
    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    if missing:
        raise ValueError(f"CSV mangler disse features: {missing}")
    return FEATURE_NAMES


def make_balanced_file_split(df):
    if "source_file" not in df.columns:
        raise ValueError("CSV mangler source_file. Kan ikke lave filbaseret split.")

    file_df = df[["source_file", "label"]].drop_duplicates()
    files_by_label = {}

    for label in sorted(file_df["label"].unique()):
        files_by_label[label] = sorted(
            file_df[file_df["label"] == label]["source_file"].tolist()
        )

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

        test_files.extend(files[-TEST_FILES_PER_CLASS:])
        train_files.extend(files[:-TEST_FILES_PER_CLASS])

    train_df = df[df["source_file"].isin(train_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()
    return train_df, test_df


# =========================
# MODEL HELPERS
# =========================
def quantize_features(x_scaled):
    xq = np.round(x_scaled * QUANT_SCALE)
    xq = np.clip(xq, -32768, 32767)
    return xq.astype(np.int16)


def choose_best_k(x_train, y_train, x_test, y_test):
    best_k = None
    best_acc = -1.0
    best_pred = None

    for k in K_VALUES:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            metric="euclidean"
        )
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        acc = accuracy_score(y_test, pred)

        # Keeps the lowest k in a tie, because K_VALUES is sorted ascending.
        if acc > best_acc:
            best_k = k
            best_acc = acc
            best_pred = pred

    return best_k, best_acc, best_pred


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

{c_string_array("CHORD_KNN_LABELS", "CHORD_KNN_NUM_CLASSES", labels)}
{c_string_array("CHORD_KNN_FEATURE_NAMES", "CHORD_KNN_NUM_FEATURES", feature_names)}
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


def save_verify_c(x_test_raw, y_test, python_pred, train_samples):
    # emlearn requires distances_length > model->n_items.
    distance_items = train_samples + 1

    lines = []
    lines.append("#include <stdio.h>")
    lines.append("#include <stdint.h>")
    lines.append('#include "chord_knn_model_info.h"')
    lines.append('#include "chord_knn_model.h"')
    lines.append("")
    lines.append(f"#define TEST_N {len(x_test_raw)}")
    lines.append(f"#define DIST_N {distance_items}")
    lines.append("")

    lines.append("static const float TEST_X_RAW[TEST_N][CHORD_KNN_NUM_FEATURES] = {")
    for row in x_test_raw:
        values = ", ".join(c_float_literal(v) for v in row)
        lines.append(f"    {{ {values} }},")
    lines.append("};")
    lines.append("")

    lines.append("static const int16_t TEST_TRUE[TEST_N] = {")
    lines.append("    " + ", ".join(str(int(v)) for v in y_test))
    lines.append("};")
    lines.append("")

    lines.append("static const int16_t TEST_PYTHON[TEST_N] = {")
    lines.append("    " + ", ".join(str(int(v)) for v in python_pred))
    lines.append("};")
    lines.append("")

    lines.append("int main(void)")
    lines.append("{")
    lines.append("    int correct_vs_true = 0;")
    lines.append("    int match_vs_python = 0;")
    lines.append("    int mismatches = 0;")
    lines.append("    EmlNeighborsDistanceItem distances[DIST_N];")
    lines.append('    FILE *fp = fopen("knn_export_c_predictions.csv", "w");')
    lines.append("    if (fp != NULL) {")
    lines.append('        fprintf(fp, "sample,true_label,python_pred,c_pred\\n");')
    lines.append("    }")
    lines.append("")
    lines.append("    for (int i = 0; i < TEST_N; i++)")
    lines.append("    {")
    lines.append("        int16_t x_q[CHORD_KNN_NUM_FEATURES];")
    lines.append("        int16_t pred = -1;")
    lines.append("        chord_knn_prepare_features(TEST_X_RAW[i], x_q);")
    lines.append("")
    lines.append("        EmlError err = eml_neighbors_predict(&chord_knn, x_q, CHORD_KNN_NUM_FEATURES, distances, DIST_N, &pred);")
    lines.append("        if (err != EmlOk)")
    lines.append("        {")
    lines.append('            printf("error=%d\\n", (int)err);')
    lines.append("            if (fp != NULL) fclose(fp);")
    lines.append("            return 10;")
    lines.append("        }")
    lines.append("")
    lines.append("        if (pred == TEST_TRUE[i]) correct_vs_true++;")
    lines.append("        if (pred == TEST_PYTHON[i]) match_vs_python++;")
    lines.append("        else mismatches++;")
    lines.append("")
    lines.append("        if (fp != NULL)")
    lines.append("        {")
    lines.append('            fprintf(fp, "%d,%d,%d,%d\\n", i, TEST_TRUE[i], TEST_PYTHON[i], pred);')
    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append("    if (fp != NULL) fclose(fp);")
    lines.append('    printf("samples=%d\\n", TEST_N);')
    lines.append('    printf("c_correct_vs_true=%d\\n", correct_vs_true);')
    lines.append('    printf("c_match_vs_python=%d\\n", match_vs_python);')
    lines.append('    printf("c_mismatches_vs_python=%d\\n", mismatches);')
    lines.append("    return mismatches == 0 ? 0 : 2;")
    lines.append("}")
    lines.append("")

    VERIFY_C_OUT.write_text("\n".join(lines), encoding="utf-8")


def compile_and_run_verify():
    emlearn_dir = Path(emlearn.__file__).resolve().parent

    compile_cmd = [
        "gcc",
        "-std=c99",
        "-I", str(OUT_DIR),
        "-I", str(emlearn_dir),
        str(VERIFY_C_OUT),
        "-o", str(VERIFY_EXE_OUT),
        "-lm",
    ]

    compile_result = subprocess.run(
        compile_cmd,
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )

    if compile_result.returncode != 0:
        log_path = PROJECT_DIR / "knn_export_c_compile_error.txt"
        log_path.write_text(
            compile_result.stdout + "\n" + compile_result.stderr,
            encoding="utf-8"
        )
        return None, f"C-verifikation kunne ikke compiles. Se {log_path.name}"

    run_result = subprocess.run(
        [str(VERIFY_EXE_OUT)],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )

    if run_result.returncode not in (0, 2):
        log_path = PROJECT_DIR / "knn_export_c_run_error.txt"
        log_path.write_text(run_result.stdout + "\n" + run_result.stderr, encoding="utf-8")
        return None, f"C-verifikation kunne ikke køres. Se {log_path.name}"

    parsed = {}
    for line in run_result.stdout.splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            try:
                parsed[key] = int(value)
            except ValueError:
                parsed[key] = value

    return parsed, None


def main():
    print("\n===== KNN EXPORT + C-VERIFIKATION =====")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Kunne ikke finde {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    feature_names = select_live_features(df)

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])
    label_names = list(encoder.classes_)
    numeric_labels = list(range(len(label_names)))

    train_df, test_df = make_balanced_file_split(df)

    x_train_raw = train_df[feature_names].values.astype(np.float32)
    x_test_raw = test_df[feature_names].values.astype(np.float32)
    y_train = train_df["label_encoded"].values.astype(np.int16)
    y_test = test_df["label_encoded"].values.astype(np.int16)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    x_train = quantize_features(x_train_scaled)
    x_test = quantize_features(x_test_scaled)

    best_k, py_acc, py_pred = choose_best_k(x_train, y_train, x_test, y_test)
    K_OUT.write_text(str(best_k), encoding="utf-8")

    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        weights="uniform",
        metric="euclidean"
    )
    knn.fit(x_train, y_train)
    py_pred = knn.predict(x_test).astype(np.int16)
    py_acc = accuracy_score(y_test, py_pred)

    save_knn_info(label_names, feature_names, scaler.mean_, scaler.scale_)

    with suppress_stdout_stderr():
        cmodel = emlearn.convert(knn, method=None)
        cmodel.save(name="chord_knn", file=str(MODEL_OUT))

    patch_knn_model_header(len(label_names))
    save_verify_c(x_test_raw, y_test, py_pred, train_samples=len(x_train))

    py_pred_df = pd.DataFrame({
        "source_file": test_df["source_file"].values,
        "segment_id": test_df["segment_id"].values,
        "true_label": encoder.inverse_transform(y_test),
        "predicted_label_python": encoder.inverse_transform(py_pred),
    })
    py_pred_df.to_csv(PY_PRED_OUT, index=False)

    save_confusion_csv(PY_CM_OUT, y_test, py_pred, numeric_labels, label_names)
    try_save_confusion_png(
        PROJECT_DIR / "knn_export_python_confusion_matrix.png",
        y_test, py_pred, numeric_labels, label_names,
        "Confusion matrix - KNN Python export-test"
    )

    c_info, c_error = compile_and_run_verify()

    c_acc = None
    c_match = None
    status = "FEJL"

    if c_info is not None:
        samples = c_info.get("samples", len(y_test))
        c_correct = c_info.get("c_correct_vs_true", 0)
        c_match_count = c_info.get("c_match_vs_python", 0)
        mismatches = c_info.get("c_mismatches_vs_python", -1)

        c_acc = c_correct / samples
        c_match = c_match_count / samples

        if C_PRED_OUT.exists():
            c_pred_df = pd.read_csv(C_PRED_OUT)
            c_true = c_pred_df["true_label"].values.astype(int)
            c_pred = c_pred_df["c_pred"].values.astype(int)
            save_confusion_csv(C_CM_OUT, c_true, c_pred, numeric_labels, label_names)
            try_save_confusion_png(
                PROJECT_DIR / "knn_export_c_confusion_matrix.png",
                c_true, c_pred, numeric_labels, label_names,
                "Confusion matrix - KNN C export-test"
            )

        if mismatches == 0:
            status = "OK - C-modellen matcher Python-modellen 100 %"
        else:
            status = f"FEJL - C-modellen afviger fra Python på {mismatches} samples"
    else:
        status = c_error

    summary_lines = [
        "===== KNN EXPORT + C-VERIFIKATION =====",
        f"Features: {len(feature_names)}",
        f"Split: {len(x_train)} train-samples / {len(x_test)} test-samples",
        f"Test-WAV pr. klasse: {TEST_FILES_PER_CLASS}",
        f"Valgt k: {best_k}",
        f"Python test accuracy: {py_acc * 100.0:.2f} %",
    ]

    if c_acc is not None and c_match is not None:
        summary_lines.append(f"C test accuracy: {c_acc * 100.0:.2f} %")
        summary_lines.append(f"C vs Python match: {c_match * 100.0:.2f} %")
    else:
        summary_lines.append("C test accuracy: IKKE VERIFICERET")
        summary_lines.append("C vs Python match: IKKE VERIFICERET")

    summary_lines.append(f"Status: {status}")
    SUMMARY_OUT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Features:", len(feature_names))
    print(f"Split: {len(x_train)} train-samples / {len(x_test)} test-samples")
    print("Test-WAV pr. klasse:", TEST_FILES_PER_CLASS)
    print("Valgt k:", best_k)
    print(f"Python test accuracy: {py_acc * 100.0:.2f} %")

    if c_acc is not None and c_match is not None:
        print(f"C test accuracy: {c_acc * 100.0:.2f} %")
        print(f"C vs Python match: {c_match * 100.0:.2f} %")
    else:
        print("C test accuracy: IKKE VERIFICERET")
        print("C vs Python match: IKKE VERIFICERET")

    print("Status:", status)
    print("\nGemte filer:")
    print(" -", MODEL_OUT)
    print(" -", INFO_OUT)
    print(" -", SUMMARY_OUT)
    print(" -", PY_CM_OUT)
    if C_CM_OUT.exists():
        print(" -", C_CM_OUT)


if __name__ == "__main__":
    main()
