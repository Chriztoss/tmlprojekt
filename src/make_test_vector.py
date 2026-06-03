from pathlib import Path
import pandas as pd

# Scriptet ligger i src, så projektmappen er én mappe tilbage
PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_DIR / "c_feature2_dataset.csv"
OUT_PATH = PROJECT_DIR / "src" / "test_vector.h"

META_COLS = ["label", "filename", "source_file", "segment_id"]

df = pd.read_csv(CSV_PATH)

feature_names = [c for c in df.columns if c not in META_COLS]

row = df.iloc[0]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("#ifndef TEST_VECTOR_H\n")
    f.write("#define TEST_VECTOR_H\n\n")

    f.write('#include "chord_model_info.h"\n\n')

    f.write(f'static const char *TEST_LABEL = "{row["label"]}";\n\n')

    f.write("static float TEST_FEATURES[CHORD_NUM_FEATURES] = {\n")

    for name in feature_names:
        value = float(row[name])
        f.write(f"    {value:.9g}f, // {name}\n")

    f.write("};\n\n")

    f.write("#endif\n")

print("Saved:", OUT_PATH)
print("Test label:", row["label"])