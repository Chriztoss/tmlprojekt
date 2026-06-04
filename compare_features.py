from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

if ROOT.name == "src":
    ROOT = ROOT.parent

PYTHON_CSV = ROOT / "feature_dataset_2.csv"
C_CSV = ROOT / "c_feature2_dataset.csv"

KEYS = ["source_file", "segment_id"]
META = ["label", "filename", "source_file", "segment_id", "label_encoded"]

py_df = pd.read_csv(PYTHON_CSV)
c_df = pd.read_csv(C_CSV)

features = [
    col for col in py_df.columns
    if col in c_df.columns and col not in META
]

merged = py_df[KEYS + features].merge(
    c_df[KEYS + features],
    on=KEYS,
    suffixes=("_python", "_c")
)

print("===== PYTHON VS C FEATURE CHECK =====")
print("Python rows:", len(py_df))
print("C rows:", len(c_df))
print("Matched rows:", len(merged))
print("Features compared:", len(features))
print()

# Direkte sammenligning: Python-værdi vs C-værdi
value_rows = []

for _, row in merged.iterrows():
    for feature in features:
        py_val = float(row[f"{feature}_python"])
        c_val = float(row[f"{feature}_c"])
        diff = c_val - py_val

        value_rows.append({
            "source_file": row["source_file"],
            "segment_id": row["segment_id"],
            "feature": feature,
            "python_value": py_val,
            "c_value": c_val,
            "difference": diff,
            "abs_difference": abs(diff),
        })

values_df = pd.DataFrame(value_rows)

# Summary pr. feature inkl. korrelation
summary_rows = []

for feature in features:
    py = merged[f"{feature}_python"].astype(float)
    c = merged[f"{feature}_c"].astype(float)
    diff = c - py

    correlation = py.corr(c)

    summary_rows.append({
        "feature": feature,
        "mean_python": py.mean(),
        "mean_c": c.mean(),
        "mean_difference": diff.mean(),
        "mean_abs_difference": diff.abs().mean(),
        "max_abs_difference": diff.abs().max(),
        "correlation": correlation,
    })

summary_df = pd.DataFrame(summary_rows)

values_out = ROOT / "feature_compare_python_c_values.csv"
summary_out = ROOT / "feature_compare_python_c_summary.csv"

values_df.to_csv(values_out, index=False)
summary_df.to_csv(summary_out, index=False)

print("First direct comparisons:")
print(values_df.head(30).to_string(index=False))

print()
print("Summary with correlation:")
print(summary_df.to_string(index=False))

print()
print("Saved:")
print(values_out)
print(summary_out)
