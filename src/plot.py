from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# FILER
# =========================
DATASET_FILE = Path("feature_dataset.csv")
SEPARATION_FILE = Path("feature_separation_scores.csv")

# =========================
# LOAD DATA
# =========================
print("Indlæser CSV-filer...")

df = pd.read_csv(DATASET_FILE)
sep_df = pd.read_csv(SEPARATION_FILE)

class_order = sorted(df["label"].unique())

# =========================
# FEATURE-GRUPPER
# =========================
spectral_features = ["spectral_centroid"]
mfcc_features = [f"mfcc_{i}" for i in range(1, 6)]
tonnetz_features = [f"tonnetz_{i}" for i in range(1, 7)]

note_order = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

chroma_stft_features = [
    f"chroma_stft_{n}" for n in note_order if f"chroma_stft_{n}" in df.columns
]
chroma_cens_features = [
    f"chroma_cens_{n}" for n in note_order if f"chroma_cens_{n}" in df.columns
]

# =========================
# HJÆLPEFUNKTIONER
# =========================
def make_boxplot(df, feature_name, class_order):
    print(f"Laver boxplot for {feature_name}...")

    plt.figure(figsize=(11, 5))
    data = [df[df["label"] == label][feature_name].values for label in class_order]

    plt.boxplot(data, tick_labels=class_order)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(feature_name)
    plt.title(f"Fordeling af {feature_name} for hver klasse")
    plt.tight_layout()
    plt.savefig(f"boxplot_{feature_name}.png", dpi=300)
    plt.close()


def make_barplot_top_features(sep_df, top_n=15):
    print("Laver plot over top-features...")

    top_sep = sep_df.head(top_n)

    plt.figure(figsize=(12, 5))
    plt.bar(top_sep["feature"], top_sep["separation_score"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Separation score")
    plt.title(f"Top {top_n} features med størst adskillelse mellem klasser")
    plt.tight_layout()
    plt.savefig("plot_top_features.png", dpi=300)
    plt.close()


def make_group_heatmap(df, feature_cols, class_order, title, filename, normalize=False):
    print(f"Laver {filename}...")

    mean_table = df.groupby("label")[feature_cols].mean().loc[class_order]

    if normalize:
        mean_table = (mean_table - mean_table.min()) / (
            mean_table.max() - mean_table.min() + 1e-12
        )

    plt.figure(figsize=(12, 6))
    plt.imshow(mean_table.values, aspect="auto")
    plt.colorbar(label="Feature-værdi" if not normalize else "Normaliseret feature-værdi")
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.yticks(range(len(class_order)), class_order)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# =========================
# 1. TOP FEATURES
# =========================
make_barplot_top_features(sep_df, top_n=15)

# =========================
# 2. HEATMAPS
# =========================
make_group_heatmap(
    df,
    chroma_stft_features,
    class_order,
    "Heatmap af Chroma STFT pr. klasse",
    "heatmap_chroma_stft.png",
    normalize=False
)

make_group_heatmap(
    df,
    chroma_cens_features,
    class_order,
    "Heatmap af Chroma CENS pr. klasse",
    "heatmap_chroma_cens.png",
    normalize=False
)

make_group_heatmap(
    df,
    tonnetz_features,
    class_order,
    "Heatmap af Tonnetz pr. klasse",
    "heatmap_tonnetz.png",
    normalize=True
)

make_group_heatmap(
    df,
    mfcc_features,
    class_order,
    "Heatmap af MFCC pr. klasse",
    "heatmap_mfcc.png",
    normalize=True
)

make_group_heatmap(
    df,
    spectral_features,
    class_order,
    "Heatmap af Spectral Centroid pr. klasse",
    "heatmap_spectral_centroid.png",
    normalize=True
)

# =========================
# 3. BOXPLOTS FOR NOGLE FASTE FEATURES
# =========================
selected_boxplots = [
    "spectral_centroid",
    "mfcc_1",
    "mfcc_2",
    "tonnetz_1",
    "tonnetz_2",
    "chroma_stft_C",
    "chroma_stft_G",
    "chroma_cens_C",
    "chroma_cens_G"
]

for feat in selected_boxplots:
    if feat in df.columns:
        make_boxplot(df, feat, class_order)

# =========================
# 4. BOXPLOTS FOR TOP 6 FEATURES
# =========================
top6 = sep_df.head(6)["feature"].tolist()

for feat in top6:
    if feat in df.columns:
        make_boxplot(df, feat, class_order)

print("\nGrafer gemt som:")
print("- plot_top_features.png")
print("- heatmap_chroma_stft.png")
print("- heatmap_chroma_cens.png")
print("- heatmap_tonnetz.png")
print("- heatmap_mfcc.png")
print("- heatmap_spectral_centroid.png")

for feat in selected_boxplots:
    if feat in df.columns:
        print(f"- boxplot_{feat}.png")

for feat in top6:
    if feat in df.columns:
        print(f"- boxplot_{feat}.png")