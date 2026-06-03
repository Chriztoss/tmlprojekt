#ifndef CHORD_MODEL_INFO_H
#define CHORD_MODEL_INFO_H

#define CHORD_NUM_CLASSES 12
#define CHORD_NUM_FEATURES 36

static const char *CHORD_LABELS[CHORD_NUM_CLASSES] = {
    "Adur",
    "Aisdur",
    "Bdur",
    "Cdur",
    "Cisdur",
    "Ddur",
    "Disdur",
    "Edur",
    "Fdur",
    "Fisdur",
    "Gdur",
    "Gisdur",
};

static const char *CHORD_FEATURE_NAMES[CHORD_NUM_FEATURES] = {
    "spectral_centroid",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "chroma_stft_C",
    "chroma_stft_C#",
    "chroma_stft_D",
    "chroma_stft_D#",
    "chroma_stft_E",
    "chroma_stft_F",
    "chroma_stft_F#",
    "chroma_stft_G",
    "chroma_stft_G#",
    "chroma_stft_A",
    "chroma_stft_A#",
    "chroma_stft_B",
    "chroma_cens_C",
    "chroma_cens_C#",
    "chroma_cens_D",
    "chroma_cens_D#",
    "chroma_cens_E",
    "chroma_cens_F",
    "chroma_cens_F#",
    "chroma_cens_G",
    "chroma_cens_G#",
    "chroma_cens_A",
    "chroma_cens_A#",
    "chroma_cens_B",
    "tonnetz_1",
    "tonnetz_2",
    "tonnetz_3",
    "tonnetz_4",
    "tonnetz_5",
    "tonnetz_6",
};

#endif
