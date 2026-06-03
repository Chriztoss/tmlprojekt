#ifndef TEST_VECTOR_H
#define TEST_VECTOR_H

#include "chord_model_info.h"

static const char *TEST_LABEL = "Adur";

static float TEST_FEATURES[CHORD_NUM_FEATURES] = {
    647.442831f, // spectral_centroid
    88.3127561f, // mfcc_1
    35.0168665f, // mfcc_2
    7.05221966f, // mfcc_3
    7.32970504f, // mfcc_4
    3.5322126f, // mfcc_5
    0.40647595f, // chroma_stft_C
    0.605547482f, // chroma_stft_C#
    0.417159641f, // chroma_stft_D
    0.34931273f, // chroma_stft_D#
    0.581700853f, // chroma_stft_E
    0.460674284f, // chroma_stft_F
    0.407913718f, // chroma_stft_F#
    0.415861231f, // chroma_stft_G
    0.776410257f, // chroma_stft_G#
    0.924489485f, // chroma_stft_A
    0.59699955f, // chroma_stft_A#
    0.569545228f, // chroma_stft_B
    0.139649425f, // chroma_cens_C
    0.269668183f, // chroma_cens_C#
    0.146171316f, // chroma_cens_D
    0.091921953f, // chroma_cens_D#
    0.298918195f, // chroma_cens_E
    0.179202672f, // chroma_cens_F
    0.140077577f, // chroma_cens_F#
    0.175391545f, // chroma_cens_G
    0.430607475f, // chroma_cens_G#
    0.470599532f, // chroma_cens_A
    0.277932022f, // chroma_cens_A#
    0.225680924f, // chroma_cens_B
    -0.066333821f, // tonnetz_1
    0.032309115f, // tonnetz_2
    0.052596652f, // tonnetz_3
    0.100734483f, // tonnetz_4
    -0.019004248f, // tonnetz_5
    -0.003149179f, // tonnetz_6
};

#endif
