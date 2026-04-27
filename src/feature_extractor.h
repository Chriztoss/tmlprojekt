#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FE_SAMPLE_RATE      8000.0f
#define FE_FRAME_LEN        1024
#define FE_HOP_LEN          512
#define FE_SPEC_BINS        (FE_FRAME_LEN / 2 + 1)

#define FE_NUM_MFCC         5
#define FE_NUM_CHROMA       12
#define FE_NUM_TONNETZ      6
#define FE_NUM_MEL_BANDS    20
#define FE_CENS_HISTORY     8

typedef struct
{
    float spectral_centroid;
    float mfcc[FE_NUM_MFCC];
    float chroma_stft[FE_NUM_CHROMA];
    float chroma_cens[FE_NUM_CHROMA];
    float tonnetz[FE_NUM_TONNETZ];
} FeatureVector;

typedef struct
{
    float sample_rate;

    float window[FE_FRAME_LEN];
    float frame[FE_FRAME_LEN];
    float power_spec[FE_SPEC_BINS];

    float chroma_fb[FE_NUM_CHROMA][FE_SPEC_BINS];
    float tonnetz_basis[FE_NUM_TONNETZ][FE_NUM_CHROMA];
    float mel_fb[FE_NUM_MEL_BANDS][FE_SPEC_BINS];

    float cens_history[FE_CENS_HISTORY][FE_NUM_CHROMA];
    int cens_index;
    int cens_count;

    int initialized;
} FeatureExtractor;

void feature_vector_zero(FeatureVector *fv);
bool feature_extractor_init(FeatureExtractor *fx, float sample_rate);

bool feature_extractor_process_segment(
    FeatureExtractor *fx,
    const float *samples,
    uint32_t num_samples,
    FeatureVector *out_features
);

void feature_debug_print(const FeatureVector *fv);

#ifdef __cplusplus
}
#endif

#endif