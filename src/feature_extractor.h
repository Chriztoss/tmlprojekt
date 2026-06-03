#ifndef FEATURE2_EXTRACTOR_H
#define FEATURE2_EXTRACTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define F2_SAMPLE_RATE      8000.0
#define F2_FRAME_LEN        2048
#define F2_HOP_LEN          512
#define F2_SPEC_BINS        (F2_FRAME_LEN / 2 + 1)

#define F2_NUM_MFCC         5
#define F2_NUM_CHROMA       12
#define F2_NUM_MEL_BANDS    26

#define F2_NUM_FEATURES     (1 + F2_NUM_MFCC + F2_NUM_CHROMA)

typedef struct
{
    double spectral_centroid;
    double mfcc[F2_NUM_MFCC];
    double chroma_stft[F2_NUM_CHROMA];
} Feature2Vector;

typedef struct
{
    double sample_rate;

    double window[F2_FRAME_LEN];
    double fft_re[F2_FRAME_LEN];
    double fft_im[F2_FRAME_LEN];

    double magnitude[F2_SPEC_BINS];
    double power[F2_SPEC_BINS];

    int mel_bin_points[F2_NUM_MEL_BANDS + 2];

    int initialized;
} Feature2Extractor;

void feature2_vector_zero(Feature2Vector *fv);
bool feature2_extractor_init(Feature2Extractor *fx, double sample_rate);

/*
   samples skal helst allerede være preprocessing-matchet til feature2.py:
   - samplet/resamplet til 8 kHz
   - silence trimmet
   - DC fjernet
   - peak-normaliseret
*/
bool feature2_extract_segment(
    Feature2Extractor *fx,
    const double *samples,
    uint32_t num_samples,
    Feature2Vector *out_features
);

void feature2_debug_print(const Feature2Vector *fv);

#ifdef __cplusplus
}
#endif

#endif
