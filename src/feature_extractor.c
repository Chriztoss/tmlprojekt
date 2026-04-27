#include "feature_extractor.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FE_EPS 1e-12f

static const char *NOTE_NAMES[FE_NUM_CHROMA] = {
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
};

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void feature_vector_zero(FeatureVector *fv)
{
    if (fv == NULL) return;
    memset(fv, 0, sizeof(FeatureVector));
}

static void build_hann_window(float *w, int N)
{
    for (int n = 0; n < N; n++)
    {
        w[n] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)n) / (float)(N - 1));
    }
}

static void clear_cens_history(FeatureExtractor *fx)
{
    memset(fx->cens_history, 0, sizeof(fx->cens_history));
    fx->cens_index = 0;
    fx->cens_count = 0;
}

static void build_chroma_filterbank(FeatureExtractor *fx)
{
    memset(fx->chroma_fb, 0, sizeof(fx->chroma_fb));

    for (int k = 1; k < FE_SPEC_BINS; k++)
    {
        float freq = ((float)k * fx->sample_rate) / (float)FE_FRAME_LEN;

        if (freq < 40.0f)
            continue;

        {
            float midi = 69.0f + 12.0f * log2f(freq / 440.0f);
            float pc = fmodf(midi, 12.0f);
            int pc0, pc1;
            float frac;

            if (pc < 0.0f)
                pc += 12.0f;

            pc0 = (int)floorf(pc);
            pc1 = (pc0 + 1) % 12;
            frac = pc - (float)pc0;

            fx->chroma_fb[pc0][k] += (1.0f - frac);
            fx->chroma_fb[pc1][k] += frac;
        }
    }
}

static void build_tonnetz_basis(FeatureExtractor *fx)
{
    for (int pc = 0; pc < FE_NUM_CHROMA; pc++)
    {
        float p = (float)pc;

        float angle_fifth     = 2.0f * (float)M_PI * (7.0f * p / 12.0f);
        float angle_min_third = 2.0f * (float)M_PI * (3.0f * p / 12.0f);
        float angle_maj_third = 2.0f * (float)M_PI * (4.0f * p / 12.0f);

        fx->tonnetz_basis[0][pc] = cosf(angle_fifth);
        fx->tonnetz_basis[1][pc] = sinf(angle_fifth);

        fx->tonnetz_basis[2][pc] = cosf(angle_min_third);
        fx->tonnetz_basis[3][pc] = sinf(angle_min_third);

        fx->tonnetz_basis[4][pc] = cosf(angle_maj_third);
        fx->tonnetz_basis[5][pc] = sinf(angle_maj_third);
    }
}

static void build_mel_filterbank(FeatureExtractor *fx)
{
    float fmin = 0.0f;
    float fmax = fx->sample_rate * 0.5f;
    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    float mel_points[FE_NUM_MEL_BANDS + 2];
    float hz_points[FE_NUM_MEL_BANDS + 2];
    int   bin_points[FE_NUM_MEL_BANDS + 2];

    memset(fx->mel_fb, 0, sizeof(fx->mel_fb));

    for (int i = 0; i < FE_NUM_MEL_BANDS + 2; i++)
    {
        float t = (float)i / (float)(FE_NUM_MEL_BANDS + 1);
        int bin;

        mel_points[i] = mel_min + t * (mel_max - mel_min);
        hz_points[i] = mel_to_hz(mel_points[i]);

        bin = (int)floorf(((float)FE_FRAME_LEN + 1.0f) * hz_points[i] / fx->sample_rate);

        if (bin < 0) bin = 0;
        if (bin >= FE_SPEC_BINS) bin = FE_SPEC_BINS - 1;

        bin_points[i] = bin;
    }

    for (int m = 0; m < FE_NUM_MEL_BANDS; m++)
    {
        int left   = bin_points[m];
        int center = bin_points[m + 1];
        int right  = bin_points[m + 2];

        if (center <= left) center = left + 1;
        if (right <= center) right = center + 1;
        if (right >= FE_SPEC_BINS) right = FE_SPEC_BINS - 1;

        for (int k = left; k < center; k++)
        {
            fx->mel_fb[m][k] = (float)(k - left) / (float)(center - left);
        }

        for (int k = center; k < right; k++)
        {
            fx->mel_fb[m][k] = (float)(right - k) / (float)(right - center);
        }
    }
}

bool feature_extractor_init(FeatureExtractor *fx, float sample_rate)
{
    if (fx == NULL)
        return false;

    memset(fx, 0, sizeof(FeatureExtractor));

    fx->sample_rate = sample_rate;

    build_hann_window(fx->window, FE_FRAME_LEN);
    build_chroma_filterbank(fx);
    build_tonnetz_basis(fx);
    build_mel_filterbank(fx);
    clear_cens_history(fx);

    fx->initialized = 1;
    return true;
}

static void copy_and_window_frame(
    FeatureExtractor *fx,
    const float *samples,
    uint32_t start_idx,
    uint32_t num_samples
)
{
    for (int n = 0; n < FE_FRAME_LEN; n++)
    {
        uint32_t idx = start_idx + (uint32_t)n;
        float x = 0.0f;

        if (idx < num_samples)
            x = samples[idx];

        fx->frame[n] = x * fx->window[n];
    }
}

static void compute_power_spectrum_naive(
    const float *frame,
    float *power_spec,
    float sample_rate
)
{
    (void)sample_rate;

    for (int k = 0; k < FE_SPEC_BINS; k++)
    {
        float re = 0.0f;
        float im = 0.0f;

        for (int n = 0; n < FE_FRAME_LEN; n++)
        {
            float angle = -2.0f * (float)M_PI * (float)k * (float)n / (float)FE_FRAME_LEN;
            re += frame[n] * cosf(angle);
            im += frame[n] * sinf(angle);
        }

        power_spec[k] = re * re + im * im;
    }
}

static float compute_spectral_centroid(
    const float *power_spec,
    float sample_rate
)
{
    float num = 0.0f;
    float den = 0.0f;

    for (int k = 0; k < FE_SPEC_BINS; k++)
    {
        float freq = ((float)k * sample_rate) / (float)FE_FRAME_LEN;
        num += freq * power_spec[k];
        den += power_spec[k];
    }

    if (den < FE_EPS)
        return 0.0f;

    return num / den;
}

static void compute_chroma_stft(
    const float power_spec[FE_SPEC_BINS],
    const float chroma_fb[FE_NUM_CHROMA][FE_SPEC_BINS],
    float chroma_out[FE_NUM_CHROMA]
)
{
    float norm = 0.0f;

    for (int c = 0; c < FE_NUM_CHROMA; c++)
    {
        float sum = 0.0f;

        for (int k = 0; k < FE_SPEC_BINS; k++)
        {
            sum += chroma_fb[c][k] * power_spec[k];
        }

        chroma_out[c] = sum;
        norm += sum * sum;
    }

    norm = sqrtf(norm) + FE_EPS;

    for (int c = 0; c < FE_NUM_CHROMA; c++)
    {
        chroma_out[c] /= norm;
    }
}

static float quantize_cens_value(float x)
{
    if (x > 0.40f) return 4.0f;
    if (x > 0.20f) return 3.0f;
    if (x > 0.10f) return 2.0f;
    if (x > 0.05f) return 1.0f;
    return 0.0f;
}

static void compute_chroma_cens(
    FeatureExtractor *fx,
    const float chroma_in[FE_NUM_CHROMA],
    float cens_out[FE_NUM_CHROMA]
)
{
    float temp[FE_NUM_CHROMA];
    float l1 = 0.0f;
    float norm = 0.0f;

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        l1 += fabsf(chroma_in[i]);
    }
    l1 += FE_EPS;

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        float x = chroma_in[i] / l1;
        temp[i] = quantize_cens_value(x);
    }

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        fx->cens_history[fx->cens_index][i] = temp[i];
    }

    fx->cens_index = (fx->cens_index + 1) % FE_CENS_HISTORY;
    if (fx->cens_count < FE_CENS_HISTORY)
        fx->cens_count++;

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        float sum = 0.0f;

        for (int h = 0; h < fx->cens_count; h++)
        {
            sum += fx->cens_history[h][i];
        }

        cens_out[i] = sum / (float)fx->cens_count;
        norm += cens_out[i] * cens_out[i];
    }

    norm = sqrtf(norm) + FE_EPS;

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        cens_out[i] /= norm;
    }
}

static void compute_tonnetz(
    const float tonnetz_basis[FE_NUM_TONNETZ][FE_NUM_CHROMA],
    const float chroma_in[FE_NUM_CHROMA],
    float tonnetz_out[FE_NUM_TONNETZ]
)
{
    for (int t = 0; t < FE_NUM_TONNETZ; t++)
    {
        float sum = 0.0f;

        for (int c = 0; c < FE_NUM_CHROMA; c++)
        {
            sum += tonnetz_basis[t][c] * chroma_in[c];
        }

        tonnetz_out[t] = sum;
    }
}

static void compute_mfcc(
    const float power_spec[FE_SPEC_BINS],
    const float mel_fb[FE_NUM_MEL_BANDS][FE_SPEC_BINS],
    float mfcc_out[FE_NUM_MFCC]
)
{
    float log_mel[FE_NUM_MEL_BANDS];

    for (int m = 0; m < FE_NUM_MEL_BANDS; m++)
    {
        float sum = 0.0f;

        for (int k = 0; k < FE_SPEC_BINS; k++)
        {
            sum += mel_fb[m][k] * power_spec[k];
        }

        log_mel[m] = logf(sum + FE_EPS);
    }

    for (int n = 0; n < FE_NUM_MFCC; n++)
    {
        float acc = 0.0f;

        for (int m = 0; m < FE_NUM_MEL_BANDS; m++)
        {
            acc += log_mel[m] *
                   cosf((float)M_PI * (float)n * (2.0f * (float)m + 1.0f) /
                        (2.0f * (float)FE_NUM_MEL_BANDS));
        }

        mfcc_out[n] = acc;
    }
}

static void accumulate_features(FeatureVector *dst, const FeatureVector *src)
{
    dst->spectral_centroid += src->spectral_centroid;

    for (int i = 0; i < FE_NUM_MFCC; i++)
        dst->mfcc[i] += src->mfcc[i];

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        dst->chroma_stft[i] += src->chroma_stft[i];
        dst->chroma_cens[i] += src->chroma_cens[i];
    }

    for (int i = 0; i < FE_NUM_TONNETZ; i++)
        dst->tonnetz[i] += src->tonnetz[i];
}

static void divide_features(FeatureVector *fv, float denom)
{
    if (denom < FE_EPS)
        return;

    fv->spectral_centroid /= denom;

    for (int i = 0; i < FE_NUM_MFCC; i++)
        fv->mfcc[i] /= denom;

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        fv->chroma_stft[i] /= denom;
        fv->chroma_cens[i] /= denom;
    }

    for (int i = 0; i < FE_NUM_TONNETZ; i++)
        fv->tonnetz[i] /= denom;
}

static bool process_one_frame(
    FeatureExtractor *fx,
    const float *samples,
    uint32_t start_idx,
    uint32_t num_samples,
    FeatureVector *frame_features
)
{
    feature_vector_zero(frame_features);

    copy_and_window_frame(fx, samples, start_idx, num_samples);
    compute_power_spectrum_naive(fx->frame, fx->power_spec, fx->sample_rate);

    frame_features->spectral_centroid =
        compute_spectral_centroid(fx->power_spec, fx->sample_rate);

    compute_chroma_stft(
        fx->power_spec,
        fx->chroma_fb,
        frame_features->chroma_stft
    );

    compute_chroma_cens(
        fx,
        frame_features->chroma_stft,
        frame_features->chroma_cens
    );

    compute_tonnetz(
        fx->tonnetz_basis,
        frame_features->chroma_stft,
        frame_features->tonnetz
    );

    compute_mfcc(
        fx->power_spec,
        fx->mel_fb,
        frame_features->mfcc
    );

    return true;
}

bool feature_extractor_process_segment(
    FeatureExtractor *fx,
    const float *samples,
    uint32_t num_samples,
    FeatureVector *out_features
)
{
    uint32_t frame_start = 0;
    uint32_t frame_count = 0;

    if (fx == NULL || samples == NULL || out_features == NULL)
        return false;

    if (!fx->initialized)
        return false;

    feature_vector_zero(out_features);
    clear_cens_history(fx);

    while (frame_start < num_samples)
    {
        FeatureVector frame_features;

        if (!process_one_frame(fx, samples, frame_start, num_samples, &frame_features))
            return false;

        accumulate_features(out_features, &frame_features);
        frame_count++;

        if (num_samples <= FE_FRAME_LEN && frame_start == 0)
            break;

        frame_start += FE_HOP_LEN;

        if (frame_start >= num_samples && num_samples >= FE_FRAME_LEN)
            break;
    }

    if (frame_count == 0)
        return false;

    divide_features(out_features, (float)frame_count);
    return true;
}

void feature_debug_print(const FeatureVector *fv)
{
    if (fv == NULL)
        return;

    printf("spectral_centroid = %.6f\n", fv->spectral_centroid);

    for (int i = 0; i < FE_NUM_MFCC; i++)
    {
        printf("mfcc_%d = %.6f\n", i + 1, fv->mfcc[i]);
    }

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        printf("chroma_stft_%s = %.6f\n", NOTE_NAMES[i], fv->chroma_stft[i]);
    }

    for (int i = 0; i < FE_NUM_CHROMA; i++)
    {
        printf("chroma_cens_%s = %.6f\n", NOTE_NAMES[i], fv->chroma_cens[i]);
    }

    for (int i = 0; i < FE_NUM_TONNETZ; i++)
    {
        printf("tonnetz_%d = %.6f\n", i + 1, fv->tonnetz[i]);
    }
}