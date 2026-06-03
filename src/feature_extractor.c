#include "feature_extractor.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define F2_EPS 1e-12

static const char *NOTE_NAMES[F2_NUM_CHROMA] = {
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
};

static double hz_to_mel(double hz)
{
    return 2595.0 * log10(1.0 + hz / 700.0);
}

static double mel_to_hz(double mel)
{
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

void feature2_vector_zero(Feature2Vector *fv)
{
    if (fv != NULL)
    {
        memset(fv, 0, sizeof(Feature2Vector));
    }
}

/* Matcher feature2.py: 0.5 - 0.5*cos(2*pi*i/(N-1)) */
static void build_hann_window(double *w, int n)
{
    if (n <= 1)
    {
        if (n == 1)
        {
            w[0] = 1.0;
        }
        return;
    }

    for (int i = 0; i < n; i++)
    {
        w[i] = 0.5 - 0.5 * cos((2.0 * (double)M_PI * (double)i) / (double)(n - 1));
    }
}

/*
   RAM-optimeret mel-filterbank:
   Vi gemmer kun de 28 bin-punkter i stedet for en 26 x 1025 matrix.
*/
static void build_mel_bin_points(Feature2Extractor *fx)
{
    const int n_bins = F2_SPEC_BINS;

    double mel_min = hz_to_mel(0.0);
    double mel_max = hz_to_mel(fx->sample_rate * 0.5);

    for (int i = 0; i < F2_NUM_MEL_BANDS + 2; i++)
    {
        double mel = mel_min +
                     (double)i * (mel_max - mel_min) /
                     (double)(F2_NUM_MEL_BANDS + 1);

        double hz = mel_to_hz(mel);

        int b = (int)floor(((double)F2_FRAME_LEN + 1.0) * hz / fx->sample_rate);

        if (b < 0)
        {
            b = 0;
        }

        if (b >= n_bins)
        {
            b = n_bins - 1;
        }

        fx->mel_bin_points[i] = b;
    }
}

bool feature2_extractor_init(Feature2Extractor *fx, double sample_rate)
{
    if (fx == NULL)
    {
        return false;
    }

    memset(fx, 0, sizeof(Feature2Extractor));

    fx->sample_rate = sample_rate;

    build_hann_window(fx->window, F2_FRAME_LEN);
    build_mel_bin_points(fx);

    fx->initialized = 1;

    return true;
}

static int is_power_of_two(uint32_t n)
{
    return n > 0 && ((n & (n - 1U)) == 0U);
}

static void bit_reverse(double *re, double *im, uint32_t n)
{
    uint32_t j = 0;

    for (uint32_t i = 1; i < n; i++)
    {
        uint32_t bit = n >> 1;

        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j)
        {
            double tr = re[i];
            double ti = im[i];

            re[i] = re[j];
            im[i] = im[j];

            re[j] = tr;
            im[j] = ti;
        }
    }
}

static void fft_radix2(double *re, double *im, uint32_t n)
{
    if (!is_power_of_two(n))
    {
        return;
    }

    bit_reverse(re, im, n);

    for (uint32_t length = 2; length <= n; length <<= 1)
    {
        double angle = -2.0 * (double)M_PI / (double)length;
        double wlen_re = cos(angle);
        double wlen_im = sin(angle);
        uint32_t half = length >> 1;

        for (uint32_t start = 0; start < n; start += length)
        {
            double w_re = 1.0;
            double w_im = 0.0;

            for (uint32_t k = start; k < start + half; k++)
            {
                uint32_t j = k + half;

                double v_re = re[j] * w_re - im[j] * w_im;
                double v_im = re[j] * w_im + im[j] * w_re;

                double u_re = re[k];
                double u_im = im[k];

                re[k] = u_re + v_re;
                im[k] = u_im + v_im;

                re[j] = u_re - v_re;
                im[j] = u_im - v_im;

                double next_re = w_re * wlen_re - w_im * wlen_im;
                double next_im = w_re * wlen_im + w_im * wlen_re;

                w_re = next_re;
                w_im = next_im;
            }
        }
    }
}

static void compute_spectrum_for_frame(
    Feature2Extractor *fx,
    const double *samples,
    uint32_t start_idx,
    uint32_t num_samples
)
{
    for (int n = 0; n < F2_FRAME_LEN; n++)
    {
        uint32_t idx = start_idx + (uint32_t)n;

        double x = 0.0;

        if (idx < num_samples)
        {
            x = samples[idx];
        }

        fx->fft_re[n] = x * fx->window[n];
        fx->fft_im[n] = 0.0;
    }

    fft_radix2(fx->fft_re, fx->fft_im, F2_FRAME_LEN);

    for (int k = 0; k < F2_SPEC_BINS; k++)
    {
        double re = fx->fft_re[k];
        double im = fx->fft_im[k];

        double p = re * re + im * im;

        fx->power[k] = p;
        fx->magnitude[k] = sqrt(p);
    }
}

static double compute_centroid(const double *mag, double sample_rate)
{
    double numerator = 0.0;
    double denominator = 0.0;

    for (int k = 0; k < F2_SPEC_BINS; k++)
    {
        double freq = ((double)k * sample_rate) / (double)F2_FRAME_LEN;

        numerator += freq * mag[k];
        denominator += mag[k];
    }

    return numerator / (denominator + F2_EPS);
}

static void compute_mfcc(
    const double power[F2_SPEC_BINS],
    const int mel_bin_points[F2_NUM_MEL_BANDS + 2],
    double mfcc_out[F2_NUM_MFCC]
)
{
    double log_mel[F2_NUM_MEL_BANDS];

    for (int m = 0; m < F2_NUM_MEL_BANDS; m++)
    {
        int left = mel_bin_points[m];
        int center = mel_bin_points[m + 1];
        int right = mel_bin_points[m + 2];

        double e = 0.0;

        if (center > left)
        {
            for (int k = left; k < center && k < F2_SPEC_BINS; k++)
            {
                double w = (double)(k - left) / (double)(center - left);
                e += power[k] * w;
            }
        }

        if (right > center)
        {
            for (int k = center; k < right && k < F2_SPEC_BINS; k++)
            {
                double w = (double)(right - k) / (double)(right - center);
                e += power[k] * w;
            }
        }

        log_mel[m] = log(e + F2_EPS);
    }

    for (int k = 0; k < F2_NUM_MFCC; k++)
    {
        double s = 0.0;

        for (int i = 0; i < F2_NUM_MEL_BANDS; i++)
        {
            double angle = (double)M_PI * (double)k * (2.0 * (double)i + 1.0) /
                           (2.0 * (double)F2_NUM_MEL_BANDS);

            s += log_mel[i] * cos(angle);
        }

        mfcc_out[k] = s;
    }
}

static int rounded_midi_from_freq(double freq)
{
    double midi = 69.0 + 12.0 * log2(freq / 440.0);

    if (midi >= 0.0)
    {
        return (int)floor(midi + 0.5);
    }
    else
    {
        return (int)ceil(midi - 0.5);
    }
}

static void compute_chroma_from_magnitude(
    const double magnitude[F2_SPEC_BINS],
    double sample_rate,
    double chroma_out[F2_NUM_CHROMA]
)
{
    for (int i = 0; i < F2_NUM_CHROMA; i++)
    {
        chroma_out[i] = 0.0;
    }

    for (int k = 1; k < F2_SPEC_BINS; k++)
    {
        double freq = ((double)k * sample_rate) / (double)F2_FRAME_LEN;

        if (freq < 50.0)
        {
            continue;
        }

        int midi = rounded_midi_from_freq(freq);
        int pitch_class = midi % 12;

        if (pitch_class < 0)
        {
            pitch_class += 12;
        }

        chroma_out[pitch_class] += magnitude[k];
    }

    double max_val = 0.0;

    for (int i = 0; i < F2_NUM_CHROMA; i++)
    {
        if (chroma_out[i] > max_val)
        {
            max_val = chroma_out[i];
        }
    }

    if (max_val > F2_EPS)
    {
        for (int i = 0; i < F2_NUM_CHROMA; i++)
        {
            chroma_out[i] /= max_val;
        }
    }
}

bool feature2_extract_segment(
    Feature2Extractor *fx,
    const double *samples,
    uint32_t num_samples,
    Feature2Vector *out_features
)
{
    if (fx == NULL || samples == NULL || out_features == NULL)
    {
        return false;
    }

    if (!fx->initialized || num_samples == 0)
    {
        return false;
    }

    feature2_vector_zero(out_features);

    double centroid_sum = 0.0;
    double mfcc_sums[F2_NUM_MFCC] = {0};
    double chroma_sums[F2_NUM_CHROMA] = {0};

    uint32_t n_frames = 0;
    uint32_t start = 0;

    while (1)
    {
        compute_spectrum_for_frame(fx, samples, start, num_samples);

        double frame_mfcc[F2_NUM_MFCC];
        double frame_chroma[F2_NUM_CHROMA];

        centroid_sum += compute_centroid(fx->magnitude, fx->sample_rate);

        compute_mfcc(fx->power, fx->mel_bin_points, frame_mfcc);
        compute_chroma_from_magnitude(fx->magnitude, fx->sample_rate, frame_chroma);

        for (int i = 0; i < F2_NUM_MFCC; i++)
        {
            mfcc_sums[i] += frame_mfcc[i];
        }

        for (int i = 0; i < F2_NUM_CHROMA; i++)
        {
            chroma_sums[i] += frame_chroma[i];
        }

        n_frames++;

        if (num_samples <= F2_FRAME_LEN)
        {
            break;
        }

        start += F2_HOP_LEN;

        if (start >= num_samples)
        {
            break;
        }

        if (start + F2_FRAME_LEN > num_samples && num_samples - start < F2_HOP_LEN)
        {
            break;
        }
    }

    if (n_frames == 0)
    {
        n_frames = 1;
    }

    double inv_frames = 1.0 / (double)n_frames;

    out_features->spectral_centroid = centroid_sum * inv_frames;

    for (int i = 0; i < F2_NUM_MFCC; i++)
    {
        out_features->mfcc[i] = mfcc_sums[i] * inv_frames;
    }

    for (int i = 0; i < F2_NUM_CHROMA; i++)
    {
        out_features->chroma_stft[i] = chroma_sums[i] * inv_frames;
    }

    return true;
}

void feature2_debug_print(const Feature2Vector *fv)
{
    if (fv == NULL)
    {
        return;
    }

    printf("spectral_centroid = %.9f\n", fv->spectral_centroid);

    for (int i = 0; i < F2_NUM_MFCC; i++)
    {
        printf("mfcc_%d = %.9f\n", i + 1, fv->mfcc[i]);
    }

    for (int i = 0; i < F2_NUM_CHROMA; i++)
    {
        printf("chroma_stft_%s = %.9f\n", NOTE_NAMES[i], fv->chroma_stft[i]);
    }
}
