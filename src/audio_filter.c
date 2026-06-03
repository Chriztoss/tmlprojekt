#include "audio_filter.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct
{
    float x_prev;
    float y_prev;
} HighPassState;

typedef struct
{
    float y_prev;
} LowPassState;

static float highpass_process(HighPassState *hp, float x, float sample_rate, float cutoff)
{
    float dt = 1.0f / sample_rate;
    float rc = 1.0f / (2.0f * (float)M_PI * cutoff);
    float alpha = rc / (rc + dt);

    float y = alpha * (hp->y_prev + x - hp->x_prev);

    hp->x_prev = x;
    hp->y_prev = y;

    return y;
}

static float lowpass_process(LowPassState *lp, float x, float sample_rate, float cutoff)
{
    float dt = 1.0f / sample_rate;
    float rc = 1.0f / (2.0f * (float)M_PI * cutoff);
    float alpha = dt / (rc + dt);

    float y = lp->y_prev + alpha * (x - lp->y_prev);

    lp->y_prev = y;

    return y;
}

void audio_filter_bandpass_700_3500(float *samples, uint32_t length, float sample_rate)
{
    HighPassState hp = {0};
    LowPassState lp = {0};

    for (uint32_t i = 0; i < length; i++)
    {
        float y = highpass_process(&hp, samples[i], sample_rate, 700.0f);
        y = lowpass_process(&lp, y, sample_rate, 3500.0f);

        samples[i] = y;
    }
}