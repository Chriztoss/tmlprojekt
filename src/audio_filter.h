#ifndef AUDIO_FILTER_H
#define AUDIO_FILTER_H

#include <stdint.h>

void audio_filter_bandpass_700_3500(float *samples, uint32_t length, float sample_rate);

#endif