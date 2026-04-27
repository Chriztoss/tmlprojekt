#include "feature_extractor.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void)
{
    FeatureExtractor fx;
    FeatureVector fv;

    const float duration_sec = 2.5f;
    const uint32_t num_samples = (uint32_t)(FE_SAMPLE_RATE * duration_sec);

    float *buffer = (float *)malloc(num_samples * sizeof(float));
    if (buffer == NULL)
    {
        printf("Kunne ikke allokere buffer\n");
        return 1;
    }

    /* A-dur akkord: A, C#, E */
    const float f1 = 110.0f;     /* A2 */
    const float f2 = 138.59f;    /* C#3 */
    const float f3 = 164.81f;    /* E3 */

    for (uint32_t n = 0; n < num_samples; n++)
    {
        float t = (float)n / FE_SAMPLE_RATE;

        buffer[n] =
            0.5f * sinf(2.0f * (float)M_PI * f1 * t) +
            0.4f * sinf(2.0f * (float)M_PI * f2 * t) +
            0.4f * sinf(2.0f * (float)M_PI * f3 * t);
    }

    if (!feature_extractor_init(&fx, FE_SAMPLE_RATE))
    {
        printf("feature_extractor_init fejlede\n");
        free(buffer);
        return 1;
    }

    if (!feature_extractor_process_segment(&fx, buffer, num_samples, &fv))
    {
        printf("feature_extractor_process_segment fejlede\n");
        free(buffer);
        return 1;
    }

    printf("Feature-beregning OK\n\n");
    feature_debug_print(&fv);

    free(buffer);
    return 0;
}