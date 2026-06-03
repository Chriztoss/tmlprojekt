#ifndef CHORD_KNN_MODEL_INFO_H
#define CHORD_KNN_MODEL_INFO_H

#include <stdint.h>

#ifndef EML_NEIGHBORS_MAX_CLASSES
#define EML_NEIGHBORS_MAX_CLASSES 12
#endif

#define CHORD_KNN_NUM_CLASSES 12
#define CHORD_KNN_NUM_FEATURES 18
#define CHORD_KNN_QUANT_SCALE 100.0f

static const char * const CHORD_KNN_LABELS[CHORD_KNN_NUM_CLASSES] = {
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

static const float CHORD_KNN_MEAN[CHORD_KNN_NUM_FEATURES] = {
    648.975202f,
    99.6634426f,
    37.6775499f,
    7.3431676f,
    5.93540666f,
    1.62707198f,
    0.460577017f,
    0.469540908f,
    0.442027477f,
    0.355427608f,
    0.378110372f,
    0.474832253f,
    0.425512009f,
    0.449760244f,
    0.698449417f,
    0.567501024f,
    0.566887829f,
    0.583869677f,
};

static const float CHORD_KNN_SCALE[CHORD_KNN_NUM_FEATURES] = {
    60.8070782f,
    12.6158052f,
    3.84708818f,
    3.21540207f,
    2.48352071f,
    2.45643496f,
    0.207838143f,
    0.189481669f,
    0.20116709f,
    0.143318607f,
    0.127074214f,
    0.181969631f,
    0.181190943f,
    0.17221477f,
    0.167498039f,
    0.213361222f,
    0.227796561f,
    0.210458307f,
};

static void chord_knn_prepare_features(const float *raw, int16_t *out)
{
    for (int i = 0; i < CHORD_KNN_NUM_FEATURES; i++)
    {
        float z = (raw[i] - CHORD_KNN_MEAN[i]) / CHORD_KNN_SCALE[i];
        float qf = z * CHORD_KNN_QUANT_SCALE;

        if (qf > 32767.0f) qf = 32767.0f;
        if (qf < -32768.0f) qf = -32768.0f;

        if (qf >= 0.0f)
            out[i] = (int16_t)(qf + 0.5f);
        else
            out[i] = (int16_t)(qf - 0.5f);
    }
}

#endif
