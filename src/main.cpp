#include "Particle.h"
#include <Microphone_PDM.h>

#include <stdint.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "feature_extractor.h"
}


// KNN model
#include "chord_knn_model_info.h"
#include "chord_knn_model.h"

SYSTEM_MODE(MANUAL);

const int LED_PIN = D7;

// =========================
// Indstillinger
// =========================
#define RECORD_MS 500u
#define AUDIO_SAMPLE_RATE_HZ 8000u

#define AUDIO_SAMPLES ((AUDIO_SAMPLE_RATE_HZ * RECORD_MS) / 1000u)

#define PDM_MAX_CHUNK_SAMPLES 512

// KNN modellen har 294 training samples
#define KNN_DISTANCE_BUFFER_SIZE 294

#define TRIM_TOP_DB 25.0f
#define MAX_TRIM_FRAMES 16

#define PRINT_FEATURES 0

// =========================
// Globale buffers
// =========================
// feature_extractor bruger double, derfor double her.
static double audio_buffer[AUDIO_SAMPLES];

static int16_t pdm_chunk[PDM_MAX_CHUNK_SAMPLES];

static float trim_rms_values[MAX_TRIM_FRAMES];
static uint16_t trim_starts[MAX_TRIM_FRAMES];

static Feature2Extractor fx;

static EmlNeighborsDistanceItem knn_distances[KNN_DISTANCE_BUFFER_SIZE];


static const char *chord_name(int label)
{
    switch (label)
    {
        case 0: return "Adur";
        case 1: return "Aisdur";
        case 2: return "Bdur";
        case 3: return "Cdur";
        case 4: return "Cisdur";
        case 5: return "Ddur";
        case 6: return "Disdur";
        case 7: return "Edur";
        case 8: return "Fdur";
        case 9: return "Fisdur";
        case 10: return "Gdur";
        case 11: return "Gisdur";
        default: return "Unknown";
    }
}


static void trim_silence_feature2(double *samples, uint32_t *length)
{
    if (*length <= F2_FRAME_LEN)
    {
        return;
    }

    uint32_t max_frames = 1 + ((*length - F2_FRAME_LEN) / F2_HOP_LEN);

    if (max_frames > MAX_TRIM_FRAMES)
    {
        max_frames = MAX_TRIM_FRAMES;
    }

    uint32_t count = 0;
    uint32_t start = 0;

    while ((start + F2_FRAME_LEN <= *length) && (count < max_frames))
    {
        double power = 0.0;

        for (uint32_t i = 0; i < F2_FRAME_LEN; i++)
        {
            double x = samples[start + i];
            power += x * x;
        }

        trim_rms_values[count] = (float)sqrt(power / (double)F2_FRAME_LEN);
        trim_starts[count] = (uint16_t)start;

        count++;
        start += F2_HOP_LEN;
    }

    if (count == 0)
    {
        return;
    }

    float max_rms = 0.0f;

    for (uint32_t i = 0; i < count; i++)
    {
        if (trim_rms_values[i] > max_rms)
        {
            max_rms = trim_rms_values[i];
        }
    }

    if (max_rms <= 1e-12f)
    {
        return;
    }

    float threshold = max_rms * powf(10.0f, -TRIM_TOP_DB / 20.0f);

    int first = -1;
    int last = -1;

    for (uint32_t i = 0; i < count; i++)
    {
        if (trim_rms_values[i] >= threshold)
        {
            if (first < 0)
            {
                first = (int)i;
            }

            last = (int)i;
        }
    }

    if (first < 0 || last < 0)
    {
        return;
    }

    uint32_t start_sample = (uint32_t)trim_starts[first];
    uint32_t end_sample = (uint32_t)trim_starts[last] + F2_FRAME_LEN;

    if (end_sample > *length)
    {
        end_sample = *length;
    }

    uint32_t new_len = end_sample - start_sample;

    if (new_len > 0)
    {
        memmove(samples, samples + start_sample, new_len * sizeof(double));
        *length = new_len;
    }
}


static void remove_dc_and_peak_normalize(double *samples, uint32_t length)
{
    if (length == 0)
    {
        return;
    }

    double sum = 0.0;

    for (uint32_t i = 0; i < length; i++)
    {
        sum += samples[i];
    }

    double mean = sum / (double)length;
    double peak = 0.0;

    for (uint32_t i = 0; i < length; i++)
    {
        samples[i] -= mean;

        double a = fabs(samples[i]);

        if (a > peak)
        {
            peak = a;
        }
    }

    if (peak > 1e-12)
    {
        for (uint32_t i = 0; i < length; i++)
        {
            samples[i] /= peak;
        }
    }
}


static void feature_vector_to_float_array(const Feature2Vector *fv, float features[CHORD_KNN_NUM_FEATURES])
{
    int idx = 0;

    features[idx++] = (float)fv->spectral_centroid;

    for (int i = 0; i < F2_NUM_MFCC; i++)
    {
        features[idx++] = (float)fv->mfcc[i];
    }

    for (int i = 0; i < F2_NUM_CHROMA; i++)
    {
        features[idx++] = (float)fv->chroma_stft[i];
    }

    if (idx != CHORD_KNN_NUM_FEATURES)
    {
        Serial.print("WARNING: feature count mismatch. idx=");
        Serial.print(idx);
        Serial.print(", CHORD_KNN_NUM_FEATURES=");
        Serial.println(CHORD_KNN_NUM_FEATURES);
    }
}


static void print_features(const float features[CHORD_KNN_NUM_FEATURES])
{
#if PRINT_FEATURES
    Serial.println("Features:");

    for (int i = 0; i < CHORD_KNN_NUM_FEATURES; i++)
    {
        Serial.print(i);
        Serial.print(": ");
        Serial.println(features[i], 6);
    }
#else
    (void)features;
#endif
}


// =========================
// PDM audio capture
// =========================
static bool capture_audio_block()
{
    Serial.println();
    Serial.println("Recording...");
    Serial.print("Record time: ");
    Serial.print(RECORD_MS);
    Serial.println(" ms");

    digitalWrite(LED_PIN, HIGH);

    uint32_t written = 0;
    unsigned long start_ms = millis();

    while (written < AUDIO_SAMPLES)
    {
        Microphone_PDM::instance().loop();

        if (Microphone_PDM::instance().copySamples(pdm_chunk))
        {
            size_t n = Microphone_PDM::instance().getNumberOfSamples();

            if (n > PDM_MAX_CHUNK_SAMPLES)
            {
                n = PDM_MAX_CHUNK_SAMPLES;
            }

            for (size_t i = 0; i < n && written < AUDIO_SAMPLES; i++)
            {
                audio_buffer[written] = (double)pdm_chunk[i] / 2048.0;
                written++;
            }
        }

        if (millis() - start_ms > (RECORD_MS + 2000u))
        {
            Serial.println("ERROR: recording timeout");
            digitalWrite(LED_PIN, LOW);
            return false;
        }
    }

    digitalWrite(LED_PIN, LOW);

    Serial.print("Recorded samples: ");
    Serial.println((int)written);

    return true;
}


// =========================
// KNN prediction
// =========================
static int predict_knn(const float features[CHORD_KNN_NUM_FEATURES], unsigned long *time_us)
{
    int16_t knn_features[CHORD_KNN_NUM_FEATURES];
    int16_t knn_out = -1;

    chord_knn_prepare_features(features, knn_features);

    unsigned long t0 = micros();

    EmlError err = eml_neighbors_predict(
        &chord_knn,
        knn_features,
        CHORD_KNN_NUM_FEATURES,
        knn_distances,
        KNN_DISTANCE_BUFFER_SIZE,
        &knn_out
    );

    unsigned long t1 = micros();

    *time_us = t1 - t0;

    if (err != EmlOk)
    {
        Serial.print("KNN emlearn error: ");
        Serial.println((int)err);
        return -1;
    }

    return (int)knn_out;
}


static void classify_audio()
{
    uint32_t audio_len = AUDIO_SAMPLES;

    trim_silence_feature2(audio_buffer, &audio_len);
    remove_dc_and_peak_normalize(audio_buffer, audio_len);

    Serial.print("Audio samples after trim: ");
    Serial.println((int)audio_len);

    Feature2Vector fv;

    bool ok = feature2_extract_segment(&fx, audio_buffer, audio_len, &fv);

    if (!ok)
    {
        Serial.println("ERROR: feature2_extract_segment failed");
        return;
    }

    float features[CHORD_KNN_NUM_FEATURES];

    feature_vector_to_float_array(&fv, features);
    print_features(features);

    unsigned long knn_time = 0;
    int knn_pred = predict_knn(features, &knn_time);

    Serial.println("=================================");
    Serial.println("Prediction result");

    if (knn_pred >= 0 && knn_pred < CHORD_KNN_NUM_CLASSES)
    {
        Serial.print("KNN predicted class: ");
        Serial.println(knn_pred);

        Serial.print("KNN predicted chord: ");
        Serial.println(chord_name(knn_pred));
    }
    else
    {
        Serial.print("KNN invalid prediction: ");
        Serial.println(knn_pred);
    }

    Serial.print("KNN inference time: ");
    Serial.print(knn_time);
    Serial.println(" us");

    Serial.println("=================================");
}


// =========================
// setup / loop
// =========================
void setup()
{
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.begin(115200);
    delay(3000);

    Serial.println();
    Serial.println("=================================");
    Serial.println("Photon2 live chord recognition");
    Serial.println("PDM microphone -> 18 live features -> KNN model");
    Serial.println("Features: centroid + 5 MFCC + 12 chroma_stft");
    Serial.println("chroma_cens, tonnetz and Random Forest are disabled to reduce RAM");
    Serial.println("=================================");

    Serial.print("AUDIO_SAMPLE_RATE_HZ: ");
    Serial.println(AUDIO_SAMPLE_RATE_HZ);

    Serial.print("RECORD_MS: ");
    Serial.println(RECORD_MS);

    Serial.print("AUDIO_SAMPLES: ");
    Serial.println(AUDIO_SAMPLES);

    Serial.print("CHORD_KNN_NUM_FEATURES: ");
    Serial.println(CHORD_KNN_NUM_FEATURES);

    Serial.print("CHORD_KNN_NUM_CLASSES: ");
    Serial.println(CHORD_KNN_NUM_CLASSES);

    if (!feature2_extractor_init(&fx, AUDIO_SAMPLE_RATE_HZ))
    {
        Serial.println("ERROR: feature2_extractor_init failed");
        return;
    }

    int err = Microphone_PDM::instance()
        .withOutputSize(Microphone_PDM::OutputSize::SIGNED_16)
        .withRange(Microphone_PDM::Range::RANGE_2048)
        .withSampleRate(AUDIO_SAMPLE_RATE_HZ)
        .init();

    if (err != 0)
    {
        Serial.print("ERROR: Microphone_PDM init failed: ");
        Serial.println(err);
        return;
    }

    err = Microphone_PDM::instance().start();

    if (err != 0)
    {
        Serial.print("ERROR: Microphone_PDM start failed: ");
        Serial.println(err);
        return;
    }

    Serial.println("Microphone started");
    Serial.println("Play a chord when it says Recording...");
}


void loop()
{
    if (capture_audio_block())
    {
        classify_audio();
    }

    Serial.println("Waiting before next recording...");
    delay(2500);
}