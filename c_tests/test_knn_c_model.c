#include <stdio.h>
#include <stdint.h>

#include "chord_knn_model_info.h"
#include "chord_knn_model.h"
#include "knn_test_vectors.h"

#define KNN_DISTANCE_BUFFER_SIZE 512

static EmlNeighborsDistanceItem knn_distances[KNN_DISTANCE_BUFFER_SIZE];

static const char *label_name(int label)
{
    if (label >= 0 && label < CHORD_KNN_NUM_CLASSES)
    {
        return CHORD_KNN_LABELS[label];
    }

    return "INVALID";
}

int main(void)
{
    int correct_vs_true = 0;
    int match_vs_python = 0;
    int errors = 0;

    printf("===== KNN C MODEL TEST =====\n");
    printf("Samples: %d\n", KNN_TEST_NUM_SAMPLES);
    printf("Features: %d\n", CHORD_KNN_NUM_FEATURES);
    printf("Classes: %d\n\n", CHORD_KNN_NUM_CLASSES);

    for (int i = 0; i < KNN_TEST_NUM_SAMPLES; i++)
    {
        int16_t input[CHORD_KNN_NUM_FEATURES];
        int16_t c_pred = -1;

        chord_knn_prepare_features(KNN_TEST_X_RAW[i], input);

        EmlError err = eml_neighbors_predict(
            &chord_knn,
            input,
            CHORD_KNN_NUM_FEATURES,
            knn_distances,
            KNN_DISTANCE_BUFFER_SIZE,
            &c_pred
        );

        if (err != EmlOk)
        {
            printf("Sample %d: emlearn error = %d\n", i, (int)err);
            errors++;
            continue;
        }

        if (c_pred == KNN_TEST_TRUE[i])
        {
            correct_vs_true++;
        }

        if (c_pred == KNN_TEST_PYTHON[i])
        {
            match_vs_python++;
        }
        else
        {
            printf(
                "Mismatch sample %d: true=%s, python=%s, c=%s\n",
                i,
                label_name(KNN_TEST_TRUE[i]),
                label_name(KNN_TEST_PYTHON[i]),
                label_name(c_pred)
            );
        }
    }

    printf("\n===== RESULT =====\n");
    printf("Errors: %d/%d\n", errors, KNN_TEST_NUM_SAMPLES);
    printf("C accuracy vs true labels: %.2f %%\n",
           100.0 * (double)correct_vs_true / (double)KNN_TEST_NUM_SAMPLES);
    printf("C match vs Python predictions: %.2f %%\n",
           100.0 * (double)match_vs_python / (double)KNN_TEST_NUM_SAMPLES);

    return 0;
}