#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chord_model_info.h"
#include "chord_rf_model.h"

#define MAX_LINE 65536
#define MAX_COLS 512

static void trim(char *s)
{
    size_t n = strlen(s);

    while (n > 0 &&
           (s[n - 1] == '\n' ||
            s[n - 1] == '\r' ||
            s[n - 1] == ' '  ||
            s[n - 1] == '\t'))
    {
        s[n - 1] = '\0';
        n--;
    }
}

static int split_csv(char *line, char *tokens[], int max_tokens)
{
    int count = 0;
    char *token = strtok(line, ",");

    while (token != NULL && count < max_tokens)
    {
        trim(token);
        tokens[count++] = token;
        token = strtok(NULL, ",");
    }

    return count;
}

int main(void)
{
    FILE *fp = fopen("c_feature2_dataset.csv", "r");

    if (fp == NULL)
    {
        printf("Could not open c_feature2_dataset.csv\n");
        return 1;
    }

    char line[MAX_LINE];
    char *tokens[MAX_COLS];

    if (fgets(line, sizeof(line), fp) == NULL)
    {
        printf("CSV file is empty\n");
        fclose(fp);
        return 1;
    }

    int col_to_feature[MAX_COLS];
    for (int i = 0; i < MAX_COLS; i++)
    {
        col_to_feature[i] = -1;
    }

    int label_col = -1;

    int num_cols = split_csv(line, tokens, MAX_COLS);

    for (int col = 0; col < num_cols; col++)
    {
        if (strcmp(tokens[col], "label") == 0)
        {
            label_col = col;
        }

        for (int f = 0; f < CHORD_NUM_FEATURES; f++)
        {
            if (strcmp(tokens[col], CHORD_FEATURE_NAMES[f]) == 0)
            {
                col_to_feature[col] = f;
            }
        }
    }

    if (label_col < 0)
    {
        printf("Could not find label column\n");
        fclose(fp);
        return 1;
    }

    int rows = 0;
    int correct = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        float features[CHORD_NUM_FEATURES];

        for (int i = 0; i < CHORD_NUM_FEATURES; i++)
        {
            features[i] = 0.0f;
        }

        char *label = NULL;

        int count = split_csv(line, tokens, MAX_COLS);

        for (int col = 0; col < count; col++)
        {
            if (col == label_col)
            {
                label = tokens[col];
            }

            int feature_index = col_to_feature[col];

            if (feature_index >= 0)
            {
                features[feature_index] = (float)atof(tokens[col]);
            }
        }

        int pred = chord_rf_predict(features, CHORD_NUM_FEATURES);

        if (pred < 0 || pred >= CHORD_NUM_CLASSES)
        {
            printf("Invalid prediction: %d\n", pred);
            continue;
        }

        const char *pred_label = CHORD_LABELS[pred];

        int ok = 0;

        if (label != NULL && strcmp(label, pred_label) == 0)
        {
            ok = 1;
            correct++;
        }

        if (rows < 20)
        {
            printf("Real: %-10s  Pred: %-10s  %s\n",
                   label,
                   pred_label,
                   ok ? "OK" : "WRONG");
        }

        rows++;
    }

    fclose(fp);

    printf("\nTotal rows: %d\n", rows);
    printf("Correct: %d\n", correct);

    if (rows > 0)
    {
        printf("Accuracy: %.2f%%\n", 100.0 * correct / rows);
    }

    return 0;
}