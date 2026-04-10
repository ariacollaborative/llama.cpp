#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml/include/ggml.h"
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

int main(void) {
    const int N = 128;
    const int N_KEYS = 50;
    
    tq3_set_layer(0);
    
    // Generate Gaussian data (realistic for transformer hidden states)
    srand(0);
    float keys[50][128], query[128];
    for (int i = 0; i < N_KEYS; i++)
        for (int j = 0; j < N; j++) {
            float u1 = (float)(rand()+1) / (RAND_MAX+1.0f);
            float u2 = (float)(rand()+1) / (RAND_MAX+1.0f);
            keys[i][j] = sqrtf(-2*logf(u1)) * cosf(6.2832f*u2);
        }
    for (int j = 0; j < N; j++) {
        float u1 = (float)(rand()+1) / (RAND_MAX+1.0f);
        float u2 = (float)(rand()+1) / (RAND_MAX+1.0f);
        query[j] = sqrtf(-2*logf(u1)) * cosf(6.2832f*u2);
    }
    
    // Quantize all keys with TQ4
    block_tq4_128 blocks[50];
    for (int i = 0; i < N_KEYS; i++)
        quantize_row_tq4_128_ref(keys[i], &blocks[i], N);
    
    // Dequantize
    float keys_deq[50][128];
    for (int i = 0; i < N_KEYS; i++)
        dequantize_row_tq4_128(blocks + i, keys_deq[i], N);
    
    // Compute reference and dequantized attention scores
    float ref_scores[50], deq_scores[50];
    for (int i = 0; i < N_KEYS; i++) {
        float rs = 0, ds = 0;
        for (int j = 0; j < N; j++) {
            rs += query[j] * keys[i][j];
            ds += query[j] * keys_deq[i][j];
        }
        ref_scores[i] = rs;
        deq_scores[i] = ds;
    }
    
    // Compute correlation
    float mean_r = 0, mean_d = 0;
    for (int i = 0; i < N_KEYS; i++) { mean_r += ref_scores[i]; mean_d += deq_scores[i]; }
    mean_r /= N_KEYS; mean_d /= N_KEYS;
    
    float cov = 0, var_r = 0, var_d = 0;
    for (int i = 0; i < N_KEYS; i++) {
        float dr = ref_scores[i] - mean_r;
        float dd = deq_scores[i] - mean_d;
        cov += dr * dd;
        var_r += dr * dr;
        var_d += dd * dd;
    }
    float corr = cov / sqrtf(var_r * var_d);
    
    // Find top-5 positions
    int ref_top[5], deq_top[5];
    for (int t = 0; t < 5; t++) {
        float best_r = -1e30, best_d = -1e30;
        int bi_r = -1, bi_d = -1;
        for (int i = 0; i < N_KEYS; i++) {
            int skip = 0;
            for (int p = 0; p < t; p++) if (ref_top[p] == i) skip = 1;
            if (!skip && ref_scores[i] > best_r) { best_r = ref_scores[i]; bi_r = i; }
            skip = 0;
            for (int p = 0; p < t; p++) if (deq_top[p] == i) skip = 1;
            if (!skip && deq_scores[i] > best_d) { best_d = deq_scores[i]; bi_d = i; }
        }
        ref_top[t] = bi_r;
        deq_top[t] = bi_d;
    }
    
    // MSE
    float mse = 0, energy = 0;
    for (int i = 0; i < N_KEYS; i++)
        for (int j = 0; j < N; j++) {
            float e = keys[i][j] - keys_deq[i][j];
            mse += e*e;
            energy += keys[i][j]*keys[i][j];
        }
    
    printf("TQ4 (two-group, d=128):\n");
    printf("  Score correlation: %.4f\n", corr);
    printf("  MSE: %.4f (%.1f%%)\n", mse/energy, 100*mse/energy);
    printf("  Ref top-5:  [%d, %d, %d, %d, %d]\n", ref_top[0], ref_top[1], ref_top[2], ref_top[3], ref_top[4]);
    printf("  Deq top-5:  [%d, %d, %d, %d, %d]\n", deq_top[0], deq_top[1], deq_top[2], deq_top[3], deq_top[4]);
    printf("  Reference:  corr=0.9829 (Python TurboQuantProd uniform b=4)\n");
    
    int match = 0;
    for (int t = 0; t < 5; t++)
        for (int u = 0; u < 5; u++)
            if (ref_top[t] == deq_top[u]) match++;
    printf("  Top-5 overlap: %d/5\n", match);
    
    return (corr > 0.95) ? 0 : 1;
}

// Add vec_dot comparison after the existing test
// Appended to existing test_tq_quality.c
