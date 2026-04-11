#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ggml/include/ggml.h"
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

int main(void) {
    const int N = 128, NKEYS = 10;
    tq3_set_layer(0);
    
    srand(0);
    float keys[10][128], query[128];
    for (int i = 0; i < NKEYS; i++)
        for (int j = 0; j < N; j++) {
            float u1 = (float)(rand()+1) / ((float)RAND_MAX+1);
            float u2 = (float)(rand()+1) / ((float)RAND_MAX+1);
            keys[i][j] = sqrtf(-2*logf(u1)) * cosf(6.2832f*u2);
        }
    for (int j = 0; j < N; j++) {
        float u1 = (float)(rand()+1) / ((float)RAND_MAX+1);
        float u2 = (float)(rand()+1) / ((float)RAND_MAX+1);
        query[j] = sqrtf(-2*logf(u1)) * cosf(6.2832f*u2);
    }
    
    // Quantize keys
    block_tq3_128 blocks[10];
    for (int i = 0; i < NKEYS; i++)
        quantize_row_tq3_128_ref(keys[i], &blocks[i], N);
    
    // Preprocess query
    block_tq3_q_128 qblock;
    from_float_tq3_q_128(query, &qblock, N);
    
    printf("%-4s  %12s  %12s  %12s  %8s\n", "Key", "f32_dot", "deq_dot", "vecdot", "vd_err%");
    
    for (int i = 0; i < NKEYS; i++) {
        // f32 reference
        float ref = 0;
        for (int j = 0; j < N; j++) ref += keys[i][j] * query[j];
        
        // Dequantize path
        float kdeq[128];
        dequantize_row_tq3_128(&blocks[i], kdeq, N);
        float deq = 0;
        for (int j = 0; j < N; j++) deq += kdeq[j] * query[j];
        
        // Vec_dot path
        float vd = 0;
        ggml_vec_dot_tq3_q_128(N, &vd, 0, &blocks[i], 0, &qblock, 0, 1);
        
        float err = (ref != 0) ? fabsf(vd - ref) / fabsf(ref) * 100 : 0;
        printf("%-4d  %12.4f  %12.4f  %12.4f  %7.1f%%\n", i, ref, deq, vd, err);
    }
    return 0;
}
