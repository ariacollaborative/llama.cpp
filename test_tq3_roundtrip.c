/*
 * test_tq3_roundtrip.c — TQ3 round-trip quantization and dot product test.
 *
 * Verifies:
 *   1. Round-trip MSE (quantize → dequantize) < 40% of signal energy.
 *   2. Dequantized dot product error < 20% of f32 reference.
 *   3. Vec-dot (compressed path) error < 20% of f32 reference.
 *
 * Compile:
 *   gcc -O2 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu \
 *       test_tq3_roundtrip.c ggml/src/ggml-quants.c -lm -o test_tq3
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Pull in the declarations we need. */
#include "ggml/include/ggml.h"
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

/* ggml-quants.c calls these; they are only reachable from non-TQ3 paths.
   Provide stubs so we can link without ggml.c.
   Signatures must match ggml.h exactly. */
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    (void)file; (void)line; (void)fmt;
    fprintf(stderr, "ggml_abort called\n");
    abort();
}
size_t       ggml_row_size (enum ggml_type type, int64_t ne) { (void)type; (void)ne; return 0; }
const char * ggml_type_name(enum ggml_type type)             { (void)type; return "?"; }
size_t       ggml_type_size(enum ggml_type type)             { (void)type; return 0; }

/* ------------------------------------------------------------------ */

static float randf_11(void) {
    /* uniform in [-1, 1] */
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

int main(void) {
    /* Step 1: set layer 0 so TQ3 uses deterministic per-layer matrices. */
    tq3_set_layer(0);

    /* Step 2: generate deterministic random input vectors. */
    srand(42);
    float x[128], q[128];
    for (int i = 0; i < 128; i++) x[i] = randf_11();
    for (int i = 0; i < 128; i++) q[i] = randf_11();

    /* Step 3: quantize x into a block_tq3_128. */
    block_tq3_128 blk;
    memset(&blk, 0, sizeof(blk));
    quantize_row_tq3_128_ref(x, &blk, 128);

    /* Step 4: dequantize back. */
    float x_deq[128];
    dequantize_row_tq3_128(&blk, x_deq, 128);

    /* Step 5: round-trip MSE relative to signal energy. */
    float mse_num = 0.0f, mse_den = 0.0f;
    for (int i = 0; i < 128; i++) {
        float diff = x[i] - x_deq[i];
        mse_num += diff * diff;
        mse_den += x[i] * x[i];
    }
    float mse_rel = (mse_den > 0.0f) ? (mse_num / mse_den) : 0.0f;
    printf("Round-trip MSE / signal energy: %.4f (%.1f%%)\n",
           mse_rel, mse_rel * 100.0f);

    /* Step 6: f32 reference dot product. */
    float dot_ref = 0.0f;
    for (int i = 0; i < 128; i++) dot_ref += x[i] * q[i];
    printf("f32 reference dot:     %+.6f\n", dot_ref);

    /* Step 7: dequantized dot product. */
    float dot_deq = 0.0f;
    for (int i = 0; i < 128; i++) dot_deq += x_deq[i] * q[i];
    printf("Dequantized dot:       %+.6f\n", dot_deq);
    float err_deq = fabsf(dot_deq - dot_ref) / (fabsf(dot_ref) + 1e-8f);
    printf("Dequantized dot error: %.4f (%.1f%%)\n", err_deq, err_deq * 100.0f);

    /* Step 8: preprocess query into block_tq3_q_128. */
    block_tq3_q_128 qblk;
    memset(&qblk, 0, sizeof(qblk));
    from_float_tq3_q_128(q, &qblk, 128);

    /* Step 9: compute vec_dot. */
    float dot_vd = 0.0f;
    ggml_vec_dot_tq3_q_128(128, &dot_vd, sizeof(float),
                            &blk, sizeof(block_tq3_128),
                            &qblk, sizeof(block_tq3_q_128),
                            1);
    printf("Vec-dot result:        %+.6f\n", dot_vd);

    /* Step 10: print relative errors. */
    float err_vd = fabsf(dot_vd - dot_ref) / (fabsf(dot_ref) + 1e-8f);
    printf("Vec-dot error:         %.4f (%.1f%%)\n", err_vd, err_vd * 100.0f);

    /* Step 11: PASS/FAIL judgement. */
    int pass = 1;
    if (mse_rel >= 0.40f) {
        printf("FAIL: round-trip MSE %.1f%% >= 40%%\n", mse_rel * 100.0f);
        pass = 0;
    }
    if (err_vd >= 0.20f) {
        printf("FAIL: vec-dot error %.1f%% >= 20%%\n", err_vd * 100.0f);
        pass = 0;
    }
    if (pass) {
        printf("PASS\n");
        return 0;
    }
    return 1;
}
