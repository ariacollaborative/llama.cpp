/* Debug: compare C quantization against known reference values step by step */
#include "../ggml/src/ggml-rotorquant.c"
#include <stdio.h>

int main(void) {
    rq_init();
    rq_init_S(128);

    /* Fixed test vector: x[i] = (i - 64) * 0.01 */
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = (float)(i - 64) * 0.01f;

    /* Step 1: Norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; i++) norm_sq += x[i] * x[i];
    float norm = sqrtf(norm_sq);
    printf("norm = %.10f\n", norm);

    /* Step 2: First group rotation */
    float inv = 1.0f / norm;
    float mv[8] = {0};
    mv[1] = x[0] * inv;
    mv[2] = x[1] * inv;
    mv[3] = x[2] * inv;
    printf("unit[0..2] = %.10f %.10f %.10f\n", mv[1], mv[2], mv[3]);

    float mv_rot[8];
    cl3_rotor_sandwich(rq_rotors[0], mv, mv_rot);
    printf("rot[0] = [%.10f, %.10f, %.10f] trivec=%.2e\n",
           mv_rot[1], mv_rot[2], mv_rot[3], mv_rot[7]);

    /* Step 3: Quantize first 3 values */
    const float * cb = rq_get_centroids(128);
    printf("centroids[0..3] = %.10f %.10f %.10f %.10f\n", cb[0], cb[1], cb[2], cb[3]);

    for (int c = 0; c < 3; c++) {
        int idx = rq_nearest_centroid(mv_rot[1+c], cb);
        printf("  rot[%d]=%.10f -> idx=%d centroid=%.10f\n", c, mv_rot[1+c], idx, cb[idx]);
    }

    /* Step 4: Full quantize */
    block_rq4_128 blk;
    quantize_row_rq4_128_ref(x, &blk, 128);
    printf("\nBlock: norm=%.6f resnorm=%.6f\n",
           GGML_FP16_TO_FP32(blk.norm), GGML_FP16_TO_FP32(blk.resnorm));
    printf("qs_lo[0..3] = %02x %02x %02x %02x\n", blk.qs_lo[0], blk.qs_lo[1], blk.qs_lo[2], blk.qs_lo[3]);
    printf("qs_hi[0..1] = %02x %02x\n", blk.qs_hi[0], blk.qs_hi[1]);
    printf("qjl[0..1] = %02x %02x\n", blk.qjl[0], blk.qjl[1]);

    /* Step 5: Dequantize */
    float y[128];
    dequantize_row_rq4_128(&blk, y, 128);

    /* Step 6: Compare */
    float mse = 0, dot = 0, nx = 0, ny = 0;
    for (int i = 0; i < 128; i++) {
        float d = x[i] - y[i];
        mse += d*d;
        dot += x[i]*y[i];
        nx += x[i]*x[i];
        ny += y[i]*y[i];
    }
    printf("\nRound-trip: MSE=%.4f%% cos=%.6f\n", mse/nx*100, dot/sqrtf(nx*ny));
    printf("x[0..4]  = %.6f %.6f %.6f %.6f %.6f\n", x[0], x[1], x[2], x[3], x[4]);
    printf("y[0..4]  = %.6f %.6f %.6f %.6f %.6f\n", y[0], y[1], y[2], y[3], y[4]);

    /* Step 7: Rotor values */
    printf("\nRotor[0] = [s=%.10f b12=%.10f b13=%.10f b23=%.10f]\n",
           rq_rotors[0][0], rq_rotors[0][4], rq_rotors[0][5], rq_rotors[0][6]);
    printf("S[0][0..3] = %.10f %.10f %.10f %.10f\n",
           rq_S[0][0], rq_S[0][1], rq_S[0][2], rq_S[0][3]);

    return 0;
}
