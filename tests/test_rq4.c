/* Quick standalone test for RotorQuant round-trip quality */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Include the implementation directly */
#include "../ggml/src/ggml-rotorquant.c"

int main(void) {
    /* Generate test vector: random-ish values simulating key vector */
    float x[128];
    srand(12345);
    for (int i = 0; i < 128; i++) {
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    /* No outlier — test uniform random data first */
    /* x[127] = 50.0f; */

    /* Compute original norm */
    float orig_norm = 0.0f;
    for (int i = 0; i < 128; i++) orig_norm += x[i] * x[i];
    orig_norm = sqrtf(orig_norm);

    /* Quantize */
    uint8_t block[128]; /* oversized buffer */
    memset(block, 0, sizeof(block));
    rq4_quantize_row(x, block, 128);

    /* Dequantize — need to fix the fp16 norm first */
    /* For now, manually store the corrected norm as raw float at block[0..3] */
    /* This is a hack for testing — real impl uses fp16 */

    /* Actually, let's test the rotor round-trip directly */
    printf("=== RotorQuant Round-Trip Test ===\n\n");

    rq_init();

    /* Test rotor sandwich round-trip on a single group */
    float mv[8] = {0, 0.5f, -0.3f, 0.7f, 0, 0, 0, 0};
    float mv_rot[8], mv_back[8];

    cl3_rotor_sandwich(rq_rotors[0], mv, mv_rot);
    cl3_rotor_sandwich_inverse(rq_rotors[0], mv_rot, mv_back);

    printf("Rotor round-trip (group 0):\n");
    printf("  Original:  [%.6f, %.6f, %.6f]\n", mv[1], mv[2], mv[3]);
    printf("  Rotated:   [%.6f, %.6f, %.6f]\n", mv_rot[1], mv_rot[2], mv_rot[3]);
    printf("  Recovered: [%.6f, %.6f, %.6f]\n", mv_back[1], mv_back[2], mv_back[3]);
    printf("  Trivector after sandwich: %.2e (should be ~0)\n", mv_rot[7]);

    /* Check norm preservation */
    float norm_orig = sqrtf(mv[1]*mv[1] + mv[2]*mv[2] + mv[3]*mv[3]);
    float norm_rot  = sqrtf(mv_rot[1]*mv_rot[1] + mv_rot[2]*mv_rot[2] + mv_rot[3]*mv_rot[3]);
    printf("  Norm original: %.6f, rotated: %.6f (should match)\n", norm_orig, norm_rot);

    /* Verify rotor normalization: R·R̃ should be [1,0,0,0,0,0,0,0] */
    float rotor_rev[8], rrt[8];
    cl3_reverse(rq_rotors[0], rotor_rev);
    cl3_geometric_product(rq_rotors[0], rotor_rev, rrt);
    printf("  R·R̃ = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
           rrt[0], rrt[1], rrt[2], rrt[3], rrt[4], rrt[5], rrt[6], rrt[7]);
    printf("  (should be [1, 0, 0, 0, 0, 0, 0, 0])\n");

    /* Check all 8 components after sandwich */
    printf("  Full mv_rot: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n\n",
           mv_rot[0], mv_rot[1], mv_rot[2], mv_rot[3],
           mv_rot[4], mv_rot[5], mv_rot[6], mv_rot[7]);

    /* Test full quantize-dequantize on normalized unit vector */
    printf("Full quantize-dequantize (128-dim):\n");

    float x_unit[128];
    float inv = 1.0f / orig_norm;
    for (int i = 0; i < 128; i++) x_unit[i] = x[i] * inv;

    /* Pad to 129 */
    float x_padded[RQ_N_GROUPS * 3];
    for (int i = 0; i < 128; i++) x_padded[i] = x_unit[i];
    for (int i = 128; i < RQ_N_GROUPS * 3; i++) x_padded[i] = 0.0f;

    /* Rotate + quantize + dequantize + unrotate per group */
    float x_recon[129];
    memset(x_recon, 0, sizeof(x_recon));

    float recon_sq = 0.0f;
    for (int g = 0; g < RQ_N_GROUPS; g++) {
        float fwd[8] = {0};
        fwd[1] = x_padded[g*3 + 0];
        fwd[2] = x_padded[g*3 + 1];
        fwd[3] = x_padded[g*3 + 2];

        float rot[8];
        cl3_rotor_sandwich(rq_rotors[g], fwd, rot);

        /* quantize vector components */
        float qvals[3];
        for (int c = 0; c < 3; c++) {
            int idx = rq_nearest_centroid(rot[1 + c]);
            qvals[c] = RQ_CENTROIDS_4BIT[idx];
            recon_sq += qvals[c] * qvals[c];
        }

        /* inverse sandwich on quantized values */
        float qmv[8] = {0};
        qmv[1] = qvals[0];
        qmv[2] = qvals[1];
        qmv[3] = qvals[2];

        float back[8];
        cl3_rotor_sandwich_inverse(rq_rotors[g], qmv, back);

        x_recon[g*3 + 0] = back[1];
        x_recon[g*3 + 1] = back[2];
        x_recon[g*3 + 2] = back[3];
    }

    /* Corrected norm */
    float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

    /* Compute MSE and cosine on original-scale vectors */
    float mse = 0.0f, dot = 0.0f, norm_r = 0.0f;
    for (int i = 0; i < 128; i++) {
        float r = x_recon[i] * corrected;
        float diff = x[i] - r;
        mse += diff * diff;
        dot += x[i] * r;
        norm_r += r * r;
    }
    mse /= 128;
    float cosine = dot / (orig_norm * sqrtf(norm_r));

    printf("  Original norm: %.4f\n", orig_norm);
    printf("  Corrected norm: %.4f\n", corrected);
    printf("  MSE: %.6f (%.2f%%)\n", mse, mse / (orig_norm * orig_norm / 128) * 100);
    printf("  Cosine similarity: %.6f\n", cosine);
    printf("  First 5 orig:  [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           x[0], x[1], x[2], x[3], x[4]);
    printf("  First 5 recon: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           x_recon[0]*corrected, x_recon[1]*corrected, x_recon[2]*corrected,
           x_recon[3]*corrected, x_recon[4]*corrected);
    printf("  ch127 orig: %.4f, recon: %.4f\n", x[127], x_recon[127]*corrected);

    return 0;
}
