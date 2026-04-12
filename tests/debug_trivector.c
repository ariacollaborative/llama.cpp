/* Measure trivector energy leakage across all 43 groups */
#include "../ggml/src/ggml-rotorquant.c"
#include <stdio.h>

int main(void) {
    rq_init();

    /* Test with the same vector as the main tests */
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = (float)(i - 64) * 0.01f;

    float norm_sq = 0.0f;
    for (int i = 0; i < 128; i++) norm_sq += x[i] * x[i];
    float norm = sqrtf(norm_sq);
    float inv = 1.0f / norm;

    float total_vec_energy = 0.0f;
    float total_tri_energy = 0.0f;

    printf("=== Trivector Energy Leakage Per Group ===\n\n");
    printf("Group | Vec Energy | Tri Energy | Leakage%%\n");
    printf("------+------------+------------+---------\n");

    for (int g = 0; g < 43; g++) {
        float mv[8] = {0};
        int base = g * 3;
        mv[1] = (base+0 < 128) ? x[base+0] * inv : 0.0f;
        mv[2] = (base+1 < 128) ? x[base+1] * inv : 0.0f;
        mv[3] = (base+2 < 128) ? x[base+2] * inv : 0.0f;

        float input_energy = mv[1]*mv[1] + mv[2]*mv[2] + mv[3]*mv[3];

        float mv_rot[8];
        cl3_rotor_sandwich(rq_rotors[g], mv, mv_rot);

        float vec_energy = mv_rot[1]*mv_rot[1] + mv_rot[2]*mv_rot[2] + mv_rot[3]*mv_rot[3];
        float tri_energy = mv_rot[7]*mv_rot[7];
        float other_energy = mv_rot[0]*mv_rot[0] + mv_rot[4]*mv_rot[4] + mv_rot[5]*mv_rot[5] + mv_rot[6]*mv_rot[6];

        total_vec_energy += vec_energy;
        total_tri_energy += tri_energy;

        float leakage = (input_energy > 1e-10f) ? tri_energy / input_energy * 100.0f : 0.0f;
        printf("  %2d  |  %.6f  |  %.6f  |  %5.2f%%\n", g, vec_energy, tri_energy, leakage);
    }

    printf("\n");
    printf("Total vector energy:    %.6f (should be ~1.0)\n", total_vec_energy);
    printf("Total trivector energy: %.6f\n", total_tri_energy);
    printf("Total leakage: %.2f%%\n", total_tri_energy / (total_vec_energy + total_tri_energy) * 100.0f);
    printf("\nFor comparison, q4_0 has ZERO leakage (no rotation).\n");
    printf("This leakage directly reduces the effective cosine similarity.\n");
    printf("Predicted cosine impact: ~%.6f\n", 1.0f - total_tri_energy);

    /* Now compare: quantize with rq4, dequant, measure cosine */
    /* vs just the leakage prediction */
    printf("\n=== Actual Round-Trip Quality ===\n");

    /* First without QJL - use the 4-bit codebook manually */
    extern const float rq3_centroids_d128[8];
    const float * cb = rq3_centroids_d128;

    float x_recon[128];
    memset(x_recon, 0, sizeof(x_recon));
    float recon_sq = 0.0f;

    for (int g = 0; g < 43; g++) {
        float mv[8] = {0};
        int base = g * 3;
        mv[1] = (base+0 < 128) ? x[base+0] * inv : 0.0f;
        mv[2] = (base+1 < 128) ? x[base+1] * inv : 0.0f;
        mv[3] = (base+2 < 128) ? x[base+2] * inv : 0.0f;

        float mv_rot[8];
        cl3_rotor_sandwich(rq_rotors[g], mv, mv_rot);

        /* Quantize vector components only */
        for (int c = 0; c < 3; c++) {
            int idx = rq_nearest_centroid(mv_rot[1+c], cb);
            mv_rot[1+c] = cb[idx];
            recon_sq += cb[idx] * cb[idx];
        }

        /* Inverse sandwich on quantized values */
        float mv_back[8];
        cl3_rotor_sandwich_inverse(rq_rotors[g], mv_rot, mv_back);

        if (base+0 < 128) x_recon[base+0] = mv_back[1];
        if (base+1 < 128) x_recon[base+1] = mv_back[2];
        if (base+2 < 128) x_recon[base+2] = mv_back[3];
    }

    float corrected = norm / sqrtf(recon_sq);

    float mse = 0, dot = 0, nx = 0, ny = 0;
    for (int i = 0; i < 128; i++) {
        float r = x_recon[i] * corrected;
        float d = x[i] - r;
        mse += d*d;
        dot += x[i] * r;
        nx += x[i]*x[i];
        ny += r*r;
    }
    printf("3-bit MSE-only: MSE=%.4f%% cos=%.6f\n", mse/nx*100, dot/sqrtf(nx*ny));

    return 0;
}
