/*
 * RotorQuant 4-bit SET_ROWS kernel: quantize f32 → rq4 blocks on GPU
 * Clifford Cl(3,0) rotor sandwich + 4-bit Lloyd-Max
 */

#define RQ_D       128
#define RQ_N_GROUPS 43
#define RQ_N_LEVELS 16
#define RQ_BLOCK_BYTES 68

/* Lloyd-Max centroids for d=128, 4-bit */
constant float RQ_CENTROIDS[16] = {
    -0.2415722564f, -0.1829257937f, -0.1430662130f, -0.1110830968f,
    -0.0833324871f, -0.0580811446f, -0.0343186837f, -0.0113553670f,
     0.0113553670f,  0.0343186837f,  0.0580811446f,  0.0833324871f,
     0.1110830968f,  0.1430662130f,  0.1829257937f,  0.2415722564f,
};

/* Rotor parameters: 43 groups × 4 components [s, b12, b13, b23]
 * Pre-computed from seed 42, same as CPU implementation.
 * These will be uploaded as a buffer from the host.
 */

/* Cl(3,0) geometric product for rotor × multivector
 * Only computes grade-1 output (indices 1,2,3) since that's all we need.
 * Rotor has components [s, 0, 0, 0, b12, b13, b23, 0] (indices 0,4,5,6)
 * Input multivector has components at indices 1,2,3 (grade-1 vector)
 *
 * For the FIRST product R·v (v is pure grade-1):
 *   temp[1] = s*v1 + b12*v2 + b13*v3
 *   temp[2] = s*v2 - b12*v1 + b23*v3
 *   temp[3] = s*v3 - b13*v1 - b23*v2
 *   temp[7] = -b23*v1 + b13*v2 - b12*v3  (trivector)
 *
 * For the SECOND product temp·R̃ (R̃ = [s,0,0,0,-b12,-b13,-b23,0]):
 *   Need full product using temp[1,2,3,7] × R̃[0,4,5,6]
 */
void cl3_sandwich_vec(float rs, float rb12, float rb13, float rb23,
                      float v1, float v2, float v3,
                      float *out1, float *out2, float *out3) {
    /* First product: R · v */
    float t1 = rs*v1 + rb12*v2 + rb13*v3;
    float t2 = rs*v2 - rb12*v1 + rb23*v3;
    float t3 = rs*v3 - rb13*v1 - rb23*v2;
    float t7 = -rb23*v1 + rb13*v2 - rb12*v3;

    /* Second product: temp · R̃  (negate bivector components of R) */
    float nb12 = -rb12, nb13 = -rb13, nb23 = -rb23;

    /* Only need grade-1 output */
    *out1 = rs*t1 - t2*nb12 - t3*nb13 + t7*nb23;  /* e1 from: s×e1, e2×e12, e3×e13, e123×e23 */
    *out2 = rs*t2 + t1*nb12 - t3*nb23 - t7*nb13;  /* e2 from: s×e2, e1×e12, e3×e23, e123×e13 */
    *out3 = rs*t3 + t1*nb13 + t2*nb23 + t7*nb12;  /* e3 from: s×e3, e1×e13, e2×e23, e123×e12 */
}

int nearest_centroid(float val) {
    int best = 0;
    float best_d = fabs(val - RQ_CENTROIDS[0]);
    for (int i = 1; i < RQ_N_LEVELS; i++) {
        float d = fabs(val - RQ_CENTROIDS[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

kernel void kernel_set_rows_rq4(
        global const float * src,
        global const int   * dst_row_idxs,
        global       uchar * dst,
        int ne00,        /* src elements per row (should be 128) */
        long nb1,        /* dst stride between rows (bytes) */
        long nb2,        /* dst stride for dim 2 */
        long nb3,        /* dst stride for dim 3 */
        int ne10,        /* number of rows */
        int ne11,
        int ne12,
        global const float * rotors  /* 43 × 4 floats: [s, b12, b13, b23] per group */
) {
    int ir = get_global_id(0);  /* row index */
    if (ir >= ne10) return;

    int i02 = 0;  /* TODO: batch dims */
    int i03 = 0;
    int dst_row_idx = dst_row_idxs[ir];

    global const float * src_row = src + ir * ne00;
    global uchar * dst_row = dst + dst_row_idx * nb1 + i02 * nb2 + i03 * nb3;

    /* 1. L2 norm */
    float norm_sq = 0.0f;
    for (int j = 0; j < RQ_D; j++) {
        norm_sq += src_row[j] * src_row[j];
    }
    float grp_norm = sqrt(norm_sq);
    float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    /* 2. Normalize */
    float x_unit[129];  /* private — fits in registers for d=128+1 */
    for (int j = 0; j < RQ_D; j++) x_unit[j] = src_row[j] * inv_norm;
    x_unit[128] = 0.0f;  /* padding for group 42 */

    /* Clear output indices */
    for (int j = 2; j < RQ_BLOCK_BYTES; j++) dst_row[j] = 0;

    float recon_sq = 0.0f;
    int idx_pos = 0;

    /* 3. Per-group rotation + quantization */
    for (int g = 0; g < RQ_N_GROUPS; g++) {
        float rs   = rotors[g * 4 + 0];
        float rb12 = rotors[g * 4 + 1];
        float rb13 = rotors[g * 4 + 2];
        float rb23 = rotors[g * 4 + 3];

        float v1 = x_unit[g*3 + 0];
        float v2 = x_unit[g*3 + 1];
        float v3 = x_unit[g*3 + 2];

        float r1, r2, r3;
        cl3_sandwich_vec(rs, rb12, rb13, rb23, v1, v2, v3, &r1, &r2, &r3);

        /* Quantize 3 rotated components */
        float rotated[3] = {r1, r2, r3};
        for (int c = 0; c < 3; c++) {
            int idx = nearest_centroid(rotated[c]);
            int pos = idx_pos + c;
            int byte_idx = 2 + pos / 2;  /* skip 2 bytes norm */
            if (pos % 2 == 0) {
                dst_row[byte_idx] = (uchar)(idx & 0x0F);
            } else {
                dst_row[byte_idx] |= (uchar)((idx & 0x0F) << 4);
            }
            recon_sq += RQ_CENTROIDS[idx] * RQ_CENTROIDS[idx];
        }
        idx_pos += 3;
    }

    /* 4. Corrected norm as fp16 */
    float recon_norm = sqrt(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    vstore_half(corrected, 0, (global half *)dst_row);
}
