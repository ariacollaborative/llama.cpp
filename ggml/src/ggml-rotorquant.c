/*
 * RotorQuant: Clifford algebra Cl(3,0) KV cache quantization
 * Paper: Pope, "RotorQuant: Clifford Algebra Vector Quantization for LLM KV Cache Compression", 2026
 *
 * Groups of 3 dimensions rotated by Cl(3,0) rotors via sandwich product R·v·R̃.
 * 4 non-zero rotor components per group, ~56 FMAs per sandwich.
 * 43 groups for d=128 (last group zero-padded).
 *
 * Multivector basis: [1, e1, e2, e3, e12, e13, e23, e123]
 *                     0    1   2   3   4    5    6     7
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#define RQ_MAX_D    256
#define RQ_MAX_GROUPS ((RQ_MAX_D + 2) / 3)  /* ceil(256/3) = 86 */
#define RQ_SEED    42

/* ── Cl(3,0) geometric product ──────────────────────────────────────
 *
 * Full product a·b in Cl(3,0) with signature (+,+,+).
 * a, b: 8-component multivectors [s, e1, e2, e3, e12, e13, e23, e123]
 * r:    8-component result
 *
 * 64 multiplies, 56 adds = 64 FMAs.
 */
static void cl3_geometric_product(const float a[8], const float b[8], float r[8]) {
    const float a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const float a12 = a[4], a13 = a[5], a23 = a[6], a123 = a[7];
    const float b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    const float b12 = b[4], b13 = b[5], b23 = b[6], b123 = b[7];

    /* grade 0 (scalar) */
    r[0] = a0*b0 + a1*b1 + a2*b2 + a3*b3
         - a12*b12 - a13*b13 - a23*b23 - a123*b123;

    /* grade 1 (vector) */
    r[1] = a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3
         + a23*b123 + a123*b23;
    r[2] = a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3
         - a13*b123 - a123*b13;
    r[3] = a0*b3 + a3*b0 + a1*b13 - a13*b1 + a2*b23 - a23*b2
         + a12*b123 + a123*b12;

    /* grade 2 (bivector) */
    r[4] = a0*b12 + a12*b0 + a1*b2 - a2*b1 + a13*b23 - a23*b13
         + a3*b123 - a123*b3;
    r[5] = a0*b13 + a13*b0 + a1*b3 - a3*b1 - a12*b23 + a23*b12
         - a2*b123 + a123*b2;
    r[6] = a0*b23 + a23*b0 + a2*b3 - a3*b2 + a12*b13 - a13*b12
         + a1*b123 - a123*b1;

    /* grade 3 (pseudoscalar) */
    r[7] = a0*b123 + a123*b0 + a1*b23 - a23*b1 - a2*b13 + a13*b2
         + a3*b12 - a12*b3;
}

/* ── Clifford reverse ───────────────────────────────────────────────
 *
 * Reverse x̃: grades 0,1 unchanged, grades 2,3 negated.
 * For rotor R, R̃ is used in the sandwich product R·v·R̃.
 */
static void cl3_reverse(const float x[8], float r[8]) {
    r[0] =  x[0]; r[1] =  x[1]; r[2] =  x[2]; r[3] =  x[3];
    r[4] = -x[4]; r[5] = -x[5]; r[6] = -x[6]; r[7] = -x[7];
}

/* ── Rotor sandwich R·v·R̃ ──────────────────────────────────────────
 *
 * Rotates multivector v by rotor R while preserving algebraic structure.
 * Two geometric products: temp = R·v, then result = temp·R̃.
 */
static void cl3_rotor_sandwich(const float rotor[8], const float v[8], float result[8]) {
    float rotor_rev[8];
    float temp[8];
    cl3_reverse(rotor, rotor_rev);
    cl3_geometric_product(rotor, v, temp);
    cl3_geometric_product(temp, rotor_rev, result);
}

/* ── Inverse sandwich R̃·v·R ────────────────────────────────────────*/
static void cl3_rotor_sandwich_inverse(const float rotor[8], const float v[8], float result[8]) {
    float rotor_rev[8];
    float temp[8];
    cl3_reverse(rotor, rotor_rev);
    cl3_geometric_product(rotor_rev, v, temp);
    cl3_geometric_product(temp, rotor, result);
}

/* ── PRNG (same as reference: LCG + Box-Muller) ────────────────────*/
static uint64_t rq_prng_state;

static void rq_prng_seed(uint64_t seed) {
    rq_prng_state = seed;
}

static double rq_prng_uniform(void) {
    rq_prng_state = rq_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rq_prng_state >> 11) / (double)(1ULL << 53);
}

static double rq_prng_normal(void) {
    double u1 = rq_prng_uniform();
    if (u1 < 1e-15) u1 = 1e-15;
    double u2 = rq_prng_uniform();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ── Rotor generation ──────────────────────────────────────────────
 *
 * Generate a rotor from random bivector + random angle:
 *   R = cos(θ/2) + sin(θ/2) · B̂
 * where B̂ = (b12·e12 + b13·e13 + b23·e23) / ||b||
 * Normalized so R·R̃ = 1.
 */
static void rq_make_rotor(float rotor[8], int seed) {
    rq_prng_seed(seed);

    /* random bivector direction */
    float bv[3];
    bv[0] = (float)rq_prng_normal();
    bv[1] = (float)rq_prng_normal();
    bv[2] = (float)rq_prng_normal();

    float bv_norm = sqrtf(bv[0]*bv[0] + bv[1]*bv[1] + bv[2]*bv[2]);
    if (bv_norm < 1e-8f) bv_norm = 1e-8f;
    bv[0] /= bv_norm;
    bv[1] /= bv_norm;
    bv[2] /= bv_norm;

    /* random angle */
    float angle = (float)(rq_prng_uniform() * 2.0 * M_PI);
    float half = angle * 0.5f;
    float c = cosf(half);
    float s = sinf(half);

    /* R = cos(θ/2) + sin(θ/2)·B̂ */
    memset(rotor, 0, 8 * sizeof(float));
    rotor[0] = c;        /* scalar */
    rotor[4] = s * bv[0]; /* e12 */
    rotor[5] = s * bv[1]; /* e13 */
    rotor[6] = s * bv[2]; /* e23 */

    /* normalize: R / sqrt(R·R̃) */
    float norm_sq = rotor[0]*rotor[0] + rotor[4]*rotor[4]
                  + rotor[5]*rotor[5] + rotor[6]*rotor[6];
    float inv = 1.0f / sqrtf(norm_sq);
    rotor[0] *= inv;
    rotor[4] *= inv;
    rotor[5] *= inv;
    rotor[6] *= inv;
}

/* ── Static rotor storage (max 86 rotors for d=256, generated once) ─*/
static float rq_rotors[RQ_MAX_GROUPS][8];
static int rq_initialized = 0;

static void rq_init(void) {
    if (rq_initialized) return;
    for (int g = 0; g < RQ_MAX_GROUPS; g++) {
        rq_make_rotor(rq_rotors[g], RQ_SEED + g);
    }
    rq_initialized = 1;
}

/* ── Lloyd-Max codebooks for 4-bit (16 centroids) ──────────────────
 * Generated via scipy for the Gaussian approximation N(0, 1/d).
 */
#define RQ_N_LEVELS 16

static const float rq_centroids_d64[16] = {
    -0.3416347612f, -0.2586961384f, -0.2023261788f, -0.1570952221f,
    -0.1178499334f, -0.0821391425f, -0.0485339480f, -0.0160589141f,
     0.0160589141f,  0.0485339480f,  0.0821391425f,  0.1178499334f,
     0.1570952221f,  0.2023261788f,  0.2586961384f,  0.3416347612f,
};
static const float rq_centroids_d128[16] = {
    -0.2415722564f, -0.1829257937f, -0.1430662130f, -0.1110830968f,
    -0.0833324871f, -0.0580811446f, -0.0343186837f, -0.0113553670f,
     0.0113553670f,  0.0343186837f,  0.0580811446f,  0.0833324871f,
     0.1110830968f,  0.1430662130f,  0.1829257937f,  0.2415722564f,
};
static const float rq_centroids_d256[16] = {
    -0.1708173806f, -0.1293480692f, -0.1011630894f, -0.0785476111f,
    -0.0589249667f, -0.0410695712f, -0.0242669740f, -0.0080294570f,
     0.0080294570f,  0.0242669740f,  0.0410695712f,  0.0589249667f,
     0.0785476111f,  0.1011630894f,  0.1293480692f,  0.1708173806f,
};

static const float * rq_get_centroids(int d) {
    if (d <= 64)  return rq_centroids_d64;
    if (d <= 128) return rq_centroids_d128;
    return rq_centroids_d256;
}

static int rq_nearest_centroid(float val, const float * centroids) {
    int best = 0;
    float best_d = fabsf(val - centroids[0]);
    for (int i = 1; i < RQ_N_LEVELS; i++) {
        float d = fabsf(val - centroids[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

/* ── Quantize ──────────────────────────────────────────────────────
 *
 * x:   128 float input values
 * out: packed output (norm fp16 + 4-bit indices)
 *
 * Algorithm:
 *   1. L2 norm, normalize to unit sphere
 *   2. Pad to 129 elements (43 groups × 3)
 *   3. Per-group: embed as grade-1 multivector → rotor sandwich → quantize vector components
 *   4. Corrected norm = original_norm / quantized_unit_norm
 */
void quantize_row_rq4_128_ref(const float * GGML_RESTRICT x, block_rq4_128 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ4_128 == 0);
    rq_init();

    const int d = QK_RQ4_128;  /* head dimension from block size */
    const int n_groups = (d + 2) / 3;
    const float * centroids = rq_get_centroids(d);
    const int nb = k / d;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;
        block_rq4_128 * blk = &y[block];

        /* 1. L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            norm_sq += src[j] * src[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Pad to multiple of 3 */
        float x_padded[RQ_MAX_GROUPS * 3];
        for (int j = 0; j < d; j++) {
            x_padded[j] = src[j] * inv_norm;
        }
        for (int j = d; j < n_groups * 3; j++) {
            x_padded[j] = 0.0f;
        }

        memset(blk->qs, 0, sizeof(blk->qs));
        float recon_sq = 0.0f;
        int idx_pos = 0;

        /* 3. Per-group rotation + quantization */
        for (int g = 0; g < n_groups; g++) {
            float mv[8] = {0};
            mv[1] = x_padded[g*3 + 0];
            mv[2] = x_padded[g*3 + 1];
            mv[3] = x_padded[g*3 + 2];

            float mv_rot[8];
            cl3_rotor_sandwich(rq_rotors[g], mv, mv_rot);

            for (int c = 0; c < 3; c++) {
                int idx = rq_nearest_centroid(mv_rot[1 + c], centroids);
                if (idx_pos % 2 == 0) {
                    blk->qs[idx_pos / 2] = (uint8_t)(idx & 0x0F);
                } else {
                    blk->qs[idx_pos / 2] |= (uint8_t)((idx & 0x0F) << 4);
                }
                idx_pos++;
                recon_sq += centroids[idx] * centroids[idx];
            }
        }

        /* 4. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
    }
}

/* ── Dequantize ────────────────────────────────────────────────────
 *
 * Unpack indices → centroid lookup → inverse rotor sandwich → scale by norm
 */
void dequantize_row_rq4_128(const block_rq4_128 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ4_128 == 0);
    rq_init();

    const int d = QK_RQ4_128;
    const int n_groups = (d + 2) / 3;
    const float * centroids = rq_get_centroids(d);
    const int nb = k / d;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        int idx_pos = 0;

        for (int g = 0; g < n_groups; g++) {
            float mv_rot[8] = {0};
            for (int c = 0; c < 3; c++) {
                uint8_t packed;
                if (idx_pos % 2 == 0) {
                    packed = x[block].qs[idx_pos / 2] & 0x0F;
                } else {
                    packed = (x[block].qs[idx_pos / 2] >> 4) & 0x0F;
                }
                idx_pos++;
                mv_rot[1 + c] = centroids[packed];
            }

            float mv_recon[8];
            cl3_rotor_sandwich_inverse(rq_rotors[g], mv_rot, mv_recon);

            int base = block * d + g * 3;
            if (g * 3 + 0 < d) { y[base + 0] = mv_recon[1] * norm; }
            if (g * 3 + 1 < d) { y[base + 1] = mv_recon[2] * norm; }
            if (g * 3 + 2 < d) { y[base + 2] = mv_recon[3] * norm; }
        }
    }
}

size_t quantize_rq4_128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_RQ4_128 == 0);
    size_t row_size = (n_per_row / QK_RQ4_128) * sizeof(block_rq4_128);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_rq4_128_ref(
            src + row * n_per_row,
            (block_rq4_128 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
