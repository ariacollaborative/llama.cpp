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
void cl3_geometric_product(const float a[8], const float b[8], float r[8]) {
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
void cl3_reverse(const float x[8], float r[8]) {
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
float rq_rotors[RQ_MAX_GROUPS][8];
static int rq_initialized = 0;

void rq_init(void) {
    if (rq_initialized) return;
    for (int g = 0; g < RQ_MAX_GROUPS; g++) {
        rq_make_rotor(rq_rotors[g], RQ_SEED + g);
    }
    rq_initialized = 1;
}

/* ── Lloyd-Max codebooks for 3-bit MSE (8 centroids) ─────────────
 * Generated via scipy for the Gaussian approximation N(0, 1/d).
 * Used in RotorQuantProd: 3-bit MSE + 1-bit QJL = 4-bit total.
 */
#define RQ_MSE_LEVELS 8

const float rq3_centroids_d64[8] = {
    -0.2689932131f, -0.1679886598f, -0.0945006602f, -0.0306367724f,
     0.0306367724f,  0.0945006602f,  0.1679886598f,  0.2689932131f,
};
const float rq3_centroids_d128[8] = {
    -0.1902069251f, -0.1187859205f, -0.0668220576f, -0.0216634695f,
     0.0216634695f,  0.0668220576f,  0.1187859205f,  0.1902069251f,
};
const float rq3_centroids_d256[8] = {
    -0.1344966065f, -0.0839943299f, -0.0472503301f, -0.0153183862f,
     0.0153183862f,  0.0472503301f,  0.0839943299f,  0.1344966065f,
};

static const float * rq_get_centroids(int d) {
    if (d <= 64)  return rq3_centroids_d64;
    if (d <= 128) return rq3_centroids_d128;
    return rq3_centroids_d256;
}

static int rq_nearest_centroid(float val, const float * centroids) {
    int best = 0;
    float best_d = fabsf(val - centroids[0]);
    for (int i = 1; i < RQ_MSE_LEVELS; i++) {
        float d = fabsf(val - centroids[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

/* ── QJL S matrix (128×128 Gaussian, shared across layers) ────────*/
#define RQ_QJL_SEED 43  /* different from rotor seed */
float rq_S[RQ_MAX_D][RQ_MAX_D];
static int rq_S_initialized = 0;

void rq_init_S(int d) {
    if (rq_S_initialized) return;
    rq_prng_seed(RQ_QJL_SEED);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            rq_S[i][j] = (float)rq_prng_normal();
        }
    }
    rq_S_initialized = 1;
}

#define RQ_QJL_SCALE(d) (sqrtf(M_PI / 2.0f) / (float)(d))

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

    const int d = QK_RQ4_128;
    const int n_groups = (d + 2) / 3;
    const float * centroids = rq_get_centroids(d);
    const int nb = k / d;

    rq_init_S(d);

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

        /* 2. Pad unit vector */
        float x_padded[RQ_MAX_GROUPS * 3];
        for (int j = 0; j < d; j++) {
            x_padded[j] = src[j] * inv_norm;
        }
        for (int j = d; j < n_groups * 3; j++) {
            x_padded[j] = 0.0f;
        }

        memset(blk->qs_lo, 0, sizeof(blk->qs_lo));
        memset(blk->qs_hi, 0, sizeof(blk->qs_hi));
        memset(blk->qjl, 0, sizeof(blk->qjl));

        /* 3. Stage 1: Per-group rotor rotation + 3-bit MSE quantization */
        float x_mse[RQ_MAX_D]; /* MSE reconstruction (unit sphere) */
        memset(x_mse, 0, sizeof(float) * d);
        float recon_sq = 0.0f;
        int idx_pos = 0;

        for (int g = 0; g < n_groups; g++) {
            float mv[8] = {0};
            mv[1] = x_padded[g*3 + 0];
            mv[2] = x_padded[g*3 + 1];
            mv[3] = x_padded[g*3 + 2];

            float mv_rot[8];
            cl3_rotor_sandwich(rq_rotors[g], mv, mv_rot);

            /* Quantize + store 3-bit index as 2-bit low + 1-bit high */
            for (int c = 0; c < 3; c++) {
                int idx = rq_nearest_centroid(mv_rot[1 + c], centroids);
                /* 2-bit low */
                blk->qs_lo[idx_pos / 4] |= (uint8_t)((idx & 0x3) << ((idx_pos % 4) * 2));
                /* 1-bit high */
                if (idx & 0x4) {
                    blk->qs_hi[idx_pos / 8] |= (uint8_t)(1 << (idx_pos % 8));
                }
                recon_sq += centroids[idx] * centroids[idx];

                /* Reconstruct rotated value for residual computation */
                mv_rot[1 + c] = centroids[idx];
                idx_pos++;
            }

            /* Inverse sandwich to get MSE reconstruction in original space */
            float mv_recon[8];
            cl3_rotor_sandwich_inverse(rq_rotors[g], mv_rot, mv_recon);
            int base = g * 3;
            if (base + 0 < d) { x_mse[base + 0] = mv_recon[1]; }
            if (base + 1 < d) { x_mse[base + 1] = mv_recon[2]; }
            if (base + 2 < d) { x_mse[base + 2] = mv_recon[3]; }
        }

        /* Corrected norm for MSE component */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);

        /* 4. Stage 2: QJL on residual */
        /* Residual = original - MSE reconstruction (in original space) */
        float residual[RQ_MAX_D];
        float res_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            residual[j] = src[j] - x_mse[j] * corrected;
            res_sq += residual[j] * residual[j];
        }
        float res_norm = sqrtf(res_sq);
        blk->resnorm = GGML_FP32_TO_FP16(res_norm);

        /* QJL: sign(S × residual) */
        for (int i = 0; i < d; i++) {
            float proj = 0.0f;
            for (int j = 0; j < d; j++) {
                proj += rq_S[i][j] * residual[j];
            }
            if (proj >= 0.0f) {
                blk->qjl[i / 8] |= (uint8_t)(1 << (i % 8));
            }
        }
    }
}

/* ── Dequantize ────────────────────────────────────────────────────
 *
 * Stage 1: Unpack 3-bit indices → centroid → inverse sandwich → scale by norm
 * Stage 2: QJL correction: += (sqrt(pi/2)/d) × resnorm × S^T × signs
 */
void dequantize_row_rq4_128(const block_rq4_128 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ4_128 == 0);
    rq_init();

    const int d = QK_RQ4_128;
    const int n_groups = (d + 2) / 3;
    const float * centroids = rq_get_centroids(d);
    const int nb = k / d;
    const float qjl_scale = RQ_QJL_SCALE(d);

    rq_init_S(d);

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float resnorm = GGML_FP16_TO_FP32(x[block].resnorm);

        /* Stage 1: MSE reconstruction */
        int idx_pos = 0;
        for (int g = 0; g < n_groups; g++) {
            float mv_rot[8] = {0};
            for (int c = 0; c < 3; c++) {
                uint8_t lo = (x[block].qs_lo[idx_pos / 4] >> ((idx_pos % 4) * 2)) & 0x3;
                uint8_t hi = (x[block].qs_hi[idx_pos / 8] >> (idx_pos % 8)) & 0x1;
                uint8_t idx = lo | (hi << 2);
                mv_rot[1 + c] = centroids[idx];
                idx_pos++;
            }

            float mv_recon[8];
            cl3_rotor_sandwich_inverse(rq_rotors[g], mv_rot, mv_recon);

            int base = block * d + g * 3;
            if (g * 3 + 0 < d) { y[base + 0] = mv_recon[1] * norm; }
            if (g * 3 + 1 < d) { y[base + 1] = mv_recon[2] * norm; }
            if (g * 3 + 2 < d) { y[base + 2] = mv_recon[3] * norm; }
        }

        /* Stage 2: QJL correction */
        if (resnorm > 1e-10f) {
            float scale = qjl_scale * resnorm;
            int base = block * d;
            for (int j = 0; j < d; j++) {
                float correction = 0.0f;
                for (int i = 0; i < d; i++) {
                    float sign = (x[block].qjl[i / 8] >> (i % 8)) & 1 ? 1.0f : -1.0f;
                    correction += rq_S[i][j] * sign;  /* S^T × signs */
                }
                y[base + j] += scale * correction;
            }
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
