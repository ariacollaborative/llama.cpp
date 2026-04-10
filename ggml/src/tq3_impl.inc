// tq3_impl.inc — TurboQuant mixed-precision KV cache quantization
//
// Included TWICE from ggml-quants.c:
//   First  include: TQ_OUT_BITS=3, TQ_REG_BITS=1  (tq3_128, 2.5-bit effective)
//   Second include: TQ_OUT_BITS=4, TQ_REG_BITS=2  (tq4_128, 3.5-bit effective)
//
// Each 128-element block is split into:
//   Outlier group  — 32 channels (TQ_N_OUTLIER), TQ_OUT_BITS MSE + 1-bit QJL
//   Regular group  — 96 channels (TQ_N_REGULAR), TQ_REG_BITS MSE + 1-bit QJL
//
// Algorithm per group:
//   Quantize: normalize → rotate (Pi) → scalar VQ → inverse-rotate → residual
//             → sign(S × residual) stored as QJL bits
//   Dequantize: centroids → inverse-rotate → scale by norm
//               → add (sqrt(pi/2)/d) * resnorm * S^T × sign_vec
//   Vec-dot:  MSE term in rotated domain + QJL term without reconstructing float

// ---------------------------------------------------------------------------
// Shared helpers — compiled only on the FIRST include.
// ---------------------------------------------------------------------------
#ifndef TQ3_IMPL_HELPERS_DEFINED
#define TQ3_IMPL_HELPERS_DEFINED

// sqrt(pi/2) for QJL amplitude correction.
#define TQ_SQRT_PI_2  0.886226925452758f

// Matrix–vector multiply: out[d1] = M[d1×d2] × v[d2]
static void tq_matvec(const float * M, const float * v, float * out, int d1, int d2) {
    for (int i = 0; i < d1; i++) {
        float acc = 0.0f;
        for (int j = 0; j < d2; j++) acc += M[i*d2+j] * v[j];
        out[i] = acc;
    }
}

// Transpose matrix–vector multiply: out[d2] = M[d1×d2]^T × v[d1]
static void tq_matvec_t(const float * M, const float * v, float * out, int d1, int d2) {
    for (int j = 0; j < d2; j++) {
        float acc = 0.0f;
        for (int i = 0; i < d1; i++) acc += M[i*d2+j] * v[i];
        out[j] = acc;
    }
}

// Tight 3-bit packing: 32 indices (0-7) → 12 bytes (8 per 3-byte group).
//   byte0 = i0[2:0] | i1[2:0]<<3 | i2[1:0]<<6
//   byte1 = i2[2]   | i3[2:0]<<1 | i4[2:0]<<4 | i5[0]<<7
//   byte2 = i5[2:1] | i6[2:0]<<2 | i7[2:0]<<5
static void tq_pack_3bit(const uint8_t * idx, uint8_t * packed) {
    for (int g = 0; g < 4; g++) {
        const uint8_t * in  = idx    + g*8;
        uint8_t       * out = packed + g*3;
        out[0] = (in[0] & 7) | ((in[1] & 7) << 3) | ((in[2] & 3) << 6);
        out[1] = ((in[2] >> 2) & 1) | ((in[3] & 7) << 1) | ((in[4] & 7) << 4) | ((in[5] & 1) << 7);
        out[2] = ((in[5] >> 1) & 3) | ((in[6] & 7) << 2) | ((in[7] & 7) << 5);
    }
}

static void tq_unpack_3bit(const uint8_t * packed, uint8_t * idx) {
    for (int g = 0; g < 4; g++) {
        const uint8_t * in  = packed + g*3;
        uint8_t       * out = idx    + g*8;
        out[0] =  in[0]        & 7;
        out[1] = (in[0] >> 3)  & 7;
        out[2] = ((in[0] >> 6) & 3) | ((in[1] & 1) << 2);
        out[3] = (in[1] >> 1)  & 7;
        out[4] = (in[1] >> 4)  & 7;
        out[5] = ((in[1] >> 7) & 1) | ((in[2] & 3) << 1);
        out[6] = (in[2] >> 2)  & 7;
        out[7] = (in[2] >> 5)  & 7;
    }
}

// Codebook search: index of nearest centroid given (nclusters-1) ascending
// decision boundaries.
static inline int tq_find_idx(float val, const float * bd, int nclusters) {
    int best = 0;
    for (int b = 0; b < nclusters - 1; b++) {
        if (val > bd[b]) best = b + 1;
    }
    return best;
}

// Build outlier membership mask: is_out[c] = 1 iff channel c is an outlier.
static void tq_build_outlier_mask(const uint8_t * outlier_ch, uint8_t * is_out) {
    memset(is_out, 0, 128);
    for (int i = 0; i < TQ_N_OUTLIER; i++) is_out[outlier_ch[i]] = 1;
}

// Populate out_pos[] and reg_pos[] from membership mask.
static void tq_build_channel_split(const uint8_t * is_out,
                                   uint8_t * out_pos, uint8_t * reg_pos) {
    int oi = 0, ri = 0;
    for (int c = 0; c < 128; c++) {
        if (is_out[c]) out_pos[oi++] = (uint8_t)c;
        else           reg_pos[ri++] = (uint8_t)c;
    }
}

// Populate channel split and matrix pointers from per-layer state.
// Falls back to channels 0-31 as outliers and NULL matrices when unavailable.
static void tq_resolve_layer(int layer,
                             uint8_t * out_pos, uint8_t * reg_pos,
                             const float ** Pi_out, const float ** Pi_reg,
                             const float ** S) {
    uint8_t is_out[128];
    if (layer >= 0 && layer < TQ_MAX_LAYERS) {
        tq3_ensure_layer_matrices(layer);
        if (tq3_layers[layer].calibrated) {
            tq_build_outlier_mask(tq3_layers[layer].outlier_ch, is_out);
        } else {
            memset(is_out, 0, 128);
            for (int c = 0; c < TQ_N_OUTLIER; c++) is_out[c] = 1;
        }
        if (tq3_layers[layer].matrices_initialized) {
            *Pi_out = &tq3_layers[layer].Pi_out[0][0];
            *Pi_reg = &tq3_layers[layer].Pi_reg[0][0];
            *S      = &tq3_layers[layer].S[0][0];
        } else {
            *Pi_out = *Pi_reg = *S = NULL;
        }
    } else {
        memset(is_out, 0, 128);
        for (int c = 0; c < TQ_N_OUTLIER; c++) is_out[c] = 1;
        *Pi_out = *Pi_reg = *S = NULL;
    }
    tq_build_channel_split(is_out, out_pos, reg_pos);
}

#endif // TQ3_IMPL_HELPERS_DEFINED

// ---------------------------------------------------------------------------
// Per-instantiation name macro.  Defined fresh on each include so TQ_FN(foo)
// expands to foo_tq3_128 (first) or foo_tq4_128 (second).
// ---------------------------------------------------------------------------
#define TQ_PASTE2(a, b) a##_##b
#define TQ_PASTE(a, b)  TQ_PASTE2(a, b)
#define TQ_FN(name)     TQ_PASTE(name, TQ_NAME)

// ===========================================================================
// QUANTIZE — float → TQ_BLOCK
// Public names: quantize_row_tq3_128_ref  /  quantize_row_tq4_128_ref
// ===========================================================================
static void TQ_FN(quant_impl)(const float * GGML_RESTRICT x,
                               TQ_BLOCK * GGML_RESTRICT y, int64_t k) {
    assert(k % TQ_N == 0);
    const int64_t nb = k / TQ_N;

    int layer = tq3_get_layer();
    if (layer >= 0 && layer < TQ_MAX_LAYERS) {
        tq3_ensure_layer_matrices(layer);
        tq3_calibrate_outliers(layer, x, k);
    }

    uint8_t out_pos[TQ_N_OUTLIER];
    uint8_t reg_pos[TQ_N_REGULAR];
    const float * Pi_out, * Pi_reg, * S;
    tq_resolve_layer(layer, out_pos, reg_pos, &Pi_out, &Pi_reg, &S);

    for (int64_t i = 0; i < nb; i++) {
        const float * xi = x + i * TQ_N;
        TQ_BLOCK    * yi = y + i;

        // Split into outlier / regular sub-vectors.
        float x_out[TQ_N_OUTLIER];
        float x_reg[TQ_N_REGULAR];
        for (int j = 0; j < TQ_N_OUTLIER; j++) x_out[j] = xi[out_pos[j]];
        for (int j = 0; j < TQ_N_REGULAR; j++) x_reg[j] = xi[reg_pos[j]];

        // ---- Outlier group MSE --------------------------------------------
        float out_norm;
        uint8_t out_idx[TQ_N_OUTLIER];
        float x_mse_out[TQ_N_OUTLIER];
        {
            // Norm and unit vector.
            float norm = 0.0f;
            for (int j = 0; j < TQ_N_OUTLIER; j++) norm += x_out[j] * x_out[j];
            norm = sqrtf(norm);
            out_norm = norm;
            float inv_norm = (norm > 1e-12f) ? 1.0f / norm : 0.0f;

            float x_unit[TQ_N_OUTLIER];
            for (int j = 0; j < TQ_N_OUTLIER; j++) x_unit[j] = x_out[j] * inv_norm;

            // y_rot = Pi_out × x_unit
            float y_rot[TQ_N_OUTLIER];
            if (Pi_out) tq_matvec(Pi_out, x_unit, y_rot, TQ_N_OUTLIER, TQ_N_OUTLIER);
            else        for (int j = 0; j < TQ_N_OUTLIER; j++) y_rot[j] = x_unit[j];

            // Scalar VQ in rotated domain.
            float y_deq[TQ_N_OUTLIER];
            for (int j = 0; j < TQ_N_OUTLIER; j++) {
                int idx    = tq_find_idx(y_rot[j], TQ_OUT_BD, TQ_OUT_NCLUSTERS);
                out_idx[j] = (uint8_t)idx;
                y_deq[j]   = TQ_OUT_CB[idx];
            }

            // Dequantize back: x_mse_out = norm * Pi_out^T × y_deq
            float x_mse_unit[TQ_N_OUTLIER];
            if (Pi_out) tq_matvec_t(Pi_out, y_deq, x_mse_unit, TQ_N_OUTLIER, TQ_N_OUTLIER);
            else        for (int j = 0; j < TQ_N_OUTLIER; j++) x_mse_unit[j] = y_deq[j];
            for (int j = 0; j < TQ_N_OUTLIER; j++) x_mse_out[j] = norm * x_mse_unit[j];

            // Pack MSE indices.
#if TQ_OUT_BITS == 3
            tq_pack_3bit(out_idx, yi->out_idx);
#else
            // 4-bit nibble: 2 indices per byte.
            for (int j = 0; j < TQ_N_OUTLIER/2; j++)
                yi->out_idx[j] = (out_idx[2*j] & 0xf) | ((out_idx[2*j+1] & 0xf) << 4);
#endif
            yi->out_norm = GGML_FP32_TO_FP16(norm);
        }

        // ---- Regular group MSE --------------------------------------------
        float reg_norm;
        uint8_t reg_idx[TQ_N_REGULAR];
        float x_mse_reg[TQ_N_REGULAR];
        {
            float norm = 0.0f;
            for (int j = 0; j < TQ_N_REGULAR; j++) norm += x_reg[j] * x_reg[j];
            norm = sqrtf(norm);
            reg_norm = norm;
            float inv_norm = (norm > 1e-12f) ? 1.0f / norm : 0.0f;

            float x_unit[TQ_N_REGULAR];
            for (int j = 0; j < TQ_N_REGULAR; j++) x_unit[j] = x_reg[j] * inv_norm;

            float y_rot[TQ_N_REGULAR];
            if (Pi_reg) tq_matvec(Pi_reg, x_unit, y_rot, TQ_N_REGULAR, TQ_N_REGULAR);
            else        for (int j = 0; j < TQ_N_REGULAR; j++) y_rot[j] = x_unit[j];

            float y_deq[TQ_N_REGULAR];
            for (int j = 0; j < TQ_N_REGULAR; j++) {
                int idx    = tq_find_idx(y_rot[j], TQ_REG_BD, TQ_REG_NCLUSTERS);
                reg_idx[j] = (uint8_t)idx;
                y_deq[j]   = TQ_REG_CB[idx];
            }

            // Dequantize back: x_mse_reg = norm * Pi_reg^T × y_deq
            float x_mse_unit[TQ_N_REGULAR];
            if (Pi_reg) tq_matvec_t(Pi_reg, y_deq, x_mse_unit, TQ_N_REGULAR, TQ_N_REGULAR);
            else        for (int j = 0; j < TQ_N_REGULAR; j++) x_mse_unit[j] = y_deq[j];
            for (int j = 0; j < TQ_N_REGULAR; j++) x_mse_reg[j] = norm * x_mse_unit[j];

            // Pack MSE indices.
#if TQ_REG_BITS == 1
            // 1-bit: 8 indices per byte.
            memset(yi->reg_idx, 0, TQ_N_REGULAR/8);
            for (int j = 0; j < TQ_N_REGULAR; j++)
                if (reg_idx[j] & 1) yi->reg_idx[j/8] |= (uint8_t)(1u << (j%8));
#else
            // 2-bit: 4 indices per byte.
            for (int j = 0; j < TQ_N_REGULAR/4; j++)
                yi->reg_idx[j] = (reg_idx[4*j] & 3) | ((reg_idx[4*j+1] & 3) << 2)
                               | ((reg_idx[4*j+2] & 3) << 4) | ((reg_idx[4*j+3] & 3) << 6);
#endif
            yi->reg_norm = GGML_FP32_TO_FP16(norm);
        }

        // ---- UNIFIED QJL on full 128-dim residual -------------------------
        {
            // Reassemble x_mse[128] at original channel positions.
            float x_mse[128];
            for (int j = 0; j < TQ_N_OUTLIER; j++) x_mse[out_pos[j]] = x_mse_out[j];
            for (int j = 0; j < TQ_N_REGULAR; j++) x_mse[reg_pos[j]] = x_mse_reg[j];

            // Unified residual: r[j] = x[j] - x_mse[j]
            float r[128];
            float resnorm = 0.0f;
            for (int j = 0; j < 128; j++) {
                r[j] = xi[j] - x_mse[j];
                resnorm += r[j] * r[j];
            }
            resnorm = sqrtf(resnorm);

            // QJL signs: bit=1 iff row j of S dotted with r is non-negative.
            memset(yi->qjl, 0, sizeof(yi->qjl));
            if (S) {
                for (int j = 0; j < 128; j++) {
                    float dot = 0.0f;
                    for (int m = 0; m < 128; m++)
                        dot += S[j*128+m] * r[m];
                    if (dot >= 0.0f) yi->qjl[j/8] |= (uint8_t)(1u << (j%8));
                }
            }

            yi->resnorm = GGML_FP32_TO_FP16(resnorm);
        }

        (void)out_norm; (void)reg_norm;
    }
}

// Public quantize_row_ref — exact names from ggml-quants.h.
#if TQ_OUT_BITS == 3
void quantize_row_tq3_128_ref(const float * GGML_RESTRICT x,
                               block_tq3_128 * GGML_RESTRICT y, int64_t k) {
    quant_impl_tq3_128(x, y, k);
}

// size_t wrapper for ggml_quantize_chunk dispatch (ggml.c, GGML_TYPE_TQ3_128).
size_t quantize_tq3_128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    quantize_row_tq3_128_ref(src, (block_tq3_128 *)dst, nrows * n_per_row);
    return (size_t)(nrows * n_per_row / 128) * sizeof(block_tq3_128);
}
#else
void quantize_row_tq4_128_ref(const float * GGML_RESTRICT x,
                               block_tq4_128 * GGML_RESTRICT y, int64_t k) {
    quant_impl_tq4_128(x, y, k);
}
#endif

// ===========================================================================
// DEQUANTIZE — TQ_BLOCK → float
// Public names: dequantize_row_tq3_128  /  dequantize_row_tq4_128
// ===========================================================================
static void TQ_FN(dequant_impl)(const TQ_BLOCK * GGML_RESTRICT x,
                                 float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQ_N == 0);
    const int64_t nb = k / TQ_N;

    int layer = tq3_get_layer();

    uint8_t out_pos[TQ_N_OUTLIER];
    uint8_t reg_pos[TQ_N_REGULAR];
    const float * Pi_out, * Pi_reg, * S;
    tq_resolve_layer(layer, out_pos, reg_pos, &Pi_out, &Pi_reg, &S);

    const float QJL_SCALE = TQ_SQRT_PI_2 / 128.0f;

    for (int64_t i = 0; i < nb; i++) {
        const TQ_BLOCK * xi = x + i;
        float          * yi = y + i * TQ_N;

        float norm_out = GGML_FP16_TO_FP32(xi->out_norm);
        float norm_reg = GGML_FP16_TO_FP32(xi->reg_norm);
        float resnorm  = GGML_FP16_TO_FP32(xi->resnorm);

        // ---- Outlier group MSE --------------------------------------------
        float x_mse_out[TQ_N_OUTLIER];
        {
            uint8_t out_idx[TQ_N_OUTLIER];
#if TQ_OUT_BITS == 3
            tq_unpack_3bit(xi->out_idx, out_idx);
#else
            for (int j = 0; j < TQ_N_OUTLIER/2; j++) {
                out_idx[2*j]   = xi->out_idx[j] & 0xf;
                out_idx[2*j+1] = (xi->out_idx[j] >> 4) & 0xf;
            }
#endif
            float y_deq[TQ_N_OUTLIER];
            for (int j = 0; j < TQ_N_OUTLIER; j++) y_deq[j] = TQ_OUT_CB[out_idx[j]];

            // x_mse_out = norm_out * Pi_out^T × y_deq
            float x_mse_unit[TQ_N_OUTLIER];
            if (Pi_out) tq_matvec_t(Pi_out, y_deq, x_mse_unit, TQ_N_OUTLIER, TQ_N_OUTLIER);
            else        for (int j = 0; j < TQ_N_OUTLIER; j++) x_mse_unit[j] = y_deq[j];
            for (int j = 0; j < TQ_N_OUTLIER; j++) x_mse_out[j] = norm_out * x_mse_unit[j];
        }

        // ---- Regular group MSE --------------------------------------------
        float x_mse_reg[TQ_N_REGULAR];
        {
            uint8_t reg_idx[TQ_N_REGULAR];
#if TQ_REG_BITS == 1
            for (int j = 0; j < TQ_N_REGULAR; j++)
                reg_idx[j] = (xi->reg_idx[j/8] >> (j%8)) & 1;
#else
            for (int j = 0; j < TQ_N_REGULAR/4; j++) {
                uint8_t b = xi->reg_idx[j];
                reg_idx[4*j]   =  b       & 3;
                reg_idx[4*j+1] = (b >> 2) & 3;
                reg_idx[4*j+2] = (b >> 4) & 3;
                reg_idx[4*j+3] = (b >> 6) & 3;
            }
#endif
            float y_deq[TQ_N_REGULAR];
            for (int j = 0; j < TQ_N_REGULAR; j++) y_deq[j] = TQ_REG_CB[reg_idx[j]];

            // x_mse_reg = norm_reg * Pi_reg^T × y_deq
            float x_mse_unit[TQ_N_REGULAR];
            if (Pi_reg) tq_matvec_t(Pi_reg, y_deq, x_mse_unit, TQ_N_REGULAR, TQ_N_REGULAR);
            else        for (int j = 0; j < TQ_N_REGULAR; j++) x_mse_unit[j] = y_deq[j];
            for (int j = 0; j < TQ_N_REGULAR; j++) x_mse_reg[j] = norm_reg * x_mse_unit[j];
        }

        // ---- Reassemble x_mse[128] and apply UNIFIED QJL correction -------
        {
            // Place MSE reconstructions at original channel positions.
            float x_mse[128];
            for (int j = 0; j < TQ_N_OUTLIER; j++) x_mse[out_pos[j]] = x_mse_out[j];
            for (int j = 0; j < TQ_N_REGULAR; j++) x_mse[reg_pos[j]] = x_mse_reg[j];

            // Unpack 128 sign bits → sign_vec.
            float sign_vec[128];
            for (int j = 0; j < 128; j++)
                sign_vec[j] = ((xi->qjl[j/8] >> (j%8)) & 1) ? 1.0f : -1.0f;

            // QJL correction: (sqrt(pi/2)/128) * resnorm * S^T × sign_vec
            float qjl_corr[128] = {0};
            if (S) tq_matvec_t(S, sign_vec, qjl_corr, 128, 128);
            float scale = QJL_SCALE * resnorm;

            for (int j = 0; j < 128; j++)
                yi[j] = x_mse[j] + scale * qjl_corr[j];
        }
    }
}

// Public dequantize_row — exact names from ggml-quants.h.
#if TQ_OUT_BITS == 3
void dequantize_row_tq3_128(const block_tq3_128 * GGML_RESTRICT x,
                             float * GGML_RESTRICT y, int64_t k) {
    dequant_impl_tq3_128(x, y, k);
}
#else
void dequantize_row_tq4_128(const block_tq4_128 * GGML_RESTRICT x,
                             float * GGML_RESTRICT y, int64_t k) {
    dequant_impl_tq4_128(x, y, k);
}
#endif

// ===========================================================================
// QUERY PREPROCESSING — float → block_tq3_q_128
// Emitted once (TQ3 include only) since the query block format is shared.
// Public name: from_float_tq3_q_128
// ===========================================================================
#if TQ_OUT_BITS == 3
void from_float_tq3_q_128(const float * GGML_RESTRICT x,
                           block_tq3_q_128 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQ_N == 0);
    const int64_t nb = k / TQ_N;

    int layer = tq3_get_layer();
    if (layer >= 0 && layer < TQ_MAX_LAYERS) {
        tq3_ensure_layer_matrices(layer);
    }

    uint8_t out_pos[TQ_N_OUTLIER];
    uint8_t reg_pos[TQ_N_REGULAR];
    const float * Pi_out, * Pi_reg, * S;
    tq_resolve_layer(layer, out_pos, reg_pos, &Pi_out, &Pi_reg, &S);

    for (int64_t i = 0; i < nb; i++) {
        const float       * xi = x + i * TQ_N;
        block_tq3_q_128   * qi = y + i;

        // Split query into outlier and regular sub-vectors.
        float q_out[TQ_N_OUTLIER];
        float q_reg[TQ_N_REGULAR];
        for (int j = 0; j < TQ_N_OUTLIER; j++) q_out[j] = xi[out_pos[j]];
        for (int j = 0; j < TQ_N_REGULAR; j++) q_reg[j] = xi[reg_pos[j]];

        // q_rot = Pi × q  (rotated domain for MSE dot product).
        if (Pi_out) tq_matvec(Pi_out, q_out, qi->q_rot_out, TQ_N_OUTLIER, TQ_N_OUTLIER);
        else        memcpy(qi->q_rot_out, q_out, TQ_N_OUTLIER * sizeof(float));

        if (Pi_reg) tq_matvec(Pi_reg, q_reg, qi->q_rot_reg, TQ_N_REGULAR, TQ_N_REGULAR);
        else        memcpy(qi->q_rot_reg, q_reg, TQ_N_REGULAR * sizeof(float));

        // q_proj = S × query[128]  (UNIFIED — full 128-dim query for QJL dot product).
        if (S) tq_matvec(S, xi, qi->q_proj, 128, 128);
        else { memcpy(qi->q_proj, xi, 128 * sizeof(float));
               static int _w; if(!_w++){fprintf(stderr,"WARN: from_float S=NULL layer=%d\n",layer);} }
    }
}
#endif // TQ_OUT_BITS == 3

// ===========================================================================
// VEC-DOT — fused dequant + dot product, no full float reconstruction.
//
//   vx: quantized key blocks  (TQ_BLOCK[n/128])
//   vy: preprocessed query    (block_tq3_q_128[n/128])
//
// Per block contribution:
//   out_MSE:  norm_out  * Σ_j CB_out[idx_out[j]] * q_rot_out[j]
//   reg_MSE:  norm_reg  * Σ_j CB_reg[idx_reg[j]] * q_rot_reg[j]
//   QJL:      (sqrt(pi/2)/128) * resnorm * Σ_j q_proj[j] * sign[j]
//
// QJL_SCALE = sqrt(pi/2)/128  (UNIFIED 128-dim residual).
// ===========================================================================
static void TQ_FN(vec_dot_impl)(int n, float * GGML_RESTRICT s, size_t bs,
                                 const void * GGML_RESTRICT vx, size_t bx,
                                 const void * GGML_RESTRICT vy, size_t by,
                                 int nrc) {
    assert(n % TQ_N == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const int64_t nb = n / TQ_N;

    const TQ_BLOCK        * xb = (const TQ_BLOCK        *)vx;
    const block_tq3_q_128 * yb = (const block_tq3_q_128 *)vy;

    const float QJL_SCALE = TQ_SQRT_PI_2 / 128.0f;

    float sum = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const TQ_BLOCK        * xi = xb + i;
        const block_tq3_q_128 * qi = yb + i;

        float norm_out = GGML_FP16_TO_FP32(xi->out_norm);
        float norm_reg = GGML_FP16_TO_FP32(xi->reg_norm);
        float resnorm  = GGML_FP16_TO_FP32(xi->resnorm);

        // ---- Outlier MSE (rotated domain) ----------------------------------
        {
            uint8_t out_idx[TQ_N_OUTLIER];
#if TQ_OUT_BITS == 3
            tq_unpack_3bit(xi->out_idx, out_idx);
#else
            for (int j = 0; j < TQ_N_OUTLIER/2; j++) {
                out_idx[2*j]   = xi->out_idx[j] & 0xf;
                out_idx[2*j+1] = (xi->out_idx[j] >> 4) & 0xf;
            }
#endif
            float dot = 0.0f;
            for (int j = 0; j < TQ_N_OUTLIER; j++)
                dot += TQ_OUT_CB[out_idx[j]] * qi->q_rot_out[j];
            sum += norm_out * dot;
        }

        // ---- Regular MSE (rotated domain) ----------------------------------
        {
            uint8_t reg_idx[TQ_N_REGULAR];
#if TQ_REG_BITS == 1
            for (int j = 0; j < TQ_N_REGULAR; j++)
                reg_idx[j] = (xi->reg_idx[j/8] >> (j%8)) & 1;
#else
            for (int j = 0; j < TQ_N_REGULAR/4; j++) {
                uint8_t b = xi->reg_idx[j];
                reg_idx[4*j]   =  b       & 3;
                reg_idx[4*j+1] = (b >> 2) & 3;
                reg_idx[4*j+2] = (b >> 4) & 3;
                reg_idx[4*j+3] = (b >> 6) & 3;
            }
#endif
            float dot = 0.0f;
            for (int j = 0; j < TQ_N_REGULAR; j++)
                dot += TQ_REG_CB[reg_idx[j]] * qi->q_rot_reg[j];
            sum += norm_reg * dot;
        }

        // ---- UNIFIED QJL (128-dim residual) --------------------------------
        {
            float dot = 0.0f;
            for (int j = 0; j < 128; j++) {
                float sign = ((xi->qjl[j/8] >> (j%8)) & 1) ? 1.0f : -1.0f;
                dot += qi->q_proj[j] * sign;
            }
            sum += QJL_SCALE * resnorm * dot;
        }
    }

    *s = sum;
}

// Public vec_dot entry points — exact names from ggml-quants.h.
#if TQ_OUT_BITS == 3
void ggml_vec_dot_tq3_q_128(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    vec_dot_impl_tq3_128(n, s, bs, vx, bx, vy, by, nrc);
}
#else
void ggml_vec_dot_tq4_q_128(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    vec_dot_impl_tq4_128(n, s, bs, vx, bx, vy, by, nrc);
}
#endif

// ---------------------------------------------------------------------------
// Undef per-instantiation macros before the next include or EOF.
// ---------------------------------------------------------------------------
#undef TQ_FN
#undef TQ_PASTE
#undef TQ_PASTE2
#undef TQ_NAME
#undef TQ_BLOCK
#undef TQ_Q_BLOCK
#undef TQ_N
#undef TQ_OUT_BITS
#undef TQ_REG_BITS
#undef TQ_OUT_CB
#undef TQ_OUT_BD
#undef TQ_OUT_NCLUSTERS
#undef TQ_REG_CB
#undef TQ_REG_BD
#undef TQ_REG_NCLUSTERS
