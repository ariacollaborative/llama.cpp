// tq3_impl.inc — TurboQuant KV cache quantization, uniform d=128
//
// Each 128-element block:
//   MSE:  normalize → Pi×x → 2-bit scalar VQ → pack → residual
//   QJL:  sign(S × residual) stored as 128 bits
//
// Vecdot (fused, no float reconstruction):
//   MSE term:  norm × Σ_j codebook[idx[j]] × q_rot[j]
//   QJL term:  (sqrt(pi/2)/128) × resnorm × Σ_j q_proj[j] × sign[j]

// sqrt(pi/2) for QJL amplitude correction
#define TQ_SQRT_PI_2  0.886226925452758f
#define TQ_QJL_SCALE  (TQ_SQRT_PI_2 / 128.0f)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Matrix-vector multiply: out[d] = M[d×d] × v[d]
static void tq_matvec(const float * M, const float * v, float * out, int d) {
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        for (int j = 0; j < d; j++) acc += M[i*d+j] * v[j];
        out[i] = acc;
    }
}

// Transpose matrix-vector multiply: out[d] = M[d×d]^T × v[d]
static void tq_matvec_t(const float * M, const float * v, float * out, int d) {
    for (int j = 0; j < d; j++) {
        float acc = 0.0f;
        for (int i = 0; i < d; i++) acc += M[i*d+j] * v[i];
        out[j] = acc;
    }
}

// Codebook search: index of nearest centroid given 3 ascending boundaries.
static inline int tq_find_idx2(float val) {
    int best = 0;
    if (val > tq3_boundaries[0]) best = 1;
    if (val > tq3_boundaries[1]) best = 2;
    if (val > tq3_boundaries[2]) best = 3;
    return best;
}

// ---------------------------------------------------------------------------
// QUANTIZE — float[128] → block_tq3_128
// ---------------------------------------------------------------------------
void quantize_row_tq3_128_ref(const float * GGML_RESTRICT x,
                               block_tq3_128 * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    const int64_t nb = k / 128;

    int layer = tq3_get_layer();
    if (layer >= 0 && layer < TQ_MAX_LAYERS) {
        tq3_ensure_layer_matrices(layer);
    }

    const float * Pi = (layer >= 0 && layer < TQ_MAX_LAYERS && tq3_layers[layer].initialized)
                       ? &tq3_layers[layer].Pi[0][0] : NULL;
    const float * S  = (layer >= 0 && layer < TQ_MAX_LAYERS && tq3_layers[layer].initialized)
                       ? &tq3_layers[layer].S[0][0]  : NULL;

    for (int64_t i = 0; i < nb; i++) {
        const float     * xi = x + i * 128;
        block_tq3_128   * yi = y + i;

        // Compute L2 norm and unit vector
        float norm = 0.0f;
        for (int j = 0; j < 128; j++) norm += xi[j] * xi[j];
        norm = sqrtf(norm);
        yi->norm = GGML_FP32_TO_FP16(norm);
        float inv_norm = (norm > 1e-12f) ? 1.0f / norm : 0.0f;

        float x_unit[128];
        for (int j = 0; j < 128; j++) x_unit[j] = xi[j] * inv_norm;

        // Rotate: y_rot = Pi × x_unit
        float y_rot[128];
        if (Pi) tq_matvec(Pi, x_unit, y_rot, 128);
        else    for (int j = 0; j < 128; j++) y_rot[j] = x_unit[j];

        // Scalar VQ: assign each rotated component to nearest centroid
        uint8_t idx[128];
        float y_deq[128];
        for (int j = 0; j < 128; j++) {
            idx[j]   = (uint8_t)tq_find_idx2(y_rot[j]);
            y_deq[j] = tq3_codebook[idx[j]];
        }

        // Pack 2-bit indices: 4 per byte
        for (int j = 0; j < 128/4; j++) {
            yi->idx[j] = (idx[4*j] & 3)
                       | ((idx[4*j+1] & 3) << 2)
                       | ((idx[4*j+2] & 3) << 4)
                       | ((idx[4*j+3] & 3) << 6);
        }

        // Inverse-rotate MSE reconstruction: x_mse = norm * Pi^T × y_deq
        float x_mse_unit[128];
        if (Pi) tq_matvec_t(Pi, y_deq, x_mse_unit, 128);
        else    for (int j = 0; j < 128; j++) x_mse_unit[j] = y_deq[j];

        // Residual: r = x - norm * Pi^T × y_deq
        float r[128];
        float resnorm = 0.0f;
        for (int j = 0; j < 128; j++) {
            r[j] = xi[j] - norm * x_mse_unit[j];
            resnorm += r[j] * r[j];
        }
        resnorm = sqrtf(resnorm);
        yi->resnorm = GGML_FP32_TO_FP16(resnorm);

        // QJL signs: bit j = sign(S_j · r) >= 0
        memset(yi->qjl, 0, sizeof(yi->qjl));
        if (S) {
            for (int j = 0; j < 128; j++) {
                float dot = 0.0f;
                for (int m = 0; m < 128; m++) dot += S[j*128+m] * r[m];
                if (dot >= 0.0f) yi->qjl[j/8] |= (uint8_t)(1u << (j%8));
            }
        }
    }
}

size_t quantize_tq3_128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    quantize_row_tq3_128_ref(src, (block_tq3_128 *)dst, nrows * n_per_row);
    return (size_t)(nrows * n_per_row / 128) * sizeof(block_tq3_128);
}

// ---------------------------------------------------------------------------
// DEQUANTIZE — block_tq3_128 → float[128]
// ---------------------------------------------------------------------------
void dequantize_row_tq3_128(const block_tq3_128 * GGML_RESTRICT x,
                             float * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    const int64_t nb = k / 128;

    int layer = tq3_get_layer();

    const float * Pi = (layer >= 0 && layer < TQ_MAX_LAYERS && tq3_layers[layer].initialized)
                       ? &tq3_layers[layer].Pi[0][0] : NULL;
    const float * S  = (layer >= 0 && layer < TQ_MAX_LAYERS && tq3_layers[layer].initialized)
                       ? &tq3_layers[layer].S[0][0]  : NULL;

    for (int64_t i = 0; i < nb; i++) {
        const block_tq3_128 * xi = x + i;
        float               * yi = y + i * 128;

        float norm    = GGML_FP16_TO_FP32(xi->norm);
        float resnorm = GGML_FP16_TO_FP32(xi->resnorm);

        // Unpack 2-bit indices
        uint8_t idx[128];
        for (int j = 0; j < 128/4; j++) {
            uint8_t b = xi->idx[j];
            idx[4*j]   =  b       & 3;
            idx[4*j+1] = (b >> 2) & 3;
            idx[4*j+2] = (b >> 4) & 3;
            idx[4*j+3] = (b >> 6) & 3;
        }

        // MSE reconstruction: norm * Pi^T × centroids
        float y_deq[128];
        for (int j = 0; j < 128; j++) y_deq[j] = tq3_codebook[idx[j]];

        float x_mse[128];
        if (Pi) tq_matvec_t(Pi, y_deq, x_mse, 128);
        else    for (int j = 0; j < 128; j++) x_mse[j] = y_deq[j];
        for (int j = 0; j < 128; j++) x_mse[j] *= norm;

        // QJL correction: (sqrt(pi/2)/128) * resnorm * S^T × sign_vec
        float sign_vec[128];
        for (int j = 0; j < 128; j++)
            sign_vec[j] = ((xi->qjl[j/8] >> (j%8)) & 1) ? 1.0f : -1.0f;

        float qjl_corr[128] = {0};
        if (S) tq_matvec_t(S, sign_vec, qjl_corr, 128);

        float scale = TQ_QJL_SCALE * resnorm;
        for (int j = 0; j < 128; j++)
            yi[j] = x_mse[j] + scale * qjl_corr[j];
    }
}

// ---------------------------------------------------------------------------
// QUERY PREPROCESSING — float[128] → block_tq3_q_128
// ---------------------------------------------------------------------------
void from_float_tq3_q_128(const float * GGML_RESTRICT x,
                           block_tq3_q_128 * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    const int64_t nb = k / 128;

    int layer = tq3_get_layer();
    if (layer >= 0 && layer < TQ_MAX_LAYERS) {
        tq3_ensure_layer_matrices(layer);
    }

    const float * Pi = (layer >= 0 && layer < TQ_MAX_LAYERS && tq3_layers[layer].initialized)
                       ? &tq3_layers[layer].Pi[0][0] : NULL;
    const float * S  = (layer >= 0 && layer < TQ_MAX_LAYERS && tq3_layers[layer].initialized)
                       ? &tq3_layers[layer].S[0][0]  : NULL;

    for (int64_t i = 0; i < nb; i++) {
        const float       * xi = x + i * 128;
        block_tq3_q_128   * qi = y + i;

        // q_rot = Pi × query  (MSE dot product domain)
        if (Pi) tq_matvec(Pi, xi, qi->q_rot, 128);
        else    memcpy(qi->q_rot, xi, 128 * sizeof(float));

        // q_proj = S × query  (QJL domain)
        if (S) tq_matvec(S, xi, qi->q_proj, 128);
        else   memcpy(qi->q_proj, xi, 128 * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// VEC-DOT — fused dequant + dot product, no float reconstruction
//
//   vx: quantized key blocks  (block_tq3_128[n/128])
//   vy: preprocessed query    (block_tq3_q_128[n/128])
//
// Per block:
//   MSE term:  norm × Σ_j codebook[idx[j]] × q_rot[j]
//   QJL term:  (sqrt(pi/2)/128) × resnorm × Σ_j q_proj[j] × sign[j]
// ---------------------------------------------------------------------------
void ggml_vec_dot_tq3_q_128(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % 128 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const int64_t nb = n / 128;
    const block_tq3_128   * xb = (const block_tq3_128   *)vx;
    const block_tq3_q_128 * yb = (const block_tq3_q_128 *)vy;

    float sum = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const block_tq3_128   * xi = xb + i;
        const block_tq3_q_128 * qi = yb + i;

        float norm    = GGML_FP16_TO_FP32(xi->norm);
        float resnorm = GGML_FP16_TO_FP32(xi->resnorm);

        // MSE term: norm × Σ codebook[idx[j]] × q_rot[j]
        float mse_dot = 0.0f;
        for (int j = 0; j < 128/4; j++) {
            uint8_t b = xi->idx[j];
            mse_dot += tq3_codebook[ b       & 3] * qi->q_rot[4*j];
            mse_dot += tq3_codebook[(b >> 2) & 3] * qi->q_rot[4*j+1];
            mse_dot += tq3_codebook[(b >> 4) & 3] * qi->q_rot[4*j+2];
            mse_dot += tq3_codebook[(b >> 6) & 3] * qi->q_rot[4*j+3];
        }
        sum += norm * mse_dot;

        // QJL term: (sqrt(pi/2)/128) × resnorm × Σ q_proj[j] × sign[j]
        float qjl_dot = 0.0f;
        for (int j = 0; j < 128; j++) {
            float sign = ((xi->qjl[j/8] >> (j%8)) & 1) ? 1.0f : -1.0f;
            qjl_dot += qi->q_proj[j] * sign;
        }
        sum += TQ_QJL_SCALE * resnorm * qjl_dot;
    }

    *s = sum;
}

#undef TQ_SQRT_PI_2
#undef TQ_QJL_SCALE
