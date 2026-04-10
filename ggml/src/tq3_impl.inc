// tq3_impl.inc — TQ3 quantization implementation body
// Instantiated twice: once for N=128 and once for N=256.
//
// Required defines before including:
//   TQ3_N      — block size (128 or 256)
//   TQ3_SUFFIX — token suffix: _128 or _256
//   TQ3_SIGNS  — float array name: tq3_signs_128 or tq3_signs_256
//   TQ3_S      — S matrix name: tq3_S_128 or tq3_S_256
//   TQ3_INIT   — init function: tq3_ensure_init_128 or tq3_ensure_init_256
//   TQ3_BLOCK  — block struct: block_tq3_128 or block_tq3_256
//   TQ3_QBLOCK — query block struct: block_tq3_q_128 or block_tq3_q_256

// Token-paste helpers to build function names.
// Three levels of indirection required so TQ3_SUFFIX expands before ##.
// TQ3_SUFFIX   must be defined as e.g. _128  → produces names like foo_128
// TQ3_SUFFIX_R must be defined as e.g. _128_ref → produces names like foo_128_ref
#define TQ3_PASTE2(a, b) a##b
#define TQ3_FNAME_IMPL(base, suf) TQ3_PASTE2(base, suf)
#define TQ3_FNAME(base)  TQ3_FNAME_IMPL(base, TQ3_SUFFIX)
#define TQ3_FNAMER(base) TQ3_FNAME_IMPL(base, TQ3_SUFFIX_R)

void TQ3_FNAMER(quantize_row_tq3)(const float * GGML_RESTRICT x,
                                   TQ3_BLOCK   * GGML_RESTRICT y,
                                   int64_t k) {
    TQ3_INIT();
    assert(k % TQ3_N == 0);
    const int nb = (int)(k / TQ3_N);

    for (int i = 0; i < nb; i++) {
        const float * xi = x + (int64_t)i * TQ3_N;

        // Compute L2 norm
        float norm = 0.0f;
        for (int j = 0; j < TQ3_N; j++) norm += xi[j] * xi[j];
        norm = sqrtf(norm);
        y[i].norm = GGML_FP32_TO_FP16(norm);

        if (norm < 1e-10f) {
            memset(y[i].idx, 0, sizeof(y[i].idx));
            memset(y[i].qjl, 0, sizeof(y[i].qjl));
            y[i].res_norm = GGML_FP32_TO_FP16(0.0f);
            continue;
        }

        // Normalize and rotate
        float x_unit[TQ3_N], y_rot[TQ3_N];
        const float inv_norm = 1.0f / norm;
        for (int j = 0; j < TQ3_N; j++) x_unit[j] = xi[j] * inv_norm;
        tq3_rotate(y_rot, x_unit, TQ3_SIGNS, TQ3_N);

        // Lloyd-Max 2-bit quantization
        float y_deq_rot[TQ3_N];
        memset(y[i].idx, 0, sizeof(y[i].idx));
        for (int j = 0; j < TQ3_N; j++) {
            int q = (y_rot[j] > tq3_boundaries[2]) ? 3 :
                    (y_rot[j] > tq3_boundaries[1]) ? 2 :
                    (y_rot[j] > tq3_boundaries[0]) ? 1 : 0;
            y[i].idx[j/4] |= (uint8_t)(q << ((j%4)*2));
            y_deq_rot[j] = tq3_codebook[q];
        }

        // Compute residual (in original space after unrotation)
        float x_deq[TQ3_N], residual[TQ3_N];
        tq3_unrotate(x_deq, y_deq_rot, TQ3_SIGNS, TQ3_N);
        float rnorm = 0.0f;
        for (int j = 0; j < TQ3_N; j++) {
            residual[j] = x_unit[j] - x_deq[j];
            rnorm += residual[j] * residual[j];
        }
        rnorm = sqrtf(rnorm);
        y[i].res_norm = GGML_FP32_TO_FP16(rnorm);

        // QJL: project residual onto Gaussian matrix S, store sign bits
        float r_proj[TQ3_N];
        for (int j = 0; j < TQ3_N; j++) {
            float sum = 0.0f;
            for (int l = 0; l < TQ3_N; l++) sum += TQ3_S[j][l] * residual[l];
            r_proj[j] = sum;
        }
        memset(y[i].qjl, 0, sizeof(y[i].qjl));
        for (int j = 0; j < TQ3_N; j++) {
            if (r_proj[j] >= 0.0f) y[i].qjl[j/8] |= (uint8_t)(1u << (j%8));
        }
    }
}

void TQ3_FNAME(dequantize_row_tq3)(const TQ3_BLOCK * GGML_RESTRICT x,
                                    float           * GGML_RESTRICT y,
                                    int64_t k) {
    TQ3_INIT();
    assert(k % TQ3_N == 0);
    const int nb = (int)(k / TQ3_N);
    // For orthonormal S (unit-norm rows), scale = sqrt(pi/2) / sqrt(N)
    // (raw Gaussian rows have norm sqrt(N), so projections are sqrt(N) larger)
    const float qjl_scale = sqrtf(3.14159265f / 2.0f) / sqrtf((float)TQ3_N);

    for (int i = 0; i < nb; i++) {
        float * yi = y + (int64_t)i * TQ3_N;
        const float norm  = GGML_FP16_TO_FP32(x[i].norm);
        const float rnorm = GGML_FP16_TO_FP32(x[i].res_norm);

        // Reconstruct MSE component
        float y_rot[TQ3_N];
        for (int j = 0; j < TQ3_N; j++) {
            int q = (x[i].idx[j/4] >> ((j%4)*2)) & 0x3;
            y_rot[j] = tq3_codebook[q];
        }
        tq3_unrotate(yi, y_rot, TQ3_SIGNS, TQ3_N);
        for (int j = 0; j < TQ3_N; j++) yi[j] *= norm;

        // QJL correction — disabled (noise > signal at d=128, see handoff)
        if (0 && rnorm > 1e-10f) {
            float qjl_signs[TQ3_N];
            for (int j = 0; j < TQ3_N; j++)
                qjl_signs[j] = (x[i].qjl[j/8] & (1u << (j%8))) ? 1.0f : -1.0f;

            // correction[j] = sum_l S[l][j] * qjl_signs[l]  (column j of S dot signs)
            float correction[TQ3_N];
            for (int j = 0; j < TQ3_N; j++) {
                float sum = 0.0f;
                for (int l = 0; l < TQ3_N; l++) sum += TQ3_S[l][j] * qjl_signs[l];
                correction[j] = sum;
            }
            const float s = norm * qjl_scale * rnorm;
            for (int j = 0; j < TQ3_N; j++) yi[j] += s * correction[j];
        }
    }
}

size_t TQ3_FNAME(quantize_tq3)(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrow, int64_t n_per_row,
                                const float * quant_weights) {
    GGML_UNUSED(quant_weights);
    TQ3_FNAMER(quantize_row_tq3)(src, (TQ3_BLOCK *)dst, nrow * n_per_row);
    return (size_t)(nrow * (n_per_row / TQ3_N) * (int64_t)sizeof(TQ3_BLOCK));
}

void TQ3_FNAME(from_float_tq3_q)(const float * GGML_RESTRICT x,
                                   void        * GGML_RESTRICT vy,
                                   int64_t k) {
    TQ3_INIT();
    assert(k % TQ3_N == 0);
    const int nb = (int)(k / TQ3_N);
    TQ3_QBLOCK * y = (TQ3_QBLOCK *)vy;

    for (int i = 0; i < nb; i++) {
        const float * xi = x + (int64_t)i * TQ3_N;

        // Compute Hadamard-rotated query
        tq3_rotate(y[i].q_rot, xi, TQ3_SIGNS, TQ3_N);

        // Compute Gaussian-projected query: q_proj = S * xi
#if defined(__ARM_NEON)
        for (int r = 0; r < TQ3_N; r++) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            for (int c = 0; c < TQ3_N; c += 16) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(&TQ3_S[r][c+ 0]), vld1q_f32(&xi[c+ 0]));
                acc1 = vfmaq_f32(acc1, vld1q_f32(&TQ3_S[r][c+ 4]), vld1q_f32(&xi[c+ 4]));
                acc2 = vfmaq_f32(acc2, vld1q_f32(&TQ3_S[r][c+ 8]), vld1q_f32(&xi[c+ 8]));
                acc3 = vfmaq_f32(acc3, vld1q_f32(&TQ3_S[r][c+12]), vld1q_f32(&xi[c+12]));
            }
            acc0 = vaddq_f32(acc0, acc1);
            acc2 = vaddq_f32(acc2, acc3);
            y[i].q_proj[r] = vaddvq_f32(vaddq_f32(acc0, acc2));
        }
#else
        for (int r = 0; r < TQ3_N; r++) {
            float sum = 0.0f;
            for (int c = 0; c < TQ3_N; c++) sum += TQ3_S[r][c] * xi[c];
            y[i].q_proj[r] = sum;
        }
#endif
    }
}

void TQ3_FNAME(ggml_vec_dot_tq3_q)(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, size_t bx,
                                    const void * GGML_RESTRICT vy, size_t by,
                                    int nrc) {
    assert(n % TQ3_N == 0);
    assert(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);

    const TQ3_BLOCK  * GGML_RESTRICT x = (const TQ3_BLOCK  *)vx;
    const TQ3_QBLOCK * GGML_RESTRICT y = (const TQ3_QBLOCK *)vy;
    const int nb = n / TQ3_N;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float norm = GGML_FP16_TO_FP32(x[i].norm);
        if (norm < 1e-10f) continue;

        float mse_dot = 0.0f;
        for (int j = 0; j < TQ3_N; j++) {
            int qi = (x[i].idx[j/4] >> ((j%4)*2)) & 0x3;
            mse_dot += y[i].q_rot[j] * tq3_codebook[qi];
        }
        sumf += norm * mse_dot;
    }

    *s = sumf;
}

#undef TQ3_FNAME
#undef TQ3_FNAMER
#undef TQ3_FNAME_IMPL
#undef TQ3_PASTE2
