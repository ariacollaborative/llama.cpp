#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Borrowed from set_rows.cl
inline uint fastdiv(uint n, uint4 v) { return (uint)(((ulong)n * v.s0 + v.s1) >> v.s3); }
inline uint fastmod(uint n, uint4 v) { return n - fastdiv(n, v) * v.s2; }

constant float tq3_codebook[4] = {-0.13304020f, -0.03999094f, 0.03999094f, 0.13304020f};
constant float tq3_boundaries[3] = {-0.08651557f, 0.0f, 0.08651557f};
constant float TQ3_QJL_SCALE = 1.2533141f;

inline void tq3_fwht(private float * data, int N) {
    for (int h = 1; h < N; h <<= 1)
        for (int i = 0; i < N; i += h * 2)
            for (int j = i; j < i + h; j++) {
                float a = data[j], b = data[j + h];
                data[j] = a + b; data[j + h] = a - b;
            }
    float s = 1.0f / sqrt((float)N);
    for (int i = 0; i < N; i++) data[i] *= s;
}

// Macro-based kernel to avoid duplicating 4 variants.
// IDX_TYPE: long or int.  TQ3_N_VAL: 128 or 256.
#define DEFINE_SET_ROWS_TQ3(SUFFIX, IDX_TYPE, TQ3_N_VAL, TQ3_BLOCK_BYTES_VAL) \
kernel void kernel_set_rows_tq3_##SUFFIX( \
        global char  * src0,   ulong offset0, \
        global char  * src1,   ulong offset1, \
        global char  * dst,    ulong offsetd, \
        int ne01, ulong nb01, ulong nb02, ulong nb03, \
        uint4 ne11, uint4 ne12, \
        ulong nb10, ulong nb11, ulong nb12, \
        int nblk0, ulong nb1, ulong nb2, ulong nb3, \
        global const float * signs, \
        global const float * s_transpose \
) { \
    const int TQ3_N = TQ3_N_VAL; \
    const int TQ3_BLOCK_BYTES = TQ3_BLOCK_BYTES_VAL; \
    \
    src0 += offset0; src1 += offset1; dst += offsetd; \
    \
    int i03 = get_group_id(2); \
    int i02 = get_group_id(1); \
    int i01 = get_group_id(0) * get_local_size(1) + get_local_id(1); \
    if (i01 >= ne01) return; \
    \
    int i12 = fastmod(i03, ne12); \
    int i11 = fastmod(i02, ne11); \
    IDX_TYPE dst_row_idx = ((global IDX_TYPE *)(src1 + i01*nb10 + i11*nb11 + i12*nb12))[0]; \
    \
    global float * block_src = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03) \
                               + get_local_id(0) * TQ3_N; \
    global uchar * block_dst = (global uchar *)(dst + dst_row_idx*nb1 + i02*nb2 + i03*nb3) \
                               + get_local_id(0) * TQ3_BLOCK_BYTES; \
    \
    if (get_local_id(0) >= nblk0) return; \
    \
    /* ONE private buffer — reused for every step */ \
    private float a[TQ3_N_VAL]; \
    \
    /* Step 1: Load and compute norm */ \
    float sum_sq = 0.0f; \
    for (int j = 0; j < TQ3_N; j++) { \
        a[j] = block_src[j]; \
        sum_sq += a[j] * a[j]; \
    } \
    float norm = sqrt(sum_sq); \
    \
    if (norm < 1e-10f) { \
        for (int j = 0; j < TQ3_BLOCK_BYTES; j++) block_dst[j] = 0; \
        return; \
    } \
    float inv_norm = 1.0f / norm; \
    \
    /* Step 2: Normalize and apply signs */ \
    for (int j = 0; j < TQ3_N; j++) \
        a[j] *= inv_norm * signs[j]; \
    \
    /* Step 3: FWHT in-place -> a = rotated */ \
    tq3_fwht(a, TQ3_N); \
    \
    /* Step 4: Quantize from a[], pack idx to output */ \
    for (int j = 0; j < TQ3_N; j++) { \
        int q = (a[j] > tq3_boundaries[2]) ? 3 : \
                (a[j] > tq3_boundaries[1]) ? 2 : \
                (a[j] > tq3_boundaries[0]) ? 1 : 0; \
        int bp = j / 4, bs = (j % 4) * 2; \
        if (bs == 0) block_dst[bp] = (uchar)q; \
        else block_dst[bp] |= ((uchar)q << bs); \
    } \
    \
    /* Step 5: Read back indices, reconstruct codebook values into a[] */ \
    for (int j = 0; j < TQ3_N; j++) { \
        int idx = (block_dst[j / 4] >> ((j % 4) * 2)) & 3; \
        a[j] = tq3_codebook[idx]; \
    } \
    /* Unrotate: FWHT then apply signs */ \
    tq3_fwht(a, TQ3_N); \
    for (int j = 0; j < TQ3_N; j++) \
        a[j] *= signs[j]; \
    /* a[] now holds deq_unrot */ \
    \
    /* Step 6: Compute residual = x_unit - deq_unrot, store in a[] */ \
    float res_sum_sq = 0.0f; \
    for (int j = 0; j < TQ3_N; j++) { \
        a[j] = block_src[j] * inv_norm - a[j]; \
        res_sum_sq += a[j] * a[j]; \
    } \
    float res_norm = sqrt(res_sum_sq); \
    \
    /* Step 7: QJL projection: sign(S * residual), pack bits */ \
    /* S[j][l] = S^T[l][j] = s_transpose[l * N + j] */ \
    int qjl_off = TQ3_N / 4; \
    for (int j = 0; j < TQ3_N / 8; j++) { \
        uchar packed = 0; \
        for (int k = 0; k < 8; k++) { \
            int jj = j * 8 + k; \
            float proj = 0.0f; \
            for (int l = 0; l < TQ3_N; l++) \
                proj += s_transpose[l * TQ3_N + jj] * a[l]; \
            if (proj >= 0.0f) packed |= (1 << k); \
        } \
        block_dst[qjl_off + j] = packed; \
    } \
    \
    /* Step 8: Write norms */ \
    int norm_off = 3 * TQ3_N / 8; \
    vstore_half(norm, 0, (global half *)(block_dst + norm_off)); \
    vstore_half(res_norm, 0, (global half *)(block_dst + norm_off + 2)); \
}

DEFINE_SET_ROWS_TQ3(128_i64, long, 128, 52)
DEFINE_SET_ROWS_TQ3(128_i32, int,  128, 52)
DEFINE_SET_ROWS_TQ3(256_i64, long, 256, 100)
DEFINE_SET_ROWS_TQ3(256_i32, int,  256, 100)
