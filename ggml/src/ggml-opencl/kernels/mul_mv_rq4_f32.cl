/*
 * RotorQuant 4-bit MUL_MV kernel: dequantize rq4 + f32 dot product
 * Each work item computes one output: dot(dequant(key_row), query)
 */

#define RQ_D       128
#define RQ_N_GROUPS 43
#define RQ_N_LEVELS 16
#define RQ_BLOCK_BYTES 68

constant float RQ_CENTROIDS[16] = {
    -0.2415722564f, -0.1829257937f, -0.1430662130f, -0.1110830968f,
    -0.0833324871f, -0.0580811446f, -0.0343186837f, -0.0113553670f,
     0.0113553670f,  0.0343186837f,  0.0580811446f,  0.0833324871f,
     0.1110830968f,  0.1430662130f,  0.1829257937f,  0.2415722564f,
};

/* Inverse rotor sandwich: R̃·v·R
 * Same math as forward but with reversed rotor order */
void cl3_inv_sandwich_vec(float rs, float rb12, float rb13, float rb23,
                          float v1, float v2, float v3,
                          float *out1, float *out2, float *out3) {
    /* R̃ = [s, 0, 0, 0, -b12, -b13, -b23, 0] */
    float nb12 = -rb12, nb13 = -rb13, nb23 = -rb23;

    /* First product: R̃ · v */
    float t1 = rs*v1 + nb12*v2 + nb13*v3;
    float t2 = rs*v2 - nb12*v1 + nb23*v3;
    float t3 = rs*v3 - nb13*v1 - nb23*v2;
    float t7 = -nb23*v1 + nb13*v2 - nb12*v3;

    /* Second product: temp · R */
    *out1 = rs*t1 - t2*rb12 - t3*rb13 + t7*rb23;
    *out2 = rs*t2 + t1*rb12 - t3*rb23 - t7*rb13;
    *out3 = rs*t3 + t1*rb13 + t2*rb23 + t7*rb12;
}

kernel void kernel_mul_mv_rq4_f32(
        global const uchar * src0,     /* rq4 key cache */
        ulong                offset0,
        global const float * src1,     /* f32 query */
        ulong                offset1,
        global       float * dst,      /* f32 output scores */
        ulong                offsetd,
        int ne00,       /* head dim (128) */
        int ne01,       /* n_ctx (key rows) */
        ulong nb01,     /* key row stride (bytes) */
        ulong nb02,     /* key head stride */
        ulong nb03,     /* key batch stride */
        int ne12,       /* query heads */
        ulong nb11,     /* query row stride */
        ulong nb12,     /* query head stride */
        ulong nb13,     /* query batch stride */
        int ne0,        /* output dim 0 */
        int ne1,        /* output dim 1 */
        int r2,         /* GQA ratio */
        int r3,
        global const float * rotors  /* 43 × 4 rotor params */
) {
    int ir = get_global_id(0);   /* key row */
    int r1 = get_global_id(1);   /* query row (usually 0 for decode) */
    int im = get_global_id(2);   /* combined head/batch */

    if (ir >= ne01) return;

    int i12 = im % ne12;
    int i13 = im / ne12;
    int i02 = i12 / r2;
    int i03 = i13 / r3;

    /* Key block pointer */
    global const uchar * key_row = src0 + offset0 + ir*nb01 + i02*nb02 + i03*nb03;

    /* Query pointer */
    global const float * query = (global const float *)((global const char *)src1 + offset1 + r1*nb11 + i12*nb12 + i13*nb13);

    /* Read norm */
    float norm = vload_half(0, (global const half *)key_row);

    /* Dequantize + dot product fused */
    float sum = 0.0f;
    int idx_pos = 0;

    for (int g = 0; g < RQ_N_GROUPS; g++) {
        /* Unpack 3 centroids */
        float qv1, qv2, qv3;
        for (int c = 0; c < 3; c++) {
            int pos = idx_pos + c;
            int byte_idx = 2 + pos / 2;
            uchar packed;
            if (pos % 2 == 0) {
                packed = key_row[byte_idx] & 0x0F;
            } else {
                packed = (key_row[byte_idx] >> 4) & 0x0F;
            }
            float val = RQ_CENTROIDS[packed];
            if (c == 0) qv1 = val;
            else if (c == 1) qv2 = val;
            else qv3 = val;
        }
        idx_pos += 3;

        /* Inverse rotor sandwich */
        float rs   = rotors[g * 4 + 0];
        float rb12 = rotors[g * 4 + 1];
        float rb13 = rotors[g * 4 + 2];
        float rb23 = rotors[g * 4 + 3];

        float r1o, r2o, r3o;
        cl3_inv_sandwich_vec(rs, rb12, rb13, rb23, qv1, qv2, qv3, &r1o, &r2o, &r3o);

        /* Dot with query (scaled by norm) */
        int base = g * 3;
        if (base + 0 < RQ_D) sum += r1o * query[base + 0];
        if (base + 1 < RQ_D) sum += r2o * query[base + 1];
        if (base + 2 < RQ_D) sum += r3o * query[base + 2];
    }

    sum *= norm;

    /* Write output */
    dst[offsetd / sizeof(float) + im*ne0*ne1 + r1*ne0 + ir] = sum;
}
