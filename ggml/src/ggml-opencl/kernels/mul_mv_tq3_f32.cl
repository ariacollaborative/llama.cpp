#pragma OPENCL EXTENSION cl_khr_fp16 : enable

constant float tq3_codebook[4] = {-0.13304020f, -0.03999094f, 0.03999094f, 0.13304020f};
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

kernel void kernel_mul_mv_tq3_128_f32(
        global void  * src0,   ulong offset0,
        global float * src1,   ulong offset1,
        global float * dst,    ulong offsetd,
        int ne00, int ne01, int ne02, int ne10, int ne12,
        int ne0, int ne1, int r2, int r3,
        global const float * signs,
        global const float * s_transpose
) {
    #define TQ3_N 128
    #define BLOCK_BYTES 52

    src0 = (global void  *)((global char *)src0 + offset0);
    src1 = (global float *)((global char *)src1 + offset1);
    dst  = (global float *)((global char *)dst  + offsetd);

    int ir = get_global_id(0);
    if (ir >= ne01) return;
    int im = get_global_id(1);

    int i12 = im % ne12;
    int i13 = im / ne12;
    int i02 = i12 / r2;
    int i03 = i13 / r3;

    int num_blocks = ne00 / TQ3_N;

    global uchar * key_row = (global uchar *)src0
        + (ulong)i03 * ne02 * ne01 * (ulong)(num_blocks * BLOCK_BYTES)
        + (ulong)i02 * ne01 * (ulong)(num_blocks * BLOCK_BYTES)
        + (ulong)ir * (ulong)(num_blocks * BLOCK_BYTES);

    global float * query = src1 + (ulong)im * ne10;
    float total = 0.0f;

    for (int block = 0; block < num_blocks; block++) {
        global uchar * blk = key_row + block * BLOCK_BYTES;

        int norm_off = 3 * TQ3_N / 8;
        float norm    = vload_half(0, (global half *)(blk + norm_off));
        float resnorm = vload_half(0, (global half *)(blk + norm_off + 2));
        if (norm < 1e-10f) continue;

        private float q_rot[TQ3_N];
        int q_base = block * TQ3_N;
        for (int j = 0; j < TQ3_N; j++)
            q_rot[j] = query[q_base + j] * signs[j];
        tq3_fwht(q_rot, TQ3_N);

        float mse_sum = 0.0f;
        for (int j = 0; j < TQ3_N; j++) {
            int idx = (blk[j / 4] >> ((j % 4) * 2)) & 3;
            mse_sum += tq3_codebook[idx] * q_rot[j];
        }

        private float q_proj[TQ3_N];
        for (int j = 0; j < TQ3_N; j++) {
            float proj = 0.0f;
            for (int l = 0; l < TQ3_N; l++)
                proj += s_transpose[l * TQ3_N + j] * query[q_base + l];
            q_proj[j] = proj;
        }

        float qjl_sum = 0.0f;
        int qjl_off = TQ3_N / 4;
        for (int j = 0; j < TQ3_N; j++) {
            float sign = ((blk[qjl_off + j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
            qjl_sum += sign * q_proj[j];
        }

        float qjl_scale = TQ3_QJL_SCALE / (float)TQ3_N;
        total += norm * mse_sum + qjl_scale * norm * resnorm * qjl_sum;
    }

    dst[im * ne0 + ir] = total;
    #undef TQ3_N
    #undef BLOCK_BYTES
}

kernel void kernel_mul_mv_tq3_256_f32(
        global void  * src0,   ulong offset0,
        global float * src1,   ulong offset1,
        global float * dst,    ulong offsetd,
        int ne00, int ne01, int ne02, int ne10, int ne12,
        int ne0, int ne1, int r2, int r3,
        global const float * signs,
        global const float * s_transpose
) {
    #define TQ3_N 256
    #define BLOCK_BYTES 100

    src0 = (global void  *)((global char *)src0 + offset0);
    src1 = (global float *)((global char *)src1 + offset1);
    dst  = (global float *)((global char *)dst  + offsetd);

    int ir = get_global_id(0);
    if (ir >= ne01) return;
    int im = get_global_id(1);

    int i12 = im % ne12;
    int i13 = im / ne12;
    int i02 = i12 / r2;
    int i03 = i13 / r3;

    int num_blocks = ne00 / TQ3_N;

    global uchar * key_row = (global uchar *)src0
        + (ulong)i03 * ne02 * ne01 * (ulong)(num_blocks * BLOCK_BYTES)
        + (ulong)i02 * ne01 * (ulong)(num_blocks * BLOCK_BYTES)
        + (ulong)ir * (ulong)(num_blocks * BLOCK_BYTES);

    global float * query = src1 + (ulong)im * ne10;
    float total = 0.0f;

    for (int block = 0; block < num_blocks; block++) {
        global uchar * blk = key_row + block * BLOCK_BYTES;

        int norm_off = 3 * TQ3_N / 8;
        float norm    = vload_half(0, (global half *)(blk + norm_off));
        float resnorm = vload_half(0, (global half *)(blk + norm_off + 2));
        if (norm < 1e-10f) continue;

        private float q_rot[TQ3_N];
        int q_base = block * TQ3_N;
        for (int j = 0; j < TQ3_N; j++)
            q_rot[j] = query[q_base + j] * signs[j];
        tq3_fwht(q_rot, TQ3_N);

        float mse_sum = 0.0f;
        for (int j = 0; j < TQ3_N; j++) {
            int idx = (blk[j / 4] >> ((j % 4) * 2)) & 3;
            mse_sum += tq3_codebook[idx] * q_rot[j];
        }

        private float q_proj[TQ3_N];
        for (int j = 0; j < TQ3_N; j++) {
            float proj = 0.0f;
            for (int l = 0; l < TQ3_N; l++)
                proj += s_transpose[l * TQ3_N + j] * query[q_base + l];
            q_proj[j] = proj;
        }

        float qjl_sum = 0.0f;
        int qjl_off = TQ3_N / 4;
        for (int j = 0; j < TQ3_N; j++) {
            float sign = ((blk[qjl_off + j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
            qjl_sum += sign * q_proj[j];
        }

        float qjl_scale = TQ3_QJL_SCALE / (float)TQ3_N;
        total += norm * mse_sum + qjl_scale * norm * resnorm * qjl_sum;
    }

    dst[im * ne0 + ir] = total;
    #undef TQ3_N
    #undef BLOCK_BYTES
}
