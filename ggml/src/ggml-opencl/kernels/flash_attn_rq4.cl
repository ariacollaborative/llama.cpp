#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * RotorQuant fused Flash Attention kernel (single-query decode path)
 * Q is f32, K and V are rq4_128 (Clifford Cl(3,0) rotor quantized)
 *
 * Fuses: rq4 dequant (inverse rotor sandwich) + Q·K dot + softmax + V accumulation
 * No intermediate f32 buffers — dequant happens in registers per work item
 */

#define RQ_D       128
#define RQ_N_GROUPS 43
#define RQ_N_LEVELS 16
#define RQ_BLOCK_BYTES 68
#define Q1_WG_SIZE 64

/* Generated from C PRNG — must match ggml-rotorquant.c exactly */
constant float RQ_ROTORS[172] = {
    0.9976642132f, 0.0088128792f, -0.0490202419f, 0.0467509702f,
    0.0107257254f, -0.1211351231f, -0.4055028856f, 0.9059684277f,
    -0.9959692955f, 0.0677697510f, -0.0417615287f, 0.0413327441f,
    0.1681031287f, -0.6755748987f, -0.6980518103f, 0.1675219685f,
    -0.9694067240f, 0.0475489013f, -0.2283701897f, 0.0764000490f,
    0.3212832510f, -0.1365998834f, -0.9289582968f, -0.1231019944f,
    -0.9186392426f, -0.1255858094f, -0.2485945821f, -0.2802336216f,
    0.4664412737f, 0.1196258217f, -0.6079652905f, -0.6312689185f,
    -0.8449348807f, -0.4745880365f, -0.2366849929f, -0.0695087016f,
    0.5999531150f, 0.3359301388f, -0.3117833138f, -0.6557425857f,
    -0.7501338124f, -0.4956262708f, -0.2980700135f, -0.3206371963f,
    0.7184848785f, -0.1913920939f, -0.2409403324f, -0.6237758398f,
    -0.6366028786f, 0.6214520335f, -0.3282245696f, -0.3174947798f,
    0.8190773726f, -0.5710723400f, -0.0440310240f, 0.0324054174f,
    -0.5071769953f, 0.4393236637f, -0.7171422839f, 0.1883432716f,
    0.8992184401f, -0.1160676703f, -0.4067419469f, 0.1117830575f,
    -0.3650875986f, -0.0082612131f, -0.5526617169f, 0.7491381168f,
    0.9569075108f, 0.1489264965f, -0.1702868044f, 0.1820750087f,
    -0.2138826251f, -0.2888408601f, -0.2510066330f, 0.8987886310f,
    0.9907040000f, 0.1238467693f, -0.0208349153f, 0.0522823296f,
    -0.0573372170f, -0.9691430926f, -0.1458082944f, 0.1903000176f,
    -0.9997642040f, 0.0182636194f, 0.0058326698f, 0.0102048591f,
    0.1006398201f, 0.8908453584f, 0.2867781818f, -0.3376751542f,
    -0.9838616848f, -0.1044625640f, 0.0390071273f, -0.1399372071f,
    0.2561039329f, 0.4611684382f, 0.3774039745f, -0.7611181736f,
    -0.9433935285f, -0.3095525503f, 0.1109823883f, -0.0432296805f,
    0.4051736593f, 0.2992885709f, 0.3761443794f, -0.7776736617f,
    -0.8793703318f, 0.0277718473f, 0.3773615956f, -0.2890237868f,
    0.5441266894f, -0.5709750652f, 0.3841791749f, -0.4799168110f,
    -0.7933904529f, 0.4857406914f, 0.3587286174f, -0.0768201500f,
    0.6694936752f, -0.2949430943f, 0.5447410345f, 0.4099318087f,
    -0.6876009703f, 0.5311442018f, 0.3492735922f, 0.3508542180f,
    0.7781444192f, -0.1064620242f, 0.5093771219f, 0.3516987860f,
    -0.5646431446f, -0.0643410310f, 0.2910266221f, 0.7696374655f,
    0.8673660755f, 0.3966456652f, 0.1537539512f, 0.2582789361f,
    -0.4275868237f, -0.3988423944f, 0.1739822924f, 0.7923538685f,
    0.9349309206f, 0.0581518933f, 0.1258534342f, 0.3266243637f,
    -0.2798544168f, -0.1683942378f, 0.9445449114f, 0.0340566300f,
    0.9791519046f, -0.0156263337f, 0.2020228058f, -0.0142886359f,
    -0.1251344085f, 0.7034746408f, 0.6870425940f, -0.1320505589f,
    0.9989250302f, -0.0248887558f, 0.0286992155f, -0.0265686344f,
    0.0327099226f, 0.4154252708f, 0.7251529694f, -0.5481835604f,
    -0.9937564135f, -0.0867442638f, 0.0688556135f, -0.0135195125f,
};

constant float RQ_CENTROIDS[16] = {
    -0.2415722564f, -0.1829257937f, -0.1430662130f, -0.1110830968f,
    -0.0833324871f, -0.0580811446f, -0.0343186837f, -0.0113553670f,
     0.0113553670f,  0.0343186837f,  0.0580811446f,  0.0833324871f,
     0.1110830968f,  0.1430662130f,  0.1829257937f,  0.2415722564f,
};

/* Inverse rotor sandwich: dequant 3 centroid values → 3 original-space values */
void cl3_inv_sandwich(float rs, float rb12, float rb13, float rb23,
                      float v1, float v2, float v3,
                      float *out1, float *out2, float *out3) {
    float nb12 = -rb12, nb13 = -rb13, nb23 = -rb23;
    float t1 = rs*v1 + nb12*v2 + nb13*v3;
    float t2 = rs*v2 - nb12*v1 + nb23*v3;
    float t3 = rs*v3 - nb13*v1 - nb23*v2;
    float t7 = -nb23*v1 + nb13*v2 - nb12*v3;
    *out1 = rs*t1 - t2*rb12 - t3*rb13 + t7*rb23;
    *out2 = rs*t2 + t1*rb12 - t3*rb23 - t7*rb13;
    *out3 = rs*t3 + t1*rb13 + t2*rb23 + t7*rb12;
}

/* Dequantize one rq4 block into a float array (128 elements) */
void rq4_dequant_block(const global uchar * blk, float * out) {
    float norm = vload_half(0, (const global half *)blk);
    int idx_pos = 0;
    for (int g = 0; g < RQ_N_GROUPS; g++) {
        float qv1, qv2, qv3;
        for (int c = 0; c < 3; c++) {
            int pos = idx_pos + c;
            int byte_idx = 2 + pos / 2;
            uchar packed = (pos % 2 == 0) ? (blk[byte_idx] & 0x0F) : ((blk[byte_idx] >> 4) & 0x0F);
            float val = RQ_CENTROIDS[packed];
            if (c == 0) qv1 = val; else if (c == 1) qv2 = val; else qv3 = val;
        }
        idx_pos += 3;

        float rs   = RQ_ROTORS[g * 4 + 0];
        float rb12 = RQ_ROTORS[g * 4 + 1];
        float rb13 = RQ_ROTORS[g * 4 + 2];
        float rb23 = RQ_ROTORS[g * 4 + 3];

        float r1, r2, r3;
        cl3_inv_sandwich(rs, rb12, rb13, rb23, qv1, qv2, qv3, &r1, &r2, &r3);

        int base = g * 3;
        out[base + 0] = (base + 0 < RQ_D) ? r1 * norm : 0.0f;
        out[base + 1] = (base + 1 < RQ_D) ? r2 * norm : 0.0f;
        out[base + 2] = (base + 2 < RQ_D) ? r3 * norm : 0.0f;
    }
}

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) return 1.0f;
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
    return pow(base, exph);
}

__kernel void flash_attn_rq4_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;
    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    /* Load query into private registers */
    float q_priv[RQ_D];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global float* q_ptr = (const global float*)(q_base + q_row_offset);
    for (int i = 0; i < RQ_D; i++) {
        q_priv[i] = q_ptr[i];
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    /* Pass 1: Q·K scores → find max */
    float m_i = -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global uchar * k_blk = (const global uchar*)(k_base + k_row_offset);

        /* Fused dequant + dot */
        float norm_k = vload_half(0, (const global half *)k_blk);
        float score = 0.0f;
        int idx_pos = 0;
        for (int g = 0; g < RQ_N_GROUPS; g++) {
            float qv1, qv2, qv3;
            for (int c = 0; c < 3; c++) {
                int pos = idx_pos + c;
                int byte_idx = 2 + pos / 2;
                uchar packed = (pos % 2 == 0) ? (k_blk[byte_idx] & 0x0F) : ((k_blk[byte_idx] >> 4) & 0x0F);
                float val = RQ_CENTROIDS[packed];
                if (c == 0) qv1 = val; else if (c == 1) qv2 = val; else qv3 = val;
            }
            idx_pos += 3;

            float rs = RQ_ROTORS[g*4], rb12 = RQ_ROTORS[g*4+1], rb13 = RQ_ROTORS[g*4+2], rb23 = RQ_ROTORS[g*4+3];
            float r1, r2, r3;
            cl3_inv_sandwich(rs, rb12, rb13, rb23, qv1, qv2, qv3, &r1, &r2, &r3);

            int base = g * 3;
            if (base + 0 < RQ_D) score += r1 * q_priv[base + 0];
            if (base + 1 < RQ_D) score += r2 * q_priv[base + 1];
            if (base + 2 < RQ_D) score += r3 * q_priv[base + 2];
        }
        score *= norm_k * scale;

        if (mask_base != NULL) {
            score += slope * (float)((const global half*)(mask_base))[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    /* Reduce max across workgroup */
    __local float local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float m_final = local_m[0];

    /* Pass 2: softmax weights + V accumulation */
    float o_acc[RQ_D];
    for (int i = 0; i < RQ_D; i++) o_acc[i] = 0.0f;
    float l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global uchar * k_blk = (const global uchar*)(k_base + k_row_offset);
        const global uchar * v_blk = (const global uchar*)(v_base + v_row_offset);

        /* Recompute K score */
        float norm_k = vload_half(0, (const global half *)k_blk);
        float score = 0.0f;
        int idx_pos = 0;
        for (int g = 0; g < RQ_N_GROUPS; g++) {
            float qv1, qv2, qv3;
            for (int c = 0; c < 3; c++) {
                int pos = idx_pos + c;
                int byte_idx = 2 + pos / 2;
                uchar packed = (pos % 2 == 0) ? (k_blk[byte_idx] & 0x0F) : ((k_blk[byte_idx] >> 4) & 0x0F);
                float val = RQ_CENTROIDS[packed];
                if (c == 0) qv1 = val; else if (c == 1) qv2 = val; else qv3 = val;
            }
            idx_pos += 3;
            float rs = RQ_ROTORS[g*4], rb12 = RQ_ROTORS[g*4+1], rb13 = RQ_ROTORS[g*4+2], rb23 = RQ_ROTORS[g*4+3];
            float r1, r2, r3;
            cl3_inv_sandwich(rs, rb12, rb13, rb23, qv1, qv2, qv3, &r1, &r2, &r3);
            int base = g * 3;
            if (base + 0 < RQ_D) score += r1 * q_priv[base + 0];
            if (base + 1 < RQ_D) score += r2 * q_priv[base + 1];
            if (base + 2 < RQ_D) score += r3 * q_priv[base + 2];
        }
        score *= norm_k * scale;

        if (mask_base != NULL) {
            score += slope * (float)((const global half*)(mask_base))[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const float p = exp(score - m_final);
        l_i += p;

        /* Dequant V inline and accumulate — no temp array */
        float norm_v = vload_half(0, (const global half *)v_blk);
        float pn = p * norm_v;
        int vidx_pos = 0;
        for (int g = 0; g < RQ_N_GROUPS; g++) {
            float vq1, vq2, vq3;
            for (int c = 0; c < 3; c++) {
                int pos = vidx_pos + c;
                int byte_idx = 2 + pos / 2;
                uchar packed = (pos % 2 == 0) ? (v_blk[byte_idx] & 0x0F) : ((v_blk[byte_idx] >> 4) & 0x0F);
                float val = RQ_CENTROIDS[packed];
                if (c == 0) vq1 = val; else if (c == 1) vq2 = val; else vq3 = val;
            }
            vidx_pos += 3;
            float rs = RQ_ROTORS[g*4], rb12 = RQ_ROTORS[g*4+1], rb13 = RQ_ROTORS[g*4+2], rb23 = RQ_ROTORS[g*4+3];
            float vr1, vr2, vr3;
            cl3_inv_sandwich(rs, rb12, rb13, rb23, vq1, vq2, vq3, &vr1, &vr2, &vr3);
            int base = g * 3;
            if (base + 0 < RQ_D) o_acc[base + 0] += pn * vr1;
            if (base + 1 < RQ_D) o_acc[base + 1] += pn * vr2;
            if (base + 2 < RQ_D) o_acc[base + 2] += pn * vr3;
        }
    }

    /* Reduce l and o across workgroup */
    __local float local_l[Q1_WG_SIZE];
    __local float local_o[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float l_final = local_l[0];

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global float * o_row = (global float*)(o_base + o_row_offset);

    if (l_final > 0.0f) {
        const float l_inv = 1.0f / l_final;
        for (int i = 0; i < RQ_D; i++) {
            local_o[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o[tid] += local_o[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = local_o[0] * l_inv;
            }
        }
    } else if (tid == 0) {
        for (int i = 0; i < RQ_D; i++) o_row[i] = 0.0f;
    }
}
