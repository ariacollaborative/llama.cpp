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

constant float RQ_ROTORS[172] = {
    0.9976641204f, 0.0088128785f, -0.0490202330f, 0.0467509650f,
    0.0107257757f, -0.1211351076f, -0.4055028661f, 0.9059683598f,
    -0.9959692948f, 0.0677697679f, -0.0417615368f, 0.0413327516f,
    0.1681031348f, -0.6755748539f, -0.6980518226f, 0.1675219561f,
    -0.9694066010f, 0.0475489135f, -0.2283702535f, 0.0764000654f,
    0.3212832093f, -0.1365998823f, -0.9289582715f, -0.1231019960f,
    -0.9186392699f, -0.1255857821f, -0.2485945356f, -0.2802335811f,
    0.4664413211f, 0.1196258120f, -0.6079652520f, -0.6312688899f,
    -0.8449348860f, -0.4745880345f, -0.2366849729f, -0.0695087012f,
    0.5999530886f, 0.3359301662f, -0.3117833299f, -0.6557426096f,
    -0.7501337378f, -0.4956262850f, -0.2980700411f, -0.3206371962f,
    0.7184849221f, -0.1913920873f, -0.2409403222f, -0.6237757985f,
    -0.6366028688f, 0.6214519750f, -0.3282245109f, -0.3174947255f,
    0.8190772585f, -0.5710723276f, -0.0440310195f, 0.0324054115f,
    -0.5071769753f, 0.4393236738f, -0.7171422749f, 0.1883432583f,
    0.8992184570f, -0.1160676651f, -0.4067419465f, 0.1117830612f,
    -0.3650876291f, -0.0082612127f, -0.5526616926f, 0.7491380574f,
    0.9569075114f, 0.1489264954f, -0.1702868216f, 0.1820750173f,
    -0.2138825891f, -0.2888408175f, -0.2510066215f, 0.8987885715f,
    0.9907040122f, 0.1238467781f, -0.0208349138f, 0.0522823312f,
    -0.0573372198f, -0.9691430651f, -0.1458082924f, 0.1903000381f,
    -0.9997641110f, 0.0182636847f, 0.0058326909f, 0.0102048960f,
    0.1006397744f, 0.8908453442f, 0.2867781727f, -0.3376751221f,
    -0.9838615907f, -0.1044625008f, 0.0390071086f, -0.1399371350f,
    0.2561039435f, 0.4611684143f, 0.3774039463f, -0.7611180757f,
    -0.9433935135f, -0.3095525509f, 0.1109823920f, -0.0432296833f,
    0.4051735790f, 0.2992885234f, 0.3761443240f, -0.7776735808f,
    -0.8793703070f, 0.0277718504f, 0.3773616404f, -0.2890238395f,
    0.5441266344f, -0.5709750376f, 0.3841791531f, -0.4799167537f,
    -0.7933905352f, 0.4857406185f, 0.3587285566f, -0.0768201337f,
    0.6694936591f, -0.2949431078f, 0.5447410031f, 0.4099317542f,
    -0.6876009847f, 0.5311441953f, 0.3492735954f, 0.3508542222f,
    0.7781444254f, -0.1064620280f, 0.5093771207f, 0.3516987899f,
    -0.5646430629f, -0.0643410379f, 0.2910266862f, 0.7696375188f,
    0.8673660856f, 0.3966456711f, 0.1537539547f, 0.2582789315f,
    -0.4275868458f, -0.3988423202f, 0.1739822785f, 0.7923537466f,
    0.9349309081f, 0.0581518918f, 0.1258534226f, 0.3266243570f,
    -0.2798544227f, -0.1683942363f, 0.9445448795f, 0.0340566287f,
    0.9791518998f, -0.0156263342f, 0.2020227948f, -0.0142886346f,
    -0.1251344519f, 0.7034746218f, 0.6870425573f, -0.1320505581f,
    0.9989249287f, -0.0248887524f, 0.0286992127f, -0.0265686294f,
    0.0327099395f, 0.4154252165f, 0.7251529338f, -0.5481835203f,
    -0.9937562915f, -0.0867442401f, 0.0688555934f, -0.0135195089f,
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
