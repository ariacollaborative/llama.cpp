#include <stdio.h>
#include <math.h>
#include "ggml/include/ggml.h"
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

int main(void) {
    tq3_set_layer(0);
    // Force matrix init by calling a dummy quantize
    float dummy[128] = {0}; dummy[0] = 1.0f;
    block_tq4_128 dblock;
    quantize_row_tq4_128_ref(dummy, &dblock, 128);
    
    // Now read Pi_out and check Pi × Pi^T
    // Access through a test quantize/dequant roundtrip
    // Actually, let me just test: does rotation preserve norms?
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = (i % 7) * 0.1f - 0.3f;
    
    float norm_before = 0;
    for (int i = 0; i < 128; i++) norm_before += x[i] * x[i];
    
    // Quantize and immediately dequantize
    block_tq4_128 b;
    quantize_row_tq4_128_ref(x, &b, 128);
    float xd[128];
    dequantize_row_tq4_128(&b, xd, 128);
    
    float norm_after = 0, dot = 0, mse = 0;
    for (int i = 0; i < 128; i++) {
        norm_after += xd[i] * xd[i];
        dot += x[i] * xd[i];
        mse += (x[i]-xd[i])*(x[i]-xd[i]);
    }
    
    printf("norm_before=%.4f norm_after=%.4f ratio=%.4f\n",
        sqrtf(norm_before), sqrtf(norm_after), sqrtf(norm_after/norm_before));
    printf("cosine_sim=%.6f\n", dot/sqrtf(norm_before*norm_after));
    printf("MSE=%.4f (%.1f%%)\n", mse, 100*mse/norm_before);
    
    // Test with several vectors
    printf("\n5-vector quality test:\n");
    for (int v = 0; v < 5; v++) {
        for (int i = 0; i < 128; i++) x[i] = sinf(v*128+i) * (1.0f + 0.5f*cosf(i*0.1f));
        
        float nb = 0;
        for (int i = 0; i < 128; i++) nb += x[i]*x[i];
        
        quantize_row_tq4_128_ref(x, &b, 128);
        dequantize_row_tq4_128(&b, xd, 128);
        
        float na = 0, d = 0, m = 0;
        for (int i = 0; i < 128; i++) { na += xd[i]*xd[i]; d += x[i]*xd[i]; m += (x[i]-xd[i])*(x[i]-xd[i]); }
        
        printf("v%d: cos=%.6f mse=%.1f%% norm_ratio=%.4f\n",
            v, d/sqrtf(nb*na), 100*m/nb, sqrtf(na/nb));
    }
    return 0;
}
