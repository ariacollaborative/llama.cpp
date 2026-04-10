#include <stdio.h>
#include <math.h>
#include "ggml/include/ggml.h"
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

int main(void) {
    float x[256], xd[256];  // q4_0 block size = 32, test 8 blocks = 256
    
    printf("q4_0 quality (block_size=32):\n");
    for (int v = 0; v < 5; v++) {
        for (int i = 0; i < 256; i++) x[i] = sinf(v*256+i) * (1.0f + 0.5f*cosf(i*0.1f));
        
        // q4_0: quantize 256 elements = 8 blocks of 32
        block_q4_0 blocks[8];
        quantize_row_q4_0_ref(x, blocks, 256);
        dequantize_row_q4_0(blocks, xd, 256);
        
        // Measure first 128 elements (one head)
        float nb=0,na=0,d=0,m=0;
        for (int i=0;i<128;i++){nb+=x[i]*x[i]; na+=xd[i]*xd[i]; d+=x[i]*xd[i]; m+=(x[i]-xd[i])*(x[i]-xd[i]);}
        printf("v%d: cos=%.6f mse=%.1f%% norm_ratio=%.4f\n", v, d/sqrtf(nb*na), 100*m/nb, sqrtf(na/nb));
    }
    return 0;
}
