#include "../ggml/src/ggml-rotorquant.c"
#include <stdio.h>
int main(void) {
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = (float)(i - 64) * 0.01f;

    block_rq4_128 blk;
    quantize_row_rq4_128_ref(x, &blk, 128);

    float y[128];
    dequantize_row_rq4_128(&blk, y, 128);

    printf("sizeof(block) = %zu\n", sizeof(block_rq4_128));
    printf("norm = %f\n", GGML_FP16_TO_FP32(blk.norm));
    printf("x[0]=%f y[0]=%f\n", x[0], y[0]);
    printf("x[64]=%f y[64]=%f\n", x[64], y[64]);

    float mse = 0, dot = 0, nx = 0, ny = 0;
    for (int i = 0; i < 128; i++) {
        float d = x[i] - y[i];
        mse += d*d;
        dot += x[i]*y[i];
        nx += x[i]*x[i];
        ny += y[i]*y[i];
    }
    printf("MSE=%.4f%% cos=%.6f\n", mse/nx*100, dot/sqrtf(nx*ny));
    return 0;
}
