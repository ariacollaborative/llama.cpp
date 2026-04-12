#include "../ggml/src/ggml-rotorquant.c"
#include <stdio.h>
/* Quick test: 4-bit codebook with corrected geometric product */
static const float cb16[16] = {
    -0.2415722564f, -0.1829257937f, -0.1430662130f, -0.1110830968f,
    -0.0833324871f, -0.0580811446f, -0.0343186837f, -0.0113553670f,
     0.0113553670f,  0.0343186837f,  0.0580811446f,  0.0833324871f,
     0.1110830968f,  0.1430662130f,  0.1829257937f,  0.2415722564f,
};
int main(void) {
    rq_init();
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = (float)(i - 64) * 0.01f;
    float norm_sq = 0; for (int i = 0; i < 128; i++) norm_sq += x[i]*x[i];
    float norm = sqrtf(norm_sq), inv = 1.0f/norm;
    float xr[128]; memset(xr, 0, sizeof(xr));
    float rsq = 0;
    int ng = 43;
    for (int g = 0; g < ng; g++) {
        float mv[8]={0}; int b=g*3;
        mv[1]=(b+0<128)?x[b+0]*inv:0; mv[2]=(b+1<128)?x[b+1]*inv:0; mv[3]=(b+2<128)?x[b+2]*inv:0;
        float rot[8]; cl3_rotor_sandwich(rq_rotors[g], mv, rot);
        for (int c=0;c<3;c++) {
            int best=0; float bd=fabsf(rot[1+c]-cb16[0]);
            for(int i=1;i<16;i++){float d=fabsf(rot[1+c]-cb16[i]);if(d<bd){bd=d;best=i;}}
            rot[1+c]=cb16[best]; rsq+=cb16[best]*cb16[best];
        }
        float back[8]; cl3_rotor_sandwich_inverse(rq_rotors[g], rot, back);
        if(b+0<128) xr[b+0]=back[1]; if(b+1<128) xr[b+1]=back[2]; if(b+2<128) xr[b+2]=back[3];
    }
    float cn=norm/sqrtf(rsq);
    float mse=0,dot=0,nx=0,ny=0;
    for(int i=0;i<128;i++){float r=xr[i]*cn,d=x[i]-r;mse+=d*d;dot+=x[i]*r;nx+=x[i]*x[i];ny+=r*r;}
    printf("4-bit + corrected product: MSE=%.4f%% cos=%.6f\n", mse/nx*100, dot/sqrtf(nx*ny));
    return 0;
}
