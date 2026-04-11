#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ggml/include/ggml.h"
#include "ggml/src/ggml-common.h"
#include "ggml/src/ggml-quants.h"

int main(void) {
    const int N=128, NKEYS=50;
    tq3_set_layer(0);
    srand(0);
    
    float keys[50][128], query[128];
    for(int i=0;i<NKEYS;i++) for(int j=0;j<N;j++){
        float u1=(float)(rand()+1)/((float)__INT_MAX__+1);
        float u2=(float)(rand()+1)/((float)__INT_MAX__+1);
        keys[i][j]=sqrtf(-2*logf(u1))*cosf(6.2832f*u2);
    }
    for(int j=0;j<N;j++){
        float u1=(float)(rand()+1)/((float)__INT_MAX__+1);
        float u2=(float)(rand()+1)/((float)__INT_MAX__+1);
        query[j]=sqrtf(-2*logf(u1))*cosf(6.2832f*u2);
    }
    
    // Quantize
    block_tq4_128 blocks[50];
    for(int i=0;i<NKEYS;i++) quantize_row_tq4_128_ref(keys[i],&blocks[i],N);
    
    // Dequantize WITH QJL
    float kd_qjl[50][128];
    for(int i=0;i<NKEYS;i++) dequantize_row_tq4_128(&blocks[i],kd_qjl[i],N);
    
    // Dequantize WITHOUT QJL: zero out resnorm so QJL correction = 0
    block_tq4_128 blocks_noqjl[50];
    for(int i=0;i<NKEYS;i++){
        blocks_noqjl[i]=blocks[i];
        blocks_noqjl[i].resnorm=0; // zeroed resnorm disables QJL correction
    }
    float kd_mse[50][128];
    for(int i=0;i<NKEYS;i++) dequantize_row_tq4_128(&blocks_noqjl[i],kd_mse[i],N);
    
    // Compare
    float ref[50], dot_qjl[50], dot_mse[50];
    for(int i=0;i<NKEYS;i++){
        float r=0,dq=0,dm=0;
        for(int j=0;j<N;j++){r+=keys[i][j]*query[j]; dq+=kd_qjl[i][j]*query[j]; dm+=kd_mse[i][j]*query[j];}
        ref[i]=r; dot_qjl[i]=dq; dot_mse[i]=dm;
    }
    
    // Correlation
    float mr=0,mq=0,mm=0;
    for(int i=0;i<NKEYS;i++){mr+=ref[i];mq+=dot_qjl[i];mm+=dot_mse[i];}
    mr/=NKEYS;mq/=NKEYS;mm/=NKEYS;
    float cov_q=0,cov_m=0,vr=0,vq=0,vm=0;
    for(int i=0;i<NKEYS;i++){
        float dr=ref[i]-mr;
        cov_q+=dr*(dot_qjl[i]-mq); vq+=(dot_qjl[i]-mq)*(dot_qjl[i]-mq);
        cov_m+=dr*(dot_mse[i]-mm); vm+=(dot_mse[i]-mm)*(dot_mse[i]-mm);
        vr+=dr*dr;
    }
    printf("MSE+QJL correlation: %.4f\n",cov_q/sqrtf(vr*vq));
    printf("MSE-only correlation: %.4f\n",cov_m/sqrtf(vr*vm));
    
    // MSE comparison
    float mse_q=0,mse_m=0,energy=0;
    for(int i=0;i<NKEYS;i++) for(int j=0;j<N;j++){
        float eq=keys[i][j]-kd_qjl[i][j]; mse_q+=eq*eq;
        float em=keys[i][j]-kd_mse[i][j]; mse_m+=em*em;
        energy+=keys[i][j]*keys[i][j];
    }
    printf("MSE+QJL: %.1f%%\n",100*mse_q/energy);
    printf("MSE-only: %.1f%%\n",100*mse_m/energy);
    return 0;
}
