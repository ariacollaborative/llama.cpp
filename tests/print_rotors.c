#include "../ggml/src/ggml-rotorquant.c"
#include <stdio.h>
int main(void) {
    rq_init();
    printf("constant float RQ_ROTORS[172] = {\n");
    for (int g = 0; g < 43; g++) {
        printf("    %.10ff, %.10ff, %.10ff, %.10ff,\n",
               rq_rotors[g][0], rq_rotors[g][4], rq_rotors[g][5], rq_rotors[g][6]);
    }
    printf("};\n");
    return 0;
}
