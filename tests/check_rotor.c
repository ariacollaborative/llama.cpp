#include "../ggml/src/ggml-rotorquant.c"
#include <stdio.h>
int main(void) {
    rq_init();
    /* Check R·R̃ = 1 for first 5 rotors */
    for (int g = 0; g < 5; g++) {
        float rev[8], rrt[8];
        cl3_reverse(rq_rotors[g], rev);
        cl3_geometric_product(rq_rotors[g], rev, rrt);
        printf("Rotor %d: R·R̃ = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
               g, rrt[0], rrt[1], rrt[2], rrt[3], rrt[4], rrt[5], rrt[6], rrt[7]);
    }
    /* Check sandwich round-trip */
    float v[8] = {0, 0.5f, -0.3f, 0.7f, 0, 0, 0, 0};
    float rot[8], back[8];
    cl3_rotor_sandwich(rq_rotors[0], v, rot);
    cl3_rotor_sandwich_inverse(rq_rotors[0], rot, back);
    printf("\nRound-trip: [%.6f,%.6f,%.6f] -> [%.6f,%.6f,%.6f] -> [%.6f,%.6f,%.6f]\n",
           v[1], v[2], v[3], rot[1], rot[2], rot[3], back[1], back[2], back[3]);
    printf("Trivector after sandwich: %.2e\n", rot[7]);
    return 0;
}
