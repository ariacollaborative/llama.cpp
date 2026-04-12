/* Derive the correct Cl(3,0) multiplication table from first principles.
 *
 * Rules:
 *   e_i * e_i = +1  for i in {1,2,3}   (signature +,+,+)
 *   e_i * e_j = -e_j * e_i  for i != j  (anticommutation)
 *
 * Basis: [1, e1, e2, e3, e12, e13, e23, e123]
 *   e12 = e1*e2, e13 = e1*e3, e23 = e2*e3, e123 = e1*e2*e3
 *
 * For each pair (a_basis, b_basis), compute a_basis * b_basis
 * by counting transpositions needed to sort indices and applying e_i^2=+1.
 */

#include <stdio.h>
#include <string.h>

/* Basis elements as index lists */
/* 0: scalar (no indices)
 * 1: e1
 * 2: e2
 * 3: e3
 * 4: e12 = e1e2
 * 5: e13 = e1e3
 * 6: e23 = e2e3
 * 7: e123 = e1e2e3
 */

/* Multiply two basis elements by concatenating indices,
 * then bubble-sort to canonical order, counting sign flips,
 * and canceling adjacent identical indices (e_i^2 = +1). */
typedef struct {
    int sign;      /* +1 or -1 */
    int result;    /* index 0-7 of the result basis element */
} product_t;

product_t multiply_basis(int a_idx, int b_idx) {
    /* Index lists for each basis element */
    static const int basis_indices[8][3] = {
        {},           /* scalar */
        {1},          /* e1 */
        {2},          /* e2 */
        {3},          /* e3 */
        {1,2},        /* e12 */
        {1,3},        /* e13 */
        {2,3},        /* e23 */
        {1,2,3},      /* e123 */
    };
    static const int basis_len[8] = {0,1,1,1,2,2,2,3};

    /* Concatenate index lists */
    int indices[6];
    int n = 0;
    for (int i = 0; i < basis_len[a_idx]; i++) indices[n++] = basis_indices[a_idx][i];
    for (int i = 0; i < basis_len[b_idx]; i++) indices[n++] = basis_indices[b_idx][i];

    /* Bubble sort, counting transpositions */
    int sign = 1;
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < n-1; i++) {
            if (indices[i] > indices[i+1]) {
                int tmp = indices[i];
                indices[i] = indices[i+1];
                indices[i+1] = tmp;
                sign *= -1;
                changed = 1;
            } else if (indices[i] == indices[i+1]) {
                /* e_i^2 = +1 in Cl(3,0): remove both, sign unchanged */
                for (int j = i; j < n-2; j++) indices[j] = indices[j+2];
                n -= 2;
                changed = 1;
                break;
            }
        }
    }

    /* Map remaining indices to basis element */
    int result;
    if (n == 0) result = 0;        /* scalar */
    else if (n == 1 && indices[0] == 1) result = 1;  /* e1 */
    else if (n == 1 && indices[0] == 2) result = 2;  /* e2 */
    else if (n == 1 && indices[0] == 3) result = 3;  /* e3 */
    else if (n == 2 && indices[0] == 1 && indices[1] == 2) result = 4;  /* e12 */
    else if (n == 2 && indices[0] == 1 && indices[1] == 3) result = 5;  /* e13 */
    else if (n == 2 && indices[0] == 2 && indices[1] == 3) result = 6;  /* e23 */
    else if (n == 3) result = 7;   /* e123 */
    else { result = -1; }  /* shouldn't happen */

    product_t p = {sign, result};
    return p;
}

int main(void) {
    const char * names[8] = {"1", "e1", "e2", "e3", "e12", "e13", "e23", "e123"};

    printf("=== Cl(3,0) Multiplication Table (derived from axioms) ===\n\n");
    printf("       ");
    for (int j = 0; j < 8; j++) printf("%6s", names[j]);
    printf("\n");

    /* Build the full table and print */
    int table_sign[8][8];
    int table_result[8][8];

    for (int i = 0; i < 8; i++) {
        printf("%5s: ", names[i]);
        for (int j = 0; j < 8; j++) {
            product_t p = multiply_basis(i, j);
            table_sign[i][j] = p.sign;
            table_result[i][j] = p.result;
            printf(" %s%s", p.sign > 0 ? "+" : "-", names[p.result]);
        }
        printf("\n");
    }

    /* Now generate the corrected geometric product formulas */
    printf("\n=== Corrected r[k] formulas ===\n\n");
    for (int k = 0; k < 8; k++) {
        printf("r[%d] = /* %s */", k, names[k]);
        int first = 1;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (table_result[i][j] == k) {
                    if (first) {
                        printf("\n    %sa[%d]*b[%d]", table_sign[i][j] > 0 ? "" : "-", i, j);
                        first = 0;
                    } else {
                        printf(" %s a[%d]*b[%d]", table_sign[i][j] > 0 ? "+" : "-", i, j);
                    }
                }
            }
        }
        printf(";\n\n");
    }

    /* Compare against current implementation */
    printf("=== Comparison with current code (differences marked with ***) ===\n\n");

    /* Current r[7] formula from the code:
     * r[7] = a0*b7 + a7*b0 + a1*b6 - a6*b1 - a2*b5 + a5*b2 + a3*b4 - a4*b3
     *
     * Let's check each term against the derived table
     */
    printf("r[7] (e123) terms:\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (table_result[i][j] == 7) {
                printf("  a[%d]*b[%d] (%s*%s) = %s%s\n",
                       i, j, names[i], names[j],
                       table_sign[i][j] > 0 ? "+" : "-", names[7]);
            }
        }
    }

    return 0;
}
