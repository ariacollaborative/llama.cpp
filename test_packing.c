#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Copy the exact packing functions from tq3_impl.inc
static void tq_pack_3bit(const uint8_t * idx, uint8_t * packed, int n) {
    int n_groups = n / 8;
    for (int g = 0; g < n_groups; g++) {
        const uint8_t * in = idx + g*8;
        uint8_t * out = packed + g*3;
        out[0] = (in[0]&7) | ((in[1]&7)<<3) | ((in[2]&3)<<6);
        out[1] = ((in[2]>>2)&1) | ((in[3]&7)<<1) | ((in[4]&7)<<4) | ((in[5]&1)<<7);
        out[2] = ((in[5]>>1)&3) | ((in[6]&7)<<2) | ((in[7]&7)<<5);
    }
}

static void tq_unpack_3bit(const uint8_t * packed, uint8_t * idx, int n) {
    int n_groups = n / 8;
    for (int g = 0; g < n_groups; g++) {
        const uint8_t * in = packed + g*3;
        uint8_t * out = idx + g*8;
        out[0] = in[0] & 7;
        out[1] = (in[0]>>3) & 7;
        out[2] = ((in[0]>>6)&3) | ((in[1]&1)<<2);
        out[3] = (in[1]>>1) & 7;
        out[4] = (in[1]>>4) & 7;
        out[5] = ((in[1]>>7)&1) | ((in[2]&3)<<1);
        out[6] = (in[2]>>2) & 7;
        out[7] = (in[2]>>5) & 7;
    }
}

int main(void) {
    // Test with known values
    uint8_t indices[64], packed[24], unpacked[64];
    int errors = 0;
    
    // Fill with all possible 3-bit values
    for (int i = 0; i < 64; i++) indices[i] = i % 8;
    
    tq_pack_3bit(indices, packed, 64);
    tq_unpack_3bit(packed, unpacked, 64);
    
    for (int i = 0; i < 64; i++) {
        if (indices[i] != unpacked[i]) {
            printf("MISMATCH at %d: packed %d, got %d\n", i, indices[i], unpacked[i]);
            errors++;
        }
    }
    
    // Test with random values
    srand(42);
    for (int t = 0; t < 100; t++) {
        for (int i = 0; i < 64; i++) indices[i] = rand() % 8;
        tq_pack_3bit(indices, packed, 64);
        tq_unpack_3bit(packed, unpacked, 64);
        for (int i = 0; i < 64; i++) {
            if (indices[i] != unpacked[i]) { errors++; break; }
        }
    }
    
    // Test 32-element packing too
    for (int i = 0; i < 32; i++) indices[i] = i % 8;
    tq_pack_3bit(indices, packed, 32);
    tq_unpack_3bit(packed, unpacked, 32);
    for (int i = 0; i < 32; i++) {
        if (indices[i] != unpacked[i]) {
            printf("32-elem MISMATCH at %d: packed %d, got %d\n", i, indices[i], unpacked[i]);
            errors++;
        }
    }
    
    // Test 2-bit packing (for regular group)
    printf("2-bit packing: ");
    uint8_t idx2[96], packed2[24], unpacked2[96];
    for (int i = 0; i < 96; i++) idx2[i] = i % 4;
    for (int j = 0; j < 96/4; j++)
        packed2[j] = (idx2[4*j]) | (idx2[4*j+1]<<2) | (idx2[4*j+2]<<4) | (idx2[4*j+3]<<6);
    for (int j = 0; j < 96/4; j++) {
        unpacked2[4*j] = packed2[j] & 3;
        unpacked2[4*j+1] = (packed2[j]>>2) & 3;
        unpacked2[4*j+2] = (packed2[j]>>4) & 3;
        unpacked2[4*j+3] = (packed2[j]>>6) & 3;
    }
    int e2 = 0;
    for (int i = 0; i < 96; i++) if (idx2[i] != unpacked2[i]) e2++;
    printf("%s (%d errors)\n", e2 ? "FAIL" : "OK", e2);
    
    // Test 1-bit packing
    printf("1-bit packing: ");
    uint8_t idx1[96], packed1[12], unpacked1[96];
    for (int i = 0; i < 96; i++) idx1[i] = i % 2;
    for (int j = 0; j < 96/8; j++) {
        uint8_t byte = 0;
        for (int b = 0; b < 8; b++) byte |= (idx1[8*j+b] << b);
        packed1[j] = byte;
    }
    for (int j = 0; j < 96/8; j++)
        for (int b = 0; b < 8; b++)
            unpacked1[8*j+b] = (packed1[j] >> b) & 1;
    int e1 = 0;
    for (int i = 0; i < 96; i++) if (idx1[i] != unpacked1[i]) e1++;
    printf("%s (%d errors)\n", e1 ? "FAIL" : "OK", e1);
    
    printf("\n3-bit packing: %s (%d errors across 101 tests)\n", errors ? "FAIL" : "OK", errors);
    return errors;
}
