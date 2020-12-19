#include "input_params.h"

int count_setbits(int N) {
    int cnt = 0;
    while (N > 0) {
        cnt += N & 1;
        N = N >> 1;
    }
    return cnt;
}

bool hammingDistanceOne(const int *sequences, int i, int j) {
    int hamming = 0;
    // sum hamming distance for each sequence part
    for (int offset = 0; offset < SEQUENCE_LEN; offset++) {
        // increment by hamming distance of the parts
        hamming += count_setbits(sequences[i * SEQUENCE_LEN + offset] ^ sequences[j * SEQUENCE_LEN + offset]);
        //if greater than one fail fast
        if (hamming > 1) return false;
    }
    return hamming == 1;
}

void hammingWithCPU(const int *sequences, bool *pairs, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (hammingDistanceOne(sequences, i, j))
                pairs[i * count + j] = true;
        }
    }
}
