#include <iostream>
#include <cstdint>
#include <random>

#define SEQUENCE_COUNT 2 // > 100000
#define SEQUENCE_LEN 2 // > 1000
#define PRINT_PAIRS true
#define FULL_MASK 0xffffffff

int randLL() {

    return random() % 3;
}

void generateSequences(int *sequences, const int count) {
    for (int i = 0; i < count * SEQUENCE_LEN; i++) {
        sequences[i] = randLL();
//        if(i < SEQUENCE_LEN*2-1){
//            sequences[i] = 1;
//        } else if(i == SEQUENCE_LEN*2){
//            sequences[i]=0;
//        }
    }
}

int count_setbits(int N) {
    int cnt = 0;
    while (N > 0) {
        cnt += N & 1;
        N = N >> 1;
    }
    return cnt;
}

int hammingDistance(const int *sequences, int i, int j) {
    int hamming = 0;
    // sum hamming distance for each sequence part
    for (int offset = 0; offset < SEQUENCE_LEN; offset++) {
        // increment by hamming distance of the parts
        hamming += count_setbits(sequences[i * SEQUENCE_LEN + offset] ^ sequences[j * SEQUENCE_LEN + offset]);
    }
    return hamming;
}

void hammingWithCPU(const int *sequences, bool *pairs, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = i; j < count; j++) {
            if (hammingDistance(sequences, i, j) == 1)
                pairs[i * count + j] = true;
        }
    }
}

void printPairCount(const bool *isPair) {
    int counter = 0;
    for (int i = 0; i < SEQUENCE_COUNT * SEQUENCE_COUNT; i++) {
        if (isPair[i]) {
            counter++;
        }
    }
    std::cout << "Pairs of hamming one count = " << counter << std::endl;
}

void printPairs(const bool *isPair, const int *seqs) {
    int counter = 0;
    for (int i = 0; i < SEQUENCE_COUNT * SEQUENCE_COUNT; i++) {
        if (isPair[i]) {
            counter++;
                #if PRINT_PAIRS
                unsigned int x = i / SEQUENCE_COUNT;
                unsigned int y = i - x * SEQUENCE_COUNT;
                printf("-------%d-%d--------\n", y, x);
                for (int j = 0; j < SEQUENCE_LEN; j++) printf("%d ", seqs[y * SEQUENCE_LEN + j]);
                printf("\n");
                for (int j = 0; j < SEQUENCE_LEN; j++) printf("%d ", seqs[x * SEQUENCE_LEN + j]);
                printf("\n----------------\n\n");
               #endif
            }
        }
    std::cout << "Pairs of hamming one count = " << counter << std::endl;

}
//////////////////////////// GPU
#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

__device__ int getDifference(int x) {
    if (!x) return 0;
    if (!(x & (x - 1))) return 1;
    return 2;
}

__global__ void hammingKernel(const int *seqs, const bool *pairs) {
    int result = 0;
    //initial values
    unsigned int i_one = 0;
    unsigned int j_one = gridDim.y;

    for (unsigned int i = i_one; i < gridDim.y * SEQUENCE_LEN + SEQUENCE_LEN; i++) {
        unsigned int j = gridDim.y + gridDim.x * SEQUENCE_LEN + i;
        printf("calculating xor of %d and %d values of i=%d j=%d\n",seqs[i],seqs[j],i,j);
        result += seqs[i] ^ seqs[j];
    }
    if (result == 1) {
        //its a pair yay
        pairs[i_one * SEQUENCE_COUNT + j_one];
    }
}


cudaError_t hammingWithCuda(const int *sequences, bool *pairs) {
    dim3 block(SEQUENCE_LEN, 2);
    dim3 grid(SEQUENCE_COUNT-1, SEQUENCE_COUNT-1);

    hammingKernel<<<grid, block>>>(sequences, pairs);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    return cudaSuccess;
}

void resetPairs(bool *pairs, int count) {
    for (int i = 0; i < count * count; i++)
        pairs[i] = false;
}


int main() {
    std::cout << "Hello, World!" << std::endl;
    srandom(0);
    const int count = SEQUENCE_COUNT;
    const int seqlen = SEQUENCE_LEN;
    int *seqs;
    bool *isPair;
    cudaMallocManaged(&seqs, count * seqlen * sizeof(int));
    cudaMallocManaged(&isPair, count * count * sizeof(float));

    generateSequences(seqs, count);

    /* run calculations */
    std::cout << "----------------Running on CPU..----------------" << std::endl;
    hammingWithCPU(seqs, isPair, count);
    printPairs(isPair,seqs);
    resetPairs(isPair, count);

    std::cout << "----------------Running on GPU..----------------" << std::endl;
    hammingWithCuda(seqs, isPair);
    printPairs(isPair,seqs);

    checkCuda(cudaFree(seqs));
    cudaFree(isPair);
    return 0;
}
