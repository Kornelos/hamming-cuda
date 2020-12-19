#include <iostream>
#include <ctime>
#include "hamming_cpu.cpp"
#include "hamming_gpu.cu"
#include "input_params.h"
#include "sequence.cu"

int main() {
    srandom(0);
    const int count = SEQUENCE_COUNT;
    const int seqlen = SEQUENCE_LEN;
    int *seqs;
    bool *isPair;
    struct timespec start, end;
    cudaMallocManaged(&seqs, count * seqlen * sizeof(int));
    cudaMallocManaged(&isPair, count * count * sizeof(float));
    generateSequences(seqs, count);
#if DEBUG
    printGeneratedSeqs(seqs);
#endif

    std::cout << "----------------Running on CPU..----------------" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    hammingWithCPU(seqs, isPair, count);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CPU time: ", start, end);
    printPairs(isPair, seqs);
    resetPairs(isPair, count);

    std::cout << "----------------Running on CUDA..----------------" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    hammingWithCuda(seqs, isPair);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CUDA time: ", start, end);
    printPairs(isPair, seqs);

    //cleanup:
    checkCuda(cudaFree(seqs));
    checkCuda(cudaFree(isPair));

    return 0;
}
