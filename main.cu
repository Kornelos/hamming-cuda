#include <iostream>
#include <cstdint>
#include <random>
#include <chrono>
#include <ctime>

#define SEQUENCE_COUNT 10000 // > 100000
#define SEQUENCE_LEN 2 // > 1000 bits
#define PRINT_PAIRS false
#define DEBUG false

int randWithBitMasking() {

    return (random() >> 20) & INT32_MAX;
}

void generateSequences(int *sequences, const int count) {
    for (int i = 0; i < count * SEQUENCE_LEN; i++) {
        sequences[i] = randWithBitMasking();

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

void printGeneratedSeqs(const int *seqs) {
    for (int i = 0; i < SEQUENCE_COUNT; i++) {

        for (int j = 0; j < SEQUENCE_LEN; j++) {
            std::cout << seqs[i * SEQUENCE_LEN + j] << " ";
        }
        std::cout << std::endl;
    }
}

void resetPairs(bool *pairs, int count) {
    for (int i = 0; i < count * count; i++)
        pairs[i] = false;
}

void print_timediff(const char* prefix, const struct timespec& start, const
struct timespec& end)
{
    double milliseconds = end.tv_nsec >= start.tv_nsec
                          ? (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3
                          : (start.tv_nsec - end.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec - 1) * 1e3;
    printf("%s: %lf milliseconds\n", prefix, milliseconds);
}

//////////////////////////// GPU
#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

__device__ bool hasOneBitSet(unsigned int n) {
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        if (count > 1) return false;
        n >>= 1;
    }
    return count == 1;
}

__device__ int count_setbits_dev(int N) {
    int cnt = 0;
    while (N > 0) {
        cnt += N & 1;
        N = N >> 1;
    }
    return cnt;
}

__global__ void hammingKernel(const int *seqs, bool *pairs) {
    unsigned int i = threadIdx.x + blockIdx.y * SEQUENCE_LEN;
    unsigned int j = (blockIdx.x + 1) * SEQUENCE_LEN + i;
    int hamming;

    if (j >= SEQUENCE_COUNT * SEQUENCE_LEN || i > j) {
        //discard threads that are out of bounds
        return;
    }

    hamming = count_setbits_dev(seqs[i] ^ seqs[j]);
#if DEBUG
    printf("BLOCK x=%d y=%d THREAD x=%d y=%d - calculating xor: %d ^ %d = %d values of i=%d j=%d\n",
            blockIdx.x, blockIdx.y,threadIdx.x,threadIdx.y, seqs[i], seqs[j], hamming, i, j);
#endif
    //sync all threads in block
    __syncthreads();
    //count all check where hamming == 1
    unsigned int vote_result = __ballot(hamming == 1);
    //look if there any results greater than one (not valid then)
    unsigned int hamming_greater = __ballot(hamming > 1);

    if (threadIdx.x == 0) {
        //aggregate solutions on first thread
#if DEBUG
        printf(
                "Pair: %d %d and %d %d hamming=%d vote=%d\n",
                seqs[blockIdx.y * SEQUENCE_LEN],
                seqs[blockIdx.y * SEQUENCE_LEN + 1],
                seqs[(blockIdx.x+blockIdx.y+1) * SEQUENCE_LEN],
                seqs[(blockIdx.x+blockIdx.y+1)* SEQUENCE_LEN + 1],
                hasOneBitSet(vote_result),
                vote_result
        );
#endif
        if (hasOneBitSet(vote_result) && hamming_greater == 0) {
            pairs[blockIdx.y * SEQUENCE_COUNT + blockIdx.y + blockIdx.x + 1] = true;
        }
    }
}


cudaError_t hammingWithCuda(const int *sequences, bool *pairs) {
    dim3 block(SEQUENCE_LEN, 1);
    dim3 grid(SEQUENCE_COUNT - 1, SEQUENCE_COUNT - 1);
    hammingKernel<<<grid, block>>>(sequences, pairs);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    return cudaSuccess;
}

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
