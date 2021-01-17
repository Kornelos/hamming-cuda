#include <iostream>
#include <cstdint>
#include <random>
#include <chrono>
#include <ctime>

// control variables
#define SEQUENCE_COUNT 4000
#define SEQUENCE_LEN 8
#define PRINT_PAIRS false
#define DEBUG false

std::random_device rd;
std::mt19937_64 gen(rd());
std::uniform_int_distribution<uint64_t> dis;

uint64_t randWithBitMasking() {
    return (dis(gen) >> 62) & UINT64_MAX;
}

void generateSequences(uint64_t *sequences, const uint64_t count) {
    for (uint64_t i = 0; i < count * SEQUENCE_LEN; i++) {
        sequences[i] = randWithBitMasking();

    }
}

void printPairs(const bool *isPair, const uint64_t *seqs, const uint64_t count) {
    uint64_t counter = 0;
    for (uint64_t i = 0; i < count * count; i++) {
        if (isPair[i]) {
            counter++;
#if PRINT_PAIRS
            unsigned long x = i / SEQUENCE_COUNT;
            unsigned long y = i - x * SEQUENCE_COUNT;
            printf("-------%d-%d--------\n", y, x);
            for (long j = 0; j < SEQUENCE_LEN; j++) printf("%d ", seqs[y * SEQUENCE_LEN + j]);
            printf("\n");
            for (long j = 0; j < SEQUENCE_LEN; j++) printf("%d ", seqs[x * SEQUENCE_LEN + j]);
            printf("\n----------------\n\n");
#endif
        }
    }
    std::cout << "Pairs of hamming one count = " << counter << std::endl;

}

void printGeneratedSeqs(const uint64_t *seqs) {
    for (uint64_t i = 0; i < SEQUENCE_COUNT; i++) {

        for (uint64_t j = 0; j < SEQUENCE_LEN; j++) {
            std::cout << seqs[i * SEQUENCE_LEN + j] << " ";
        }
        std::cout << std::endl;
    }
}

void resetPairs(bool *pairs, uint64_t count) {
    for (long i = 0; i < count * count; i++)
        pairs[i] = false;
}

void print_timediff(const char *prefix, const struct timespec &start, const struct timespec &end) {
    double milliseconds = end.tv_nsec >= start.tv_nsec
                          ? (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3
                          : (start.tv_nsec - end.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec - 1) * 1e3;
    printf("%s: %lf milliseconds\n", prefix, milliseconds);
}

///////////////////////////// CPU

long count_setbits(uint64_t N) {
    long cnt = 0;
    while (N > 0) {
        cnt += N & 1;
        N = N >> 1;
    }
    return cnt;
}

bool hammingDistanceOne(const uint64_t *sequences, int i, int j) {
    long hamming = 0;
    // sum hamming distance for each sequence part
    for (long offset = 0; offset < SEQUENCE_LEN; offset++) {
        // increment by hamming distance of the parts
        hamming += count_setbits(sequences[i * SEQUENCE_LEN + offset] ^ sequences[j * SEQUENCE_LEN + offset]);
        //if greater than one fail fast
        if (hamming > 1) return false;
    }
    return hamming == 1;
}

void hammingWithCPU(const uint64_t *sequences, bool *pairs, long count) {
    for (long i = 0; i < count; i++) {
        for (long j = i + 1; j < count; j++) {
            if (hammingDistanceOne(sequences, i, j))
                pairs[i * count + j] = true;
        }
    }
}

//////////////////////////// GPU (NOT OPTIMAL)
#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

__device__ bool hasOneBitSet(uint64_t n) {
    unsigned long count = 0;
    while (n) {
        count += n & 1;
        if (count > 1) return false;
        n >>= 1;
    }
    return count == 1;
}

__device__ long count_setbits_dev(uint64_t N) {
    long cnt = 0;
    while (N > 0) {
        cnt += N & 1;
        N = N >> 1;
    }
    return cnt;
}

__global__ void hammingKernel(const uint64_t *seqs, bool *pairs) {
    unsigned int i = threadIdx.x + blockIdx.y * SEQUENCE_LEN;
    unsigned int j = (blockIdx.x + 1) * SEQUENCE_LEN + i;
    long hamming;

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


cudaError_t hammingWithCuda(const uint64_t *sequences, bool *pairs) {
    dim3 block(SEQUENCE_LEN, 1);
    dim3 grid(SEQUENCE_COUNT - 1, SEQUENCE_COUNT - 1);

    hammingKernel<<<grid, block>>>(sequences, pairs);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    return cudaSuccess;
}
/////////////////////////////////////////////// NEW GPU
__global__ void hammingKernelLin(const uint64_t *seqs, bool *pairs) {
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint threadId = blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    // first number copy for fast access
    uint64_t fst[SEQUENCE_LEN];
    for (uint i = 0; i < SEQUENCE_LEN; i++){
        fst[i] = seqs[threadId*SEQUENCE_LEN+i];
    }

    for(uint i = threadId+1; i<SEQUENCE_COUNT; i++){
        long hamming = 0;
        for(uint j=0;j<SEQUENCE_LEN;j++){
            hamming += count_setbits_dev(fst[j] ^ seqs[j+i*SEQUENCE_LEN]);
        }
        if(hamming == 1){
            pairs[threadId*SEQUENCE_COUNT+i] = true;
        }
    }
}

cudaError_t hammingWithCudaLin(const uint64_t *seqs, bool *pairs, uint64_t count) {
    dim3 block(32, 4);
    dim3 grid(block.x * block.y, ceil((double) SEQUENCE_COUNT / (block.x * block.y)));
    hammingKernelLin<<<grid, block>>>(seqs, pairs);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    return cudaSuccess;
}

int main() {
    srandom(0);
    const uint64_t count = SEQUENCE_COUNT;
    const uint64_t seqlen = SEQUENCE_LEN;
    uint64_t *seqs;
    bool *isPair;
    struct timespec start, end;
    cudaMallocManaged(&seqs, count * seqlen * sizeof(uint64_t));
    cudaMallocManaged(&isPair, count * count * sizeof(bool));
    generateSequences(seqs, count);
#if DEBUG
    printGeneratedSeqs(seqs);
#endif

    std::cout << "----------------Running on CPU..----------------" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    hammingWithCPU(seqs, isPair, count);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CPU time: ", start, end);
    printPairs(isPair, seqs, count);
    resetPairs(isPair, count);

    std::cout << "----------------Running on CUDA (Not optimal, thread per two vectors)..----------------" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    hammingWithCuda(seqs, isPair);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CUDA time: ", start, end);
    printPairs(isPair, seqs, count);
    resetPairs(isPair, count);

    std::cout << "------------- New CUDA (thread for one vector comparing with all others).. ------------" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    hammingWithCudaLin(seqs, isPair,count);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CUDA time: ", start, end);
    printPairs(isPair, seqs, count);
    //cleanup:
    checkCuda(cudaFree(seqs));
    checkCuda(cudaFree(isPair));

    return 0;
}
