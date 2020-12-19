#include <iostream>
#include <cstdint>
#include <random>

#define SEQUENCE_COUNT 3 // > 100000
#define SEQUENCE_LEN 2 // > 1000
#define PRINT_PAIRS true
#define DEBUG false
//define FULL_MASK 0xffffffff

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
//bool hasOneBitSet(unsigned int n){
//    unsigned int count = 0;
//    while (n) {
//        count += n & 1;
//        if(count > 1) return false;
//        n >>= 1;
//    }
//    return count == 1;
//}

bool hammingDistanceOne(const int *sequences, int i, int j) {
    int hamming = 0;
    // sum hamming distance for each sequence part
    for (int offset = 0; offset < SEQUENCE_LEN; offset++) {
        // increment by hamming distance of the parts
        hamming += count_setbits(sequences[i * SEQUENCE_LEN + offset] ^ sequences[j * SEQUENCE_LEN + offset]);
        //if greater than one fail fast
        if(hamming > 1) return false;
    }
    return hamming == 1;
}

void hammingWithCPU(const int *sequences, bool *pairs, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = i+1; j < count; j++) {
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
//////////////////////////// GPU
#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

__device__ bool hasOneBitSet(unsigned int n)
{
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        if(count > 1) return false;
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
    unsigned int j = (blockIdx.x+1) * SEQUENCE_LEN + i;
    int hamming;

    if (j >= SEQUENCE_COUNT * SEQUENCE_LEN || i > j) {
        return;
    } else {
        hamming = count_setbits_dev(seqs[i] ^ seqs[j]);
#if DEBUG
        printf("BLOCK x=%d y=%d THREAD x=%d y=%d - calculating xor: %d ^ %d = %d values of i=%d j=%d\n",
                blockIdx.x, blockIdx.y,threadIdx.x,threadIdx.y, seqs[i], seqs[j], hamming, i, j);
#endif
    }

    //sync all threads in block
    __syncthreads();
    //count all check where hamming == 1
    unsigned int vote_result = __ballot(hamming == 1);
    //look if there any results greater than one (not valid then)
    unsigned int hamming_greater = __ballot(hamming > 1);

    if (threadIdx.x == 0) {
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
            pairs[blockIdx.y * SEQUENCE_COUNT + blockIdx.y+blockIdx.x+1] = true;
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

void resetPairs(bool *pairs, int count) {
    for (int i = 0; i < count * count; i++)
        pairs[i] = false;
}


int main() {
    srandom(0);
    const int count = SEQUENCE_COUNT;
    const int seqlen = SEQUENCE_LEN;
    int *seqs;
    bool *isPair;
    cudaMallocManaged(&seqs, count * seqlen * sizeof(int));
    cudaMallocManaged(&isPair, count * count * sizeof(float));

    generateSequences(seqs, count);
    printGeneratedSeqs(seqs);


    std::cout << "----------------Running on CPU..----------------" << std::endl;
    hammingWithCPU(seqs, isPair, count);
    printPairs(isPair, seqs);
    resetPairs(isPair, count);

    std::cout << "----------------Running on CUDA..----------------" << std::endl;
    hammingWithCuda(seqs, isPair);
    printPairs(isPair, seqs);

    //cleanup:
    checkCuda(cudaFree(seqs));
    checkCuda(cudaFree(isPair));

    return 0;
}
