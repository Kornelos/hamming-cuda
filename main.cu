#include <iostream>
#include <cstdint>
#define SEQUENCE_COUNT 100 // > 100000
#define WARP_SIZE 32
#define SEQUENCE_LEN WARP_SIZE // > 1000
#define COMB_ARR_SZ(count) (count * count * sizeof(bool))
#define SEQUENCE_BITS_COUNT (SEQUENCE_LEN * 64)
#define BUCKETS_COUNT (SEQUENCE_BITS_COUNT + 1)

#define FULL_MASK 0xffffffff
uint64_t randLL() {
    uint64_t r = 0;
    for (int i = 0; i < 5; ++i) {
        r = (r << 15) | (rand() & 0x7FFF);
    }
    return r & 0xFFFFFFFFFFFFFFFFULL;
}

void generateSequences(uint64_t *sequences, const int count) {
    for (int i = 0; i < count * SEQUENCE_LEN; i++) {
        sequences[i] = randLL();
//        if (i / SEQUENCE_LEN == 1 || i / SEQUENCE_LEN == 7 || i / SEQUENCE_LEN == 32 || i / SEQUENCE_LEN == 411) {
//            sequences[i] = 0;
//        }
//        if ((i / SEQUENCE_LEN == 411 || i / SEQUENCE_LEN == 7) && i % SEQUENCE_LEN == 3) {
//            sequences[i] = 1;
//        }
    }
//    sequences[0] = 1;
//    sequences[1] = 0;
    //sequences[SEQUENCE_LEN * 7 + 6] = 1;
}

int count_setbits(uint64_t N){
    int cnt=0;
    while(N>0){
        cnt+=(N&1);
        N=N>>1;
    }
    return cnt;
}

int hammingDistance(const uint64_t *sequences, int i, int j) {
    int hamming = 0;
    // sum hamming distance for each sequence part
    for (int offset = 0; offset < SEQUENCE_LEN; offset++) {
        // increment by hamming distance of the parts
        hamming += count_setbits(sequences[i * SEQUENCE_LEN + offset] ^ sequences[j * SEQUENCE_LEN + offset]);
    }
    return hamming;
}

void hammingWithCPU(const uint64_t *sequences, bool *pairs, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = i; j < count; j++) {
            if (hammingDistance(sequences, i, j) == 1)
                pairs[i * count + j] = true;
        }
    }
}

void printPairCount(const bool* isPair){
    int counter = 0;
    for (int i = 0; i < SEQUENCE_COUNT; i++){
        if(isPair[i]){
            counter++;
        }
    }
    std::cout<<"Pairs of hamming one count = "<<counter<<std::endl;
}
//////////////////////////// GPU
void checkCuda(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n",line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

__global__ void hammingKernel(const uint64_t *sequences, bool *pairs, const unsigned int count)
{
    unsigned int i = blockIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < count && j < i) {
        // calculate hamming distance of the part of the bit sequence
        unsigned int hamming = __popcll(sequences[i*WARP_SIZE + threadIdx.x] ^ sequences[j*WARP_SIZE + threadIdx.x]);
        // check if any of the warp threads return hamming distance > 1
        unsigned int exceed = __any_sync(FULL_MASK,hamming > 1);
        // number of warp threads that return hamming distance = 1
        unsigned int ones = __popc(__ballot_sync(FULL_MASK, hamming == 1));
        // make aggreagation in leader thread
        __syncwarp(FULL_MASK);
        if (threadIdx.x == 0) {
            // set pair to true if none of the warp threads returned distance > 1 and only one returned distance = 1
            pairs[i * count + j] = !exceed && (ones == 1);
        }
    }
}


cudaError_t hammingWithCuda(const uint64_t *sequences, bool *pairs, unsigned int count)
{
    dim3 block(32, 32);
    dim3 grid(count, (count + block.y - 1) / block.y);

    uint64_t *devSequences = 0;
    bool *devPairs = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    //checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for vector
    checkCuda(cudaMalloc((void**)&devSequences, count * SEQUENCE_LEN * sizeof(uint64_t)),__LINE__);
    checkCuda(cudaMalloc((void**)&devPairs, COMB_ARR_SZ(count)),__LINE__);

    // Copy input vector from host memory to GPU buffers.
    checkCuda(cudaMemcpy(devSequences, sequences, count * SEQUENCE_LEN * sizeof(uint64_t), cudaMemcpyHostToDevice),__LINE__);

    // Launch kernel
    hammingKernel<<<grid, block>>>(devSequences, devPairs, count);
    checkCuda(cudaGetLastError(),__LINE__);
    checkCuda(cudaDeviceSynchronize(),__LINE__);

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(pairs, devPairs, COMB_ARR_SZ(count), cudaMemcpyDeviceToHost),__LINE__);

    cudaFree(devSequences);
    cudaFree(devPairs);
    return cudaSuccess;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    srand(0);

    auto* seqs = new uint64_t[SEQUENCE_COUNT * SEQUENCE_LEN];
    auto* isPair = new bool[SEQUENCE_COUNT * SEQUENCE_LEN];
    generateSequences(seqs, SEQUENCE_COUNT);

    /* run calculations */
    std::cout << "Running on CPU.."<<std::endl;
    hammingWithCPU(seqs,isPair,SEQUENCE_COUNT);
    printPairCount(isPair);

//    std::cout << "Running on GPU.."<<std::endl;
//    hammingWithCuda(seqs,isPair,SEQUENCE_COUNT);
//    printPairCount(isPair);

    free(seqs);
    free(isPair);
    return 0;
}
