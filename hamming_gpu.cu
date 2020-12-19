#include "input_params.h"

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
