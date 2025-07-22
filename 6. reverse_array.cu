#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N - 1 - idx) { // no race condition in editing v.s. reading array because of this condition
        float tmp = input[idx];
        input[idx] = input[N - 1 - idx];
        input[N - 1 - idx] = tmp; 
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
