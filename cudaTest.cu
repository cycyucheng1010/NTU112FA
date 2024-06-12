#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    if (device_count == 0) {
        printf("No CUDA devices found\n");
    } else {
        printf("Number of CUDA devices: %d\n", device_count);
    }
    
    return 0;
}