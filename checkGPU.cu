#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
 
int main(void)
{
    cudaDeviceProp prop;
    int count;
    cudaError_t cudaStatus;
    cudaStatus = cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++) {
        cudaStatus = cudaGetDeviceProperties(&prop, i);
        printf("---General information for device %d ---\n", i);
        printf("name:%s\n", prop.name);
        printf("Max threads per block:%d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    return 0;
}