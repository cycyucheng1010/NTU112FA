#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 #include<time.h>
#include <stdio.h>
 
__global__ void add(const int a, const int b, int *c)
{
        *c = a + b;
}
 
int main()
{
        clock_t start_t,finish_t;
        double total_t = 0;
        start_t = clock();
        int c;
        int *dev_c; // 定义在设备端的接收数据的指针
        cudaError_t cudaStatus;
        //为输入参数和输出参数分配内存
        cudaStatus = cudaMalloc((void**)&dev_c, sizeof(int));
        if (cudaStatus != cudaSuccess) {
                printf("cudaMalloc is failed!\n");
        }
        add<<<1, 1 >>>(2, 7, dev_c);
        cudaStatus = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
                printf(" cudaMemcpyDeviceToHost is failed!\n");
        }
        cudaFree(dev_c);
        printf("2+7=%d\n", c);
        finish_t = clock();

        total_t = (double)(finish_t - start_t) / CLOCKS_PER_SEC;//将时间转换为秒
        printf("CPU 占用的总时间：%f\n", total_t);
        return 0;
}