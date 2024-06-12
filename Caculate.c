#include<stdio.h>
#include<time.h>
void add(const int a,const int b, int c)
{
    c = a+b;
}

int main()
{
    clock_t start_t,finish_t;
    double total_t = 0;
    start_t = clock();
    int c=0;
    add(2,7,c);
    printf("2+7=%d\n", c);
    finish_t = clock();
    total_t = (double)(finish_t - start_t) / CLOCKS_PER_SEC;//将时间转换为秒
    printf("CPU 占用的总时间：%f\n", total_t);
    return 0;
}