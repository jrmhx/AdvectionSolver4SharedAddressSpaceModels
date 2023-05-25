#include <stdio.h>
#define REPs 1e6
__global__
void test()
{
  //this is a test function.
}

int main(void)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < REPs; ++i)
    test<<<1, 1>>>();

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Time (ms): %f\n", milliseconds/REPs);
}
