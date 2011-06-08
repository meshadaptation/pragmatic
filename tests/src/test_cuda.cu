#include <stdio.h>
#include <cuda.h>
#include <iostream>
using std::cout;
using std::endl;

__global__ void square(float *a, int N)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId < N)
    a[threadId] = a[threadId] * a[threadId];
}

int main(void)
{
  float *h_a, *d_a;
  const int N = 50;
  size_t size = N * sizeof(float);
  h_a = (float *) malloc(size);
  cudaMalloc((void **) &d_a, size);

  for(int i = 0; i < N; i++)
    h_a[i] = (float) i;

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

  int block_size = 4;
  int n_blocks = N / block_size + (N % block_size == 0 ? 0 : 1);
  square<<<n_blocks, block_size>>>(d_a, N);

  cudaMemcpy(h_a, d_a, sizeof(float)*N, cudaMemcpyDeviceToHost);

  bool pass = true;

  for(int i = 0; i < N; i++)
    if (h_a[i] != (i * i)) pass = false;

  if (pass)
    cout << "pass" << endl;
  else
    cout << "fail" << endl;

  free(h_a);
  cudaFree(d_a);
}
