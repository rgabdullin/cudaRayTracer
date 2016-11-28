#define SYNC_AND_CHECK_CUDA_ERRORS {cudaDeviceSynchronize(); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); exit(1); }}

#include "RegularSampler.h"

#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>

__global__
extern void kernel(void);

int main()
{
	kernel <<< 1, 1 >>> ();
	SYNC_AND_CHECK_CUDA_ERRORS;

	system("pause");
	return 0;
}