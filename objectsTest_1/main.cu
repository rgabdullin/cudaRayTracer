#include <cstdio>
#include "Objects.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERRORS {cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); return 1; }}

int main(void) {
	Object** dev_array;
	int n = 8;
	cudaMalloc((void**)&dev_array, sizeof(Object*) * n);
	test::GPUCreateTestObjectArray<<<1,n>>>(dev_array);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERRORS;
	
	test::kernel <<< 1, n >>> (dev_array);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERRORS;

	test::GPUDestroyTestObjectArray<<<1,n>>>(dev_array);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERRORS;

	cudaFree(dev_array);
	CHECK_CUDA_ERRORS;

	system("pause");
	return 0;
}