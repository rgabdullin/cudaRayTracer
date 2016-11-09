
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdio>

#define CHECK_CUDA_ERRORS {cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); return 1; }}

typedef unsigned long long ull;

__global__ void addKernel(int *c, const int *a, const int *b)
{
}

int main()
{
	cudaDeviceProp props;
	
	cudaGetDeviceProperties(&props, 0);

	ull memory_size = props.totalGlobalMem;

	printf("%s, Memory size: %.1f Gbs\n", props.name, float(memory_size) / (1024 * 1024 * 1024));

	ull num_of_floats =  1024 * 1024 * 1024;
	
	printf("Allocating memory to %llu elements(%.1f Gbs)\n", num_of_floats, float(num_of_floats * 4) / (1024 * 1024 * 1024));
	float* array = new float[num_of_floats];
	printf("Initializing array\n");
	for (int i = 0; i < num_of_floats; ++i) array[i] = i + 1;
	
	float* dev_array;
	printf("Allocating memory on GPU\n");
	cudaError_t cuerr;
	cudaMalloc((void**)&dev_array, sizeof(float) * num_of_floats);

	CHECK_CUDA_ERRORS(cuerr);

	printf("Copying data from host to GPU\n");
	cudaMemcpy(dev_array, array, sizeof(float) * num_of_floats, cudaMemcpyHostToDevice);
	
	CHECK_CUDA_ERRORS(cuerr);

	printf("%.1f Gbs of data successfully transfered to GPU\n", float(num_of_floats * 4) / (1024 * 1024 * 1024));

	printf("Copying data from GPU to host\n");
	cudaMemcpy(array, dev_array, sizeof(float) * num_of_floats, cudaMemcpyDeviceToHost);

	CHECK_CUDA_ERRORS(cuerr);

	printf("%.1f Gbs of data successfully transfered from GPU\n", float(num_of_floats * 4) / (1024 * 1024 * 1024));

	CHECK_CUDA_ERRORS(cuerr);

	printf("Ewerything OK!\n");
	system("pause");
    return 0;
}