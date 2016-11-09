#ifndef __utils_hpp__
#define __utils_hpp__

#define SYNC_AND_CHECK_CUDA_ERRORS {cudaDeviceSynchronize(); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); exit(1); }}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

void CUDAInfo(void) {
	cudaDeviceProp props;
	int dev_count;

	cudaGetDeviceCount(&dev_count);

	printf("Detected %d devices:\n", dev_count);
	for (int i = 0; i < dev_count; ++i) {
		cudaGetDeviceProperties(&props, i);
		printf("\t[ %d ] %s, %.1f GBs memory, CUDA %d.%d Compute Capability\n", i, props.name, float(props.totalGlobalMem) / (1024 * 1024 * 1024), props.major, props.minor);
	}
	printf("\n");
	SYNC_AND_CHECK_CUDA_ERRORS;
}

#endif