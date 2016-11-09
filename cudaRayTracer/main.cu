#define SYNC_AND_CHECK_CUDA_ERRORS {cudaDeviceSynchronize(); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); exit(1); }}

#include <cstdio>
#include <cstdlib>

#include "World.h"
#include "utils.hpp"

int main(void) {
	CUDAInfo();

	World* w;
	cudaMallocManaged(&w, sizeof(World));

	w->init(2 * make_int2(1024, 512), 4);

	printf("Building scene\n\t");
	w->build_scene();
	printf("\tOK\n");

	//Tracing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	
	w->render_scene();

	cudaEventRecord(stop, 0); cudaStreamSynchronize(0);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Tracing time: %.2f ms\n\n", time);
	
	//Saving image
	printf("Saving image\n\t");
	w->save_image("./kek.bmp");
	printf("OK\n");
	w->clear();
	
	cudaFree(w);

	system("pause");
	return 0;
}