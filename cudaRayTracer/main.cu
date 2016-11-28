#ifndef SYNC_AND_CHECK_CUDA_ERRORS
#define SYNC_AND_CHECK_CUDA_ERRORS {cudaStreamSynchronize(0); cudaError_t x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); fclose(stdout); exit(1); }}
#endif


#include <cstdio>
#include <cstdlib>
#include <ctime> 
#include <string>

#include "World.h"
#include "utils.hpp"

int main(void) {
	time_t tm; time(&tm);

	freopen("output.log", "a", stdout);

	time_t start_time; time(&start_time);

	clock_t start_clock = clock();

	struct tm* s_time = localtime(&start_time);

	printf("================= Runnning. time = %04d/%02d/%02d %02d:%02d:%02d =================\n", s_time->tm_year + 1900, s_time->tm_mon + 1, s_time->tm_mday, s_time->tm_hour, s_time->tm_min, s_time->tm_sec);

	CUDAInfo();

	World* w;
	cudaMallocManaged(&w, sizeof(World));

	w->init(1 * make_int2(256, 128), 4, 64);

	printf("Building scene\n\t");
	w->build_scene();
	printf("\tOK\n");

	//Tracing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	
	w->render_scene(64 * 64);

	cudaEventRecord(stop, 0); cudaStreamSynchronize(0);

	float _time;
	cudaEventElapsedTime(&_time, start, stop);
	printf("Tracing time: %.2f ms\n\n", _time);
	
	//Saving image
	w->save_image("./kek.bmp");
	w->clear();
	
	cudaFree(w);
	
	clock_t end_clock = clock();
	printf("TIME ELAPSED: %lf\n", (end_clock - start_clock) / 1000.0);

	fclose(stdout);

	return 0;
}