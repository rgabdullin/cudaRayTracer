
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "helper_math.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERRORS(x) {x = cudaGetLastError(); if ((x) != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(x)); system("pause"); return 1; }}

typedef float(*func)(struct Object, float3);

__declspec(align(128)) struct Object{
	float3 pos;
	float3 vec1;
	float3 vec2;
	float3 n;
	float len1;
	float len2;
	float2 texcoord1;
	float2 texCoord2;
	float2 texcoord3;
	float3 material;
	func* intersection;
};
__device__ void readObj(void* ptr, Object* obj) {
	int idx = threadIdx.x;
	float* target = (float*)obj;
	target[idx] = ((float*)ptr)[idx];
}
__global__ void kernel1(Object* obj, int n) {
	__shared__ Object object[8];
	int block = blockDim.y * blockIdx.y;
	readObj((void*)((char*)obj + sizeof(Object) * (block + threadIdx.y)), &object[threadIdx.y]);
	__syncthreads();
	printf("%2d %2d) %lf\n", threadIdx.x, block + threadIdx.y, ((float*)(&object[threadIdx.y]))[threadIdx.x]);
	return;
}
typedef struct Object Object;
int main()
{
	int n;
	n = 16;
	dim3 blockD(32, 8);
	dim3 gridD(1, n / blockD.y + (n % blockD.y?1:0));
	printf("sizeof(Object) == %d\n",sizeof(Object));
	Object* obj_array = new Object[n];
	for (int i = 0; i < n; ++i) {
		obj_array[i].pos.x = i;
	}
	Object* dev_obj_array;
	
	cudaError_t cuerr;
	cudaMalloc((void**)&dev_obj_array, sizeof(Object) * n);
	cudaMemcpy(dev_obj_array, obj_array, sizeof(Object) * n, cudaMemcpyHostToDevice);

	CHECK_CUDA_ERRORS(cuerr);

	kernel1 <<< gridD, blockD >>> (dev_obj_array, n);

	cudaDeviceSynchronize();

	CHECK_CUDA_ERRORS(cuerr);

	system("pause");
	return 0;
}